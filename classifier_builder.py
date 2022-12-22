"""
https://stackoverflow.com/questions/10515907/document-classification-using-naive-bayes-in-python

https://stackabuse.com/python-for-nlp-creating-tf-idf-model-from-scratch/
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

https://www.reddit.com/r/MLQuestions/comments/pg58wv/bert_model_is_being_beaten_by_random_forest/
"""

import nltk.classify.util
from nltk.probability import LaplaceProbDist
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag as nltk_pos_tag
import numpy as np
import re
import string, os, random
from collections import defaultdict
import pandas as pd
from pickle import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_selection import chi2
import importlib


class Classifier:
    """
    Wrapper for classifiers of various types (NLTK, sklearn, etc.) meant to standardize certain functions
    expected of classifiers, such as:
    * categorize singleton
    * batch categorize
    * confidence percentages for prediction (percentage value for each possible category for given sample)
    * explanation/explication of model (nodes on a decision tree, salient features for bayes, etc.)

    """

    model_obj = None  # actual model/classifier object

    def predict_single(self, text):
        raise NotImplementedError

    def predict_multiple(self, texts):  # may be able to coalesce this with above function
        raise NotImplementedError

    def predict_probabilities(self, text):
        raise NotImplementedError

    def explain_model(self, file=None):
        raise NotImplementedError


emoticons_str = r"""
    (?:
        [:=;]
        [oO\-]?
        [D\)\]\(\]/\\OpP]
    )"""
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    # r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:&amp;)',  # ampersands
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def tokenize_doc(doc_str):
    """
    Takes a string and returns all terms, hastags, and other relevant features in a list.

    :param doc_str: string containing entire document to split into sentences and tokenize
    :return: list (sentences) of lists (words and punctuations)
    """
    return [word_tokenize(t) for t in sent_tokenize(doc_str)]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # return ''
        return wordnet.NOUN


punctuation = list(string.punctuation) + ['‘', '©']
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def normalize(tokens, lemmatize=True, lowercase=True, removestop=True, removepunc=True):
    """
    Process string and turn into list of tokens, emoticons, and relevant features.

    :param lemmatize:
    :param tokens: list of tokens to process
    :param lowercase: force string to lowercase; default TRUE
    :param removestop: remove stopwords and extraneous words from string; default TRUE
    :return: list of tokens in order they appear
    """
    # tokens = tokenize_doc(s)
    if lemmatize:
        tokens = [(w,get_wordnet_pos(t)) for (w,t) in nltk_pos_tag(tokens)]
        tokens = [lemmatizer.lemmatize(w,t) for (w,t) in tokens]
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    if removestop:
        tokens = [token for token in tokens if token not in stop]
    if removepunc:
        tokens = [token for token in tokens if token not in punctuation]
    return tokens


# NLTK helper functions
def fetch_unigrams(tokens):
    #TODO
    """
    Gets list of bigrams from a given list of tokens.

    :param s:  A string to have bigrams taken from
    :return: A list of bigrams in the form of tuples
    """
    #TODO change this to unigrams
    return list(nltk.bigrams(tokens, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))


def fetch_bigrams(tokens):
    """
    Gets list of bigrams from a given list of tokens.

    :param s:  A string to have bigrams taken from
    :return: A list of bigrams in the form of tuples
    """
    # for item in nltk.bigrams(tokens):
    #     bigram_feature_vector.append(item)
    # return bigram_feature_vector
    return list(nltk.bigrams(tokens, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))


def bigram_features(docstring):
    sents = tokenize_doc(docstring)
    doc_bigrams = {} # []
    for sent in sents:
        # add the bigrams for the given sentence to the list of bigrams for the doc
        tokens = normalize(sent)
        doc_bigrams.update({bg:True for bg in fetch_bigrams(tokens)})
        # doc_bigrams.append(fetch_bigrams(tokens))
    return doc_bigrams


# scikit-learn helper functions
def tfidf_tokenizer(docstring):  # TODO verify that sklearn bigrms use sent start/end as features
    sents = tokenize_doc(docstring)
    tokens = []
    for sent in sents:
        tokens.extend(normalize(sent))
    return tokens


def sklearn_tester(doccat_list, ngram_range=(1, 2)):
    # TODO
    # docstrings = [doc for doc,cat in doccat_list]
    df = pd.DataFrame(doccat_list, columns=['text', 'category'])
    df['category_id'] = df['category'].factorize()[0]
    category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    tfidf = TfidfVectorizer(
        sublinear_tf=True  # use a logarithmic form for frequency
        , min_df=5  # min number of documents a word must appear in to be kept
        , max_df=0.95  # max percentage of documents that a feature can appear in to be considered
        , norm='l2'  # feature vectors will have a euclidean norm of 1
        # , encoding='latin-1'
        , ngram_range=ngram_range  # (2) for bigrams, (1, 2) for unigrams and bigrams
        # , stop_words='english'
        , tokenizer=tfidf_tokenizer
    )
    # features = tfidf.fit_transform(docstrings).toarray()  # change this to doctext
    # labels = [cat for doc,cat in doccat_list]
    # cats = {cat for doc,cat in doccat_list}
    features = tfidf.fit_transform(df.text).toarray()
    labels = df.category_id
    from sklearn.feature_selection import chi2
    import numpy as np
    N = 2
    # for cat in sorted(cats):
    #     features_chi2 = chi2(features, labels == cat)
    #     indices = np.argsort(features_chi2[0])
    #     feature_names = np.array(tfidf.get_feature_names())[indices]
    #     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    #     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #     print("# '{}':".format(cat))
    #     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    #     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
    for category, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(category))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    # X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'],
    #                                                     random_state=0)
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # clf = MultinomialNB().fit(X_train_tfidf, y_train)
    #

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    # models = [
    #     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    #     LinearSVC(),
    #     MultinomialNB(),
    #     LogisticRegression(random_state=0),
    # ]
    # CV = 5  # cross validation
    # cv_df = pd.DataFrame(index=range(CV * len(models)))
    # entries = []
    # for model in models:
    #     model_name = model.__class__.__name__
    #     accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    #     for fold_idx, accuracy in enumerate(accuracies):
    #         entries.append((model_name, fold_idx, accuracy))
    # cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    #
    # model = LinearSVC()
    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
    #                                                                                  test_size=0.2, random_state=0)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    # conf_mat = confusion_matrix(y_test, y_pred)


    classifier_old = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=0)
    CV = 5  # cross validation
    cv_df = pd.DataFrame(index=range(CV))
    entries = []
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                     test_size=0.2, random_state=0)

    from sklearn import metrics
    model_name = classifier_old.__class__.__name__
    print(model_name)
    classifier_old.fit(X_train, y_train)
    y_pred_probs = []
    for row in classifier_old.predict_proba(X_test):
        row_probs = {}
        for i, col in enumerate(row):
            row_probs[id_to_category[i]] = col
        y_pred_probs.append(row_probs)
    df_preds = pd.DataFrame(data=y_pred_probs)
    df_preds['actual'] = y_test.array
    df_preds['actual'].replace(to_replace=id_to_category, inplace=True)
    print(df_preds)
    y_pred = classifier_old.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=df['category'].unique()))
    accuracies = cross_val_score(classifier_old, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        print(f"fold:{fold_idx}\taccuracy:{accuracy}")
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    return features


def collect_documents(dir_corpora="./corpora"):
    """
    Given a directory containing manually-categorized documents, read the text of the documents in as
    a string, pair text with a category indicator represented as a tuple ('text', 'cat'), randomize
    the document list for a later test/train split, and return the shuffled list.

    :param dir_corpora: Path to the directory containing the categorized training documents. The
    directory must be organized like this: dir_corpora/category/document.txt, where 'category' is
    a directory containing all the training documents manually identified as belonging to the
    eponymous category and 'document.txt' is a training document txt file to be used in training
    the classification model.
    :return: List of tuples that look like ('document_text', 'category')
    """
    dd_cat_counts = defaultdict(int)

    l_doc_cat = []  # list of (document_text, category) to later be split for test/train
    for dir_obj in os.scandir(dir_corpora):
        if dir_obj.is_dir():
            cat = dir_obj.name
            print(f"Found directory for category {cat}.")
            for file in os.scandir(dir_obj.path):
                if file.name.endswith('.txt'):
                    print(f"\tFound file: {file.name}")
                    dd_cat_counts[cat] += 1
                    with open(f'{file.path}', 'r') as txtfile:
                        text = txtfile.read()
                        l_doc_cat.append((text, cat))
        else:
            continue
    print(dd_cat_counts)
    print("Corpora loaded. Randomizing now...")
    random.Random(1).shuffle(
        l_doc_cat)  # using a seed value so that we always get the same test/train split to test various models
    return l_doc_cat


def generate_features(doc_str, feature_funcs:list):
    # TODO generate features for single document; useful for training and for classifying live data
    pass


def compile_featureset(doccat_list:list, feature_funcs:list=[bigram_features]):
    featureset = []
    print("Extracting features from documents...")
    # for doc in doccat_list:
    #     pass
    for func in feature_funcs:
        featureset.extend([(func(doc),cat) for (doc,cat) in doccat_list])
    return featureset


def plot_confusion_matrix(classifier, testset, cat_vs_tf='cat'):
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    labels = []
    tests = []
    for i, (feats, label) in enumerate(testset):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        labels.append(label)
        tests.append(observed)

    # print(nltk.metrics.confusion_matrix(labels, tests))
    cm = nltk.ConfusionMatrix(labels, tests)
    print(cm)
    return cm


def build_model(featureset, doccat_list=None, model='nb', model_args=None, save:str=None):
    classifier_old = None
    clf = Classifier()
    if model in ['nb', 'dt']:
        print("Establishing a training set...")
        twenty_percent = int(len(featureset) * .2)
        test_set, train_set = featureset[:twenty_percent], featureset[twenty_percent:]
    # TODO try
    if model == 'nb':
        print("Training the Bayes classifier_old...")
        classifier_old = nltk.NaiveBayesClassifier.train(
            train_set
            , estimator=LaplaceProbDist  # Laplace will add 1 to the feature count for all features across each category
        )
        print(nltk.classify.accuracy(classifier_old, test_set))
        print(classifier_old.show_most_informative_features(20))
    elif model == 'dt':
        print("Training the decision tree classifier_old...")
        classifier_old = nltk.DecisionTreeClassifier.train(
            train_set
            # https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
            , depth_cutoff=120
        )
        print(nltk.classify.accuracy(classifier_old, test_set))
        # plot_confusion_matrix(classifier_old, test_set)
        print(classifier_old.pretty_format(depth=20))

    # TODO except
    if save:
        # if not os.path.exists(f"./models/{save}"):
        #     os.makedirs(f"./models/{save}")
        # dump(classifier_old, open(f"./models/{save}/model.pkl", 'wb'))
        # with open(f"./models/{save}/stats.txt", "w") as model_stats_file:
        #     print(nltk.classify.accuracy(classifier_old, test_set), file=model_stats_file)
        #     print(plot_confusion_matrix(classifier_old, test_set), file=model_stats_file)
        if not os.path.exists(f"./classifiers/{save}"):
            os.makedirs(f"./classifiers/{save}")
        dump(clf, open(f"./classifiers/{save}/clf.pkl", 'wb'))
        with open(f"./classifiers/{save}/stats.txt", "w") as model_stats_file:
            # print(nltk.classify.accuracy(classifier_old, test_set), file=model_stats_file)
            # print(plot_confusion_matrix(classifier_old, test_set), file=model_stats_file)
            clf.explain_model(clf, file=model_stats_file)  # write explanation to file

    return clf


class RandomForestClf(Classifier):
    classifier_old = RandomForestClassifier()
    tfidf = TfidfVectorizer()
    df_doc_cat = pd.DataFrame()
    category_to_id = dict()
    id_to_category = dict()
    features = []
    labels = pd.Series()
    X_test = []
    y_test = []

    min_df = 5
    max_df = .95
    ngram_range = (1, 2)
    estimators = 200
    max_depth = 100
    conf_threshold = 0.5
    clf_name = f"rf_est{estimators}_dep{max_depth}_ngram{str(ngram_range)}_min{min_df}_max{max_df}_conf{conf_threshold}"

    def __init__(self, doccat_list, min_df=5, max_df=.95, ngram_range=(1,2), estimators=200, max_depth=100, conf_threshold=.5):
        print("Building a random forest classifier...")
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.estimators = estimators
        self.max_depth = max_depth
        self.conf_threshold = conf_threshold
        self.clf_name = f"rf_est{estimators}_dep{max_depth}_ngram{str(ngram_range)}_min{min_df}_max{max_df}_conf{conf_threshold}"

        self.df_doc_cat = pd.DataFrame(doccat_list, columns=['text', 'category'])
        self.df_doc_cat['category_id'] = self.df_doc_cat['category'].factorize()[0]
        category_id_df = self.df_doc_cat[['category', 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', 'category']].values)
        print("Vectorizing features...")
        self.tfidf = TfidfVectorizer(
            sublinear_tf=True  # use a logarithmic form for frequency
            , min_df=min_df  # min number of documents a word must appear in to be kept
            , max_df=max_df  # max percentage of documents that a feature can appear in to be considered
            , norm='l2'  # feature vectors will have a euclidean norm of 1
            # , encoding='latin-1'
            , ngram_range=ngram_range  # (2,2) for bigrams, (1, 2) for unigrams and bigrams
            # , stop_words='english'
            , tokenizer=tfidf_tokenizer
        )
        self.features = self.tfidf.fit_transform(self.df_doc_cat.text).toarray()
        self.labels = self.df_doc_cat.category_id
        self.classifier_old = RandomForestClassifier(
            n_estimators=estimators,
            max_depth=max_depth,
            random_state=0
        )
        entries = []
        X_train, self.X_test, \
        y_train, self.y_test, \
        indices_train, indices_test = train_test_split(self.features, self.labels,
                                                       self.df_doc_cat.index,
                                                       test_size=0.2, random_state=0)
        model_name = self.classifier_old.__class__.__name__
        # print(model_name)
        print("Training classifier...")
        self.classifier_old.fit(X_train, y_train)

    def categories(self):
        return self.category_to_id.keys()

    def predict_single(self, text):
        df_sample = pd.DataFrame([text], columns=['text'])
        feats_sample = self.tfidf.transform(df_sample.text)
        row = self.classifier_old.predict_proba(feats_sample[0].reshape(1, -1))[0]
        row_probs = {}
        # Confidence Audit
        for i, col in enumerate(row):
            row_probs[self.id_to_category[i]] = col
            # if prob is > threshold, return prediction
            if col > self.conf_threshold:
                return self.id_to_category[i]
        # return "needs manual review" if none was > threshold
        return "needs_manual_review"

    def predict_multiple(self, texts):
        # TODO
        super().predict_multiple(texts)

    def predict_probabilities(self, text):
        # TODO
        super().predict_probabilities(text)

    def explain_model(self, file=None):
        y_pred_probs = []
        for row in self.classifier_old.predict_proba(self.X_test):
            row_probs = {}
            for i, col in enumerate(row):
                row_probs[self.id_to_category[i]] = col
            y_pred_probs.append(row_probs)
        df_preds = pd.DataFrame(data=y_pred_probs)
        df_preds['actual'] = self.y_test.array
        df_preds['actual'].replace(to_replace=self.id_to_category, inplace=True)
        y_pred = self.classifier_old.predict(self.X_test)
        CV = 5  # cross validation folds
        N = 5  # top 2 most correlated n-grams per category
        accuracies = cross_val_score(self.classifier_old, self.features, self.labels, scoring='accuracy', cv=CV)
        summary = f"Random Forest Classifier using the following parameters:\n" \
                  f"\t          estimators: {self.estimators}\n" \
                  f"\t               depth: {self.max_depth}\n" \
                  f"\t        min doc freq: {self.min_df}\n" \
                  f"\t        max doc freq: {self.max_df}\n" \
                  f"\tconfidence threshold: {self.conf_threshold}\n\n"
        summary += f"{str(confusion_matrix(self.y_test, y_pred))}\n"
        summary += metrics.classification_report(self.y_test, y_pred, target_names=self.df_doc_cat['category'].unique())
        summary += '\n'
        # for fold_idx, accuracy in enumerate(accuracies):
        #     summary += f"fold:{fold_idx}\taccuracy:{accuracy}\n"
        for category, category_id in sorted(self.category_to_id.items()):
            features_chi2 = chi2(self.features, self.labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(self.tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            summary += "# '{}':\n".format(category)
            summary += " . Most correlated unigrams:\n  . {}\n".format('\n  . '.join(unigrams[-N:]))
            summary += " . Most correlated bigrams:\n  . {}\n".format('\n  . '.join(bigrams[-N:]))
        if file:
            print(summary, file=file)
        else:
            print(summary)
            # return summary


# class NaiveBayesClf(Classifier):
#     classifier_old = nltk.NaiveBayesClassifier()
#     test_set = []
#     def __init__(self, doccat_list):  # TODO allow for multiple ngram features
#         featureset = compile_featureset(l_doc_cat, [bigram_features])
#         print("Establishing a training set...")
#         twenty_percent = int(len(featureset) * .2)
#         self.test_set, train_set = featureset[:twenty_percent], featureset[twenty_percent:]
#         print("Training the Bayes classifier...")
#         self.classifier_old = nltk.NaiveBayesClassifier.train(
#             train_set
#             , estimator=LaplaceProbDist  # Laplace will add 1 to the feature count for all features across each category
#         )
#
#     def categories(self):
#         return self.classifier_old.labels()
#
#     def explain_model(self, file=None):
#         summary = f"Naive Bayes Classifier using bigrams:\n"  #TODO update ad hoc
#         summary += plot_confusion_matrix(self.classifier_old, self.test_set)
#         summary += "\n"
#         summary += nltk.classify.accuracy(self.classifier_old, self.test_set)
#         summary += "\n"
#         summary += self.classifier_old.show_most_informative_features(20)
#         summary += "\n"
#         if file:
#             print(summary, file=file)
#         else:
#             print(summary)
#             # return summary




class PatternMatchingClf(Classifier):
    #TODO
    """
    Uses regular expressions to match a document to a category.

    Constructor requires a dictionary named 'cat_regex'
    where the keys are the names of the possible categories
    and values are a list of regex strings to
    match on for the respective category.
    """
    cat_regex = dict()
    clf_name = "pm"
    def __init__(self, cat_regex):
        self.cat_regex = cat_regex
        self.clf_name = "pm_"
        self.clf_name += "-".join(cat_regex.keys())


    def categories(self):
        return self.cat_regex.keys()

    def predict_single(self, text):
        # use regex to predict category
        votes_per_cat = dict()
        for cat in self.cat_regex.keys():
            votes_per_cat[cat] = 0
            for pattern in self.cat_regex[cat]:
                results = re.findall(pattern, text)
                votes_per_cat[cat] = votes_per_cat[cat] + len(results)
        return sorted(votes_per_cat, key=votes_per_cat.get, reverse=True)[0]  # return the cat with the most re matches

    def predict_multiple(self, texts):
        # TODO
        super().predict_multiple()

    def predict_probabilities(self, text):
        super().predict_probabilities()

    def explain_model(self, file=None):
        summary = f"Pattern Matching Classifier using the following expressions:\n"
        for cat in self.cat_regex.keys():
            summary += f"\t{cat}\n"
            for pattern in self.cat_regex[cat]:
                summary += f"\t\t{pattern}\n"



def save_clf(clf, save=None):
    if not save:
        save=clf.clf_name
    if not os.path.exists(f"./classifiers/{save}"):
        os.makedirs(f"./classifiers/{save}")
    dump(clf, open(f"./classifiers/{save}/clf.pkl", 'wb'))  # dump classifier to binary
    clf.explain_model(file=open(f"./classifiers/{save}/stats.txt", "w"))
    # with open(f"./classifiers/{save}/stats.txt", "w") as model_stats_file:
    #     clf.explain_model(file=model_stats_file)  # write explanation to file


if __name__ == "__main__":
    l_doc_cat = collect_documents()
    clf = RandomForestClf(doccat_list=l_doc_cat, estimators=200, max_depth=100, ngram_range=(1,2), min_df=10, max_df=.90, conf_threshold=0.7)
    save_clf(clf)
    # bigram_featureset = compile_featureset(l_doc_cat, [bigram_features])
    # build_model(bigram_featureset, model='dt', save="bg_tree_depth120")

