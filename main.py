# import preprocessing
import classifier_builder as cb
from pickle import load


l_doc_cat = cb.collect_documents()
clf = cb.RandomForestClf(
    doccat_list=l_doc_cat,
    estimators=200,
    max_depth=100,
    ngram_range=(1,3),
    min_df=10,
    max_df=.90,
    conf_threshold=0.7
)
cb.save_clf(clf)

# from cat_patterns import cat_regex
# clf = cb.PatternMatchingClf(cat_regex)
# cb.save_clf(clf)

# l_doc_cat = classifier_builder.collect_documents()
# bigram_featureset = classifier_builder.compile_featureset(l_doc_cat, [classifier_builder.bigram_features])
# # classifier.build_model(bigram_featureset, model='classifier', save="refactor.pkl")
# twenty_percent = int(len(bigram_featureset) * .2)
# test_set, train_set = bigram_featureset[:twenty_percent], bigram_featureset[twenty_percent:]
#
# classifier = load(open("bg_8020split_tree_nosample.pkl", 'rb'))
#
# classifier_builder.plot_confusion_matrix(classifier, test_set)
# print(classifier.pretty_format(depth=20))