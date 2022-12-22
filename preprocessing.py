import random
import re  #, pyap
# import address_re as pyap_re
from spellchecker import SpellChecker
from classifier_builder import tokenize_doc
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os, shutil
from collections import defaultdict

re_garbage = [
    r'<[^>]+>',  # HTML tags
    r'\$(\d{1,3},?){0,}(\d{1,3})(\.\d{2})?',  # dollar values
    r'(?i)((account|acct) ?(#|num|number|no.?)? ?\d{7,}(v\d+)?)|(\d{7,}v\d+)',  # MLX account numbers
    r'.*-{2,}.*\n',  # lines containing dashes, usually section breaks
    r'(?i)(from|to|date|cc|bcc): ?.*\n{1,2}|subject: ?(fwd:? ?)*',  # email headers but keeping subject line
    r'(?i)(re|via): ?.*',  # re/via lines
    r'(?i)(fax|facsimile).*',  # any line that starts with "FAX"
    r'(?i)(phone|tel\.?)?:? ?(\+?\d{1,2})?[ .-]?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}',  # phone numbers
    r'(?i)((p(\.|,)? ?(o|0|q)(\.|,)? ?box ?#?\d+|\d{1,}( \w+\.?){1,5}|(ste|suite|unit|building|bld|bldg)\.? ?#? ?\d+),? ?\n{0,2}){1,2}\w+, ?(\w+(\.| )?){2},? ?\n{0,2}\d{5}(-\d{4})?',  # mailing address  #TODO check this against edge cases
    # pyap_re.full_address,  # full address re from the pyap module
    r'(?i).*page.?break.*',  # pagebreak
    r'(?i)(http[s]?://)?([a-z0-9$\-_@&+!*\(\),]{2,}\.)+[a-z]{2,}(/[a-z0-9$\-_@&+!*\(\),]{1,}\.?)*'  # URLs

]


def garbage_removal(s: str):
    new_str = s[:]
    # for address in pyap.parse(new_str, country='US'):
    #     new_str = new_str.replace(str(address), '')
    for exp in re_garbage:
        # c_exp = re.compile(exp, re.IGNORECASE)
        # loc = c_exp.search(test_str)
        temp_str = re.sub(exp, '', new_str)
        new_str = temp_str[:]
    return new_str


spell = SpellChecker(distance=1, case_sensitive=True)


def fix_ocr_spelling(doc_text):  # fix spelling of doc string
    tokens = tokenize_doc(doc_text)
    fixed_sents = []
    for sent in tokens:
        fixed_words = []
        for word in sent:
            try:
                fixed_words.append(spell.correction(word))
            except:
                fixed_words.append(word)
        fixed_sents.append(fixed_words)
    return ' '.join([TreebankWordDetokenizer().detokenize(words) for words in tokens])


def clean_corpora(input_dir="./corpora", output_dir="./cleaned_corpora"):
    for dir_obj in os.scandir(input_dir):
        if dir_obj.is_dir():
            cat = dir_obj.name
            print(f"Found directory for category {cat}.")
            for file in os.scandir(dir_obj.path):
                if file.name.endswith('.txt'):
                    print(f"\t cleaning file: {file.name}")
                    with open(file.path, 'r') as doc_file:
                        doc_text = doc_file.read()
                        # TODO clean garbage before spellcheck
                        doc_text = garbage_removal(doc_text)
                        doc_text = fix_ocr_spelling(doc_text)
                        if not os.path.exists(f"{output_dir}/{cat}"):
                            os.makedirs(f"{output_dir}/{cat}")
                        with open(f"{output_dir}/{cat}/{file.name}", 'w') as cleaned_file:
                            print(doc_text, file=cleaned_file)


# TODO redo this method
def sample_corpora(input_dir="./corpora", output_dir="./sampled_corpora", clean=False):
    # dd_cat_counts = defaultdict(int)
    dd_doc_files = defaultdict(list)
    # for cat in categories:
    #     dir_cat = f"{input_dir}/{cat}"
    for dir_obj in os.scandir(input_dir):
        if dir_obj.is_dir():
            cat = dir_obj.name
            print(f"Found directory for category {cat}.")
            for file in os.scandir(dir_obj.path):
                if file.name.endswith('.txt'):
                    dd_doc_files[cat].append(f"{file.path}")
        # for file in os.scandir(dir_cat):
        #     if not file.is_file():
        #         continue
        #     elif file.name.endswith('.txt'):
        #         # print(f"Found file: {file.name}. Processing now.")
        #         # dd_cat_counts[cat] += 1
        #         dd_doc_files[cat].append(f"{file.name}")
        #         # with open(f'{dir_cat}/{file.name}', 'r') as txtfile:
        #         #     text = txtfile.read()
        #         #     # already_classified_bigrams.append((fetch_bigrams(text), cat))
        #         #     l_doc_cat.append((text, cat))
    print({key:len(dd_doc_files[key]) for key in dd_doc_files.keys()})
    # output_dir = './cleaned_corpora'
    for key in dd_doc_files.keys():
        if not os.path.exists(f"{output_dir}/{key}"):
            os.makedirs(f"{output_dir}/{key}")
        # print(len(dd_doc_files[key]))
        # if len(dd_doc_files[key]) > 100:
        #     random.Random(2).shuffle(dd_doc_files[key])
        #     dd_doc_files[key] = dd_doc_files[key][:100]
        if clean:  # check spelling, remove garbage, and save text to file
            for doc_fn in dd_doc_files[key]:
                # doc_text = ""
                with open(f"{input_dir}/{key}/{doc_fn}", 'r') as doc_file:
                    # global doc_text
                    doc_text = doc_file.read()
                    #TODO clean garbage before spellcheck
                    doc_text = garbage_removal(doc_text)
                    doc_text = fix_ocr_spelling(doc_text)

                    with open(f"{output_dir}/{key}/{doc_fn}", 'w') as cleaned_file:
                        print(doc_text, file=cleaned_file)
                    # fn = f"{doc_fn[doc_fn.rfind('/')+1:]}"


                pass
        else:  # leave content as is, then move to new folder
            # if not os.path.exists(f"{output_dir}/{key}"):
            #     os.makedirs(f"{output_dir}/{key}")
            shutil.copytree(f"{input_dir}/{key}", f"{output_dir}/{key}")
            #TODO this isn't correct
    print({key: len(dd_doc_files[key]) for key in dd_doc_files.keys()})


test_str = """
"""


def main():
    sample_corpora(clean=True)
    pass

main()