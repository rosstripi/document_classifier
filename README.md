# document_classifier
Categorize any type of raw txt documents using supervised classification based on a manually constructed corpus.

## Usage
The document_classifier project has several scripts to be used at various points in the classification pipeline:
* `corpus_builder.py`: TODO
* `classifier_builder.py`: a library for designing, creating, testing, and storing various text document classifiers (at
time of writing: naive bayes, decision tree, and random forest) by creating a common wrapper for various models from 
different statistical and NLP packages
* `cli_utility.py`: TODO a command line utility allowing users to implement classifiers made using the 
`classifier_builder.py` to categorize new data, storing the output as a .csv file and allowing for accuracy tracking if 
still in development

## Compilation and Deployment
At the moment, classifiers generated with these scripts are utilized via a command line interface (CLI) application.

### Generating a classifier file
The `classifier_builder.py` script is designed to train/build a classification model, wrap it into a python class to 
implement functionality that is common across model types (predict, explain model, list available categories), and 
save that class to a "pickled" binary file called `clf.pkl` along with a document explaining the model and its 
performance called `stats.txt`.

Sample code to generate a random forest model from a corpus of documents and save it to a binary file using the 
`classifier_builder.py` looks like this:
> import classifier_builder as cb
> 
> l_doc_cat = cb.collect_documents()
> 
> clf = cb.RandomForestClf( 
    doccat_list=l_doc_cat,
    estimators=200,
    max_depth=100,
    ngram_range=(1,3),
    min_df=10,
    max_df=.90,
    conf_threshold=0.7
)
> 
> cb.save_clf(clf)

### Compiling the cli_utility into an executable
While you are perfectly fine executing the `cli_utility.py` script from a python interpreter, provided you take the time
to install its dependencies, it usually makes more sense to compile the script to a single executable file for use in a 
user's native desktop environment.

Compiling the `cli_utility` for deployment requires installation of the `pyinstaller` package using `pip`. 
Once installed, you can generate an executable version of the application by opening a terminal, navigating to the 
directory where it is saved, and running the following command:

>pyinstaller -F cli_utility.py

It will then output an executable file to a local directory called `dist`. The type of executable generated will be 
determined by the operating system of the machine used to compile it (e.g. Windows will produce a .exe file). For more 
info on using `pyinstaller`, please visit 
[this post on StackOverflow](https://stackoverflow.com/questions/5458048/how-can-i-make-a-python-script-standalone-executable-to-run-without-any-dependen).

### Deploying the classifier
Once both the binary `clf.pkl` classifier file and the `cli_utility` executable file are generated, they simply need 
to be placed in the same directory. After that, just run the executable!

## TODO

* `corpus_builder.py`
* `cli_utility.py`

### Feature Weighting

Need to add functionality to save/load vectorized features for manual weighing.

### Other Classifier Subclasses

* Naive Bayes
* Decision Tree
* Pattern Matching
