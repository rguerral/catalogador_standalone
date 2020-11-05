# Load nltk stopwords
import json
import pathlib
from os.path import join
PATH_src = pathlib.Path(__file__).parent.absolute()
with open(join(PATH_src,'stopwords.json')) as f:
    stopwords = json.load(f)


# Clasificacion
threshold_class = 1
knn_neigh = 5
tfidf_max_features = 10000
tfidf_min_df = 5
tfidf_ngram = 2
tfidf_stopwords = stopwords

# Atributos
escape_chars_begin = "(^|$|\s|\/|,|\.|-)"
escape_chars_end = "(^|$|\s|\/|,|\.|-)"
escape_chars_between = "(\s)?"
number_regex = "(([0-9]{0,10}(\.|\,)?[0-9]+\s?/\s?)?([0-9]{0,10}(\.|\,)?[0-9]+))"
optional_plural_regex = "(s|es)?"
