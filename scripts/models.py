
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]




#Create a function to do the following Cleanup Respectively
# remove punctuation
# remove whitespace between terms with single space
# remove leading and trailing whitespace
#all_df['text'] = all_df['text'].str.replace(r'^\s+|\s*?$', ' ')
# change words to lower case
def clean_text(txt):
    clean_txt=txt.replace(r'[^\w\d\s]', ' ')
    clean_txt=clean_txt.replace(r'\s+', ' ')
    clean_txt=clean_txt.replace(r'^\s+|\s*?$', ' ')
    clean_txt=clean_txt.strip().lower()
    return clean_txt
class cleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

#we will also create a function that tokenizes our texts and removes stop words
from nltk.corpus import stopwords
def tokenizer(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence=[w for w in word_tokens if w not in stop_words]
    return filtered_sentence