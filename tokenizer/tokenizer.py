import unicodedata
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


if not 'NLTK_DATA' in os.environ:
    print("Please export NLTK_DATA environment variable")
    nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
else:
    print("NLTK_DATA:", os.environ['NLTK_DATA'])

### global variables

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
punctutation_categories = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    '''
    Description: split string into cleand tokens processed by lemmatization
                 and stemming 

    Arguments:
        text (str):     text to parse

    Returns:
        tokens ([str]): derived tokens from text string
    '''
    text = text.lower()
    # remove urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, '')
    # remove punctuations
    #text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(char for char in text if unicodedata.category(
        char) not in punctutation_categories)
    # tokenize text
    tokens = word_tokenize(text)
    # Reduce words to their stems
    tokens = [stemmer.stem(t) for t in tokens]
    # Reduce words to their root form and remove stop words
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    # Lemmatize verbs by specifying pos
    tokens = [lemmatizer.lemmatize(t, pos='v')
              for t in tokens if t not in stop_words]
    # remove tokens with less than 2 characters
    tokens = [t for t in tokens if len(t) > 2]
    return tokens
