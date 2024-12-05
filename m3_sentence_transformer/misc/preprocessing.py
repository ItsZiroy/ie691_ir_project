import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymystem3 import Mystem
from string import punctuation

nltk.download("stopwords")
nltk.download("wordnet")

def get_lemmatizer(language):
    if language == "russian":
        return Mystem()
    elif language == "english":
        return WordNetLemmatizer()
    else:
        raise ValueError(f"Lemmatizer for language '{language}' is not implemented")

class TextPreprocessor:
    def __init__(self, language):
        self.language = language
        self.stopwords = stopwords.words(language)
        self.punctuation = punctuation
        self.lemmatizer = get_lemmatizer(language)

    def preprocess_text(self, text):
        if self.language == "russian":
            tokens = self.lemmatizer.lemmatize(text.lower())
        elif self.language == "english":
            tokens = nltk.word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        tokens = [token for token in tokens if token not in self.stopwords \
                  and token != " " \
                  and token.strip() not in self.punctuation]

        return " ".join(tokens)