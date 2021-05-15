import re
import string
import spacy
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from enum import Enum
import _pickle as pickle
from xgboost import XGBClassifier
import sys

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer("english")


def normalize_whitespace(text):
    corrected = text
    corrected = re.sub(r"//t", r"\t", corrected)
    corrected = re.sub(r"( )\1+", r"\1", corrected)
    corrected = re.sub(r"(\n)\1+", r"\1", corrected)
    corrected = re.sub(r"(\r)\1+", r"\1", corrected)
    corrected = re.sub(r"(\t)\1+", r"\1", corrected)
    return corrected.strip(" ")


def to_lower(text):
    return text.lower()


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def replace_urls(text):
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "URL", text)
    return text


def remove_non_alphanumeric(text):
    return re.sub(r"[^A-Za-z0-9 ]+", "", text)


def tokenize(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return tokens


def remove_stop_words(tokens):
    return [t for t in tokens if not t in STOP_WORDS]


def stem(tokens):
    return [stemmer.stem(t) for t in tokens]


def normalize_text(text):
    normalizers = [
        replace_urls,
        remove_punctuation,
        remove_numbers,
        remove_non_alphanumeric,
        normalize_whitespace,
        to_lower,
        tokenize,
        remove_stop_words,
        stem,
    ]
    result = text
    for n in normalizers:
        result = n(result)
    return result


class CommitType(Enum):
    BUG = 0
    NONE_BUG = 1


class CommitClassifier:
    def __init__(self):
        setattr(sys.modules["__main__"], "normalize_text", normalize_text)
        with open("assets/vectorizer.pk", "rb") as f:
            self.vectorizer = pickle.load(f)
        self.clf = XGBClassifier(use_label_encoder=False, verbosity=0)
        self.clf.load_model("assets/bug-fix-classifier.model")

    def classify_commit(self, commit_msg) -> CommitType:
        X = self.vectorizer.transform([commit_msg])
        pred = self.clf.predict(X)
        return CommitType(pred[0])
