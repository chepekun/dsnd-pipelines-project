import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# ! python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


class TextProcessor(BaseEstimator, TransformerMixin):
    """Extracts natural language features from the review title and text"""

    positive_sentiment = [
        "great",
        "beautiful",
        "cute",
        "nice",
        "lovely",
        "comfortable",
        "gorgeous",
        "perfect",
        "pretty",
        "flattering",
        "good",
        "comfy",
        "amazing",
        "adorable",
        "stylish",
        "cozy",
        "fun",
        "perfection",
        "perfect",
        "awesome",
    ]
    negative_sentiment = [
        "poor",
        "meh",
        "boxy",
        "scratchy",
        "shapeless",
        "odd",
        "thin",
        "weird",
        "disapointing",
        "awkward",
        "bummer",
        "unflattering",
        "bad",
        "strange",
        "itchy",
        "cheaply",
        "cheap",
        "stiff",
        "minus",
    ]
    mixed_sentiment = ["but", "yet", "or"]

    def __init__(self):
        self.columns_ = []
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = pd.concat(
            [
                X,
                X["Title"].apply(lambda txt: self._process_text("title", txt)).apply(pd.Series),
                X["Review Text"].apply(lambda txt: self._process_text("review_text", txt)).apply(pd.Series),
            ],
            axis=0,
            ignore_index=True,
        )
        self.columns_ = result.columns
        return result

    def _process_text(self, input_field: str, input_text: str):
        doc = nlp(input_text)

        num_pronoums = 0
        num_adjectives = 0
        num_verbs = 0
        num_nouns = 0
        num_conjunctions = 0
        num_numbers = 0
        num_symbols = 0

        positive_count = 0
        negative_count = 0
        mixed_count = 0

        for token in doc:
            if token.pos_ == "PROPN":
                num_pronoums += 1
            if token.pos_ == "ADJ":
                num_adjectives += 1
            if token.pos_ == "VERB":
                num_verbs += 1
            if token.pos_ == "NOUN":
                num_nouns += 1
            if token.pos_ == "CCONJ":
                num_conjunctions += 1
            if token.pos_ == "NUM":
                num_numbers += 1
            if token.pos_ == "SYM":
                num_symbols += 1

            if token.pos_ in ["ADJ", "PROPN", "CCONJ"]:
                if token.lemma_.lower() in self.positive_sentiment:
                    positive_count += 1
                elif token.lemma_.lower() in self.negative_sentiment:
                    negative_count += 1
                elif token.lemma_.lower() in self.mixed_sentiment:
                    mixed_count += 1

        return {
            # special characters
            input_field + "_char_point": input_text.count("."),
            input_field + "_char_comma": input_text.count(","),
            input_field + "_char_semicolon": input_text.count(";"),
            input_field + "_char_quotes": input_text.count('"') + input_text.count("'"),
            input_field + "_char_exclamation": input_text.count("!"),
            input_field + "_char_question": input_text.count("?"),
            input_field + "_char_hashtag": input_text.count("#"),
            input_field + "_char_ellipsis": input_text.count("...") + input_text.count(".."),
            # NLP
            input_field + "_nlp_tokens": len(doc),
            input_field + "_nlp_sentences": len(list(doc.sents)),
            input_field + "_nlp_pronoums": num_pronoums,
            input_field + "_nlp_adjectives": num_adjectives,
            input_field + "_nlp_verbs": num_verbs,
            input_field + "_nlp_nouns": num_nouns,
            input_field + "_nlp_conjunctions": num_conjunctions,
            input_field + "_nlp_numbers": num_numbers,
            input_field + "_nlp_symbols": num_symbols,
            # Sentiment
            input_field + "_sentiment_positive": positive_count,
            input_field + "_sentiment_negative": negative_count,
            input_field + "_sentiment_mixed": mixed_count,
        }
