from importlib.metadata import SelectableGroups
from stopwords import stopwords_english

from gensim.models.phrases import Phrases, FrozenPhrases, ENGLISH_CONNECTOR_WORDS

from collections import OrderedDict, defaultdict


class MultiWordExpressionExtractor:

    stopwords: list[str]  = []
    connector_words: list[str] = []
    whitelist: list[str] = []

    
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.phrases_model: Phrases = None
        self.phrases: OrderedDict = None
        self.mwe_blacklist: list[str] = None
        self.top_ngrams: dict[str: list] = None


    def extract_mwe(self, min_count: int=10) -> None:
        phrases_model = Phrases(
            sentences=self.corpus,
            min_count=min_count,
            threshold=0,
            scoring="npmi",
            connector_words=MultiWordExpressionExtractor.connector_words)
        self.phrases_model = phrases_model


    def export_mwe(self) -> None:
        mwe_export_sorted = {}
        if self.phrases_model:
            mwe_export = self.phrases_model.export_phrases()
            mwe_export_sorted = MultiWordExpressionExtractor.sort_dict(mwe_export)
        self.phrases = mwe_export_sorted


    def blacklist_mwe(self) -> None:
        blacklist = []
        if self.phrases:
            for mwe in self.phrases.keys():
                terms = mwe.split("_")
                if len(terms) > 1:
                    if terms[0] in self.stopwords or terms[-1] in self.stopwords:
                        blacklist.append(mwe)
        self.mwe_blacklist = blacklist


    def remove_blacklisted_mwe(self) -> None:
        if self.mwe_blacklist:
            for mwe in self.mwe_blacklist:
                if mwe not in self.whitelist:
                    try:
                        del self.phrases_model.vocab[mwe]
                        self.phrases.pop(mwe, None)
                    except KeyError:
                        pass


    def tokenise_mwe(self) -> None:
        if self.phrases_model:
            corpus_mwe: list[list[str]] = []
            for doc in self.corpus:
                doc_mwe = self.phrases_model[doc]
                corpus_mwe.append(doc_mwe)
            self.corpus = corpus_mwe  


    def apply_pipeline(self, tasks: list[str]) -> None:
        [getattr(self, method)() for method in tasks]


    def get_top_ngrams(self, top_n: int = None, ngram_size: list[int] = [1,2,3], min_freq: int = 10) -> dict:
        top_ngrams = defaultdict(list)
        vocab_sorted = MultiWordExpressionExtractor.sort_dict(self.phrases_model.vocab)
        if top_n:
            vocab_top_n = list(vocab_sorted.items())[:top_n]
        else: 
            vocab_top_n = list(vocab_sorted.items())
        for term, freq in vocab_top_n:
            if freq >= min_freq:
                term_ngram_size = term.count("_") + 1
                if term_ngram_size in ngram_size:
                    top_ngrams[f"ngram_{term_ngram_size}"].append((term, freq))
        for ngram_key in top_ngrams.keys():
            top_ngrams[ngram_key].sort(key=lambda item: (len(item[0]), item[0]))
        self.top_ngrams = top_ngrams


    def remove_stopwords_from_unigrams():
        pass


    @staticmethod
    def sort_dict(input_dict: dict, reversed: bool = True) -> dict:
        sorted_dict = OrderedDict(
            sorted(
                input_dict.items(), key=lambda kv: kv[1], reverse=reversed
                )
            )
        return sorted_dict


if __name__ == "__main__":
    corpus_filepath = "data/test_data.txt"
    corpus = []
    with open(corpus_filepath, "r") as input_file:
        for line in input_file:
            corpus.append(line.strip().split())
    corpus[:10]

    stopwords = ["and", "&", "in", "of", "the", "to", "for", "a", "at", "with", "all"]
    connector_words = ["in", "of", "the", "to", "for", "a", "at", "with", "all"]
    whitelist = ["walk_in", "walk_ins", "all_you_can_eat"]

    mwe_extractor = MultiWordExpressionExtractor(corpus)
    mwe_extractor.stopwords = stopwords
    mwe_extractor.connector_words = connector_words
    mwe_extractor.whitelist = whitelist

    tasks = [
        "extract_mwe", 
        # "export_mwe", 
        # "blacklist_mwe",
        # "remove_blacklisted_mwe",
        "tokenise_mwe",
        "extract_mwe",
        "export_mwe", 
        "blacklist_mwe",
        "remove_blacklisted_mwe"
        ]
    
    mwe_extractor.apply_pipeline(tasks)
    mwe_extractor.get_top_ngrams()
    mwe_extractor.top_ngrams
