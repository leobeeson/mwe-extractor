from importlib.metadata import SelectableGroups
from stopwords import stopwords_english

from gensim.models.phrases import Phrases, FrozenPhrases, ENGLISH_CONNECTOR_WORDS

from collections import OrderedDict, defaultdict


class MultiWordExpressionExtractor:

    stopwords: list[str]  = ["and", "&"] 
    connector_words: list[str] = []

    
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.phrases_model: Phrases = None
        self.phrases: OrderedDict = None
        self.blacklist: list[str] = None
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
        self.blacklist = blacklist


    def remove_blacklisted_mwe(self) -> None:
        if self.blacklist:
            if len(self.blacklist) > 0:
                for mwe in self.blacklist:
                    try:
                        del self.phrases_model.vocab[mwe]
                        self.phrases.pop(mwe, None)
                    except KeyError:
                        print("MWE missing in vocab or phrases export: {mwe}")
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


    def get_top_ngrams(self, top_n: int, ngram_size: list[int], min_freq: int = 2) -> dict:
        top_ngrams = defaultdict(list)
        vocab_sorted = MultiWordExpressionExtractor.sort_dict(self.phrases_model.vocab)
        vocab_top_n = list(vocab_sorted.items())[:top_n]
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

    mwe_extractor = MultiWordExpressionExtractor(corpus)

    tasks = [
        "extract_mwe", 
        "tokenise_mwe",
        "extract_mwe",
        "export_mwe", 
        "blacklist_mwe",
        "remove_blacklisted_mwe"
        ]
    
    mwe_extractor.apply_pipeline(tasks)
    mwe_extractor.get_top_ngrams(1000, [1, 2, 3])
    mwe_extractor.top_ngrams

    mwe_extractor.phrases_model
    list(mwe_extractor.phrases_model.vocab.items())[0:10]
    list(mwe_extractor.phrases.keys())[0:10]
    mwe_extractor.blacklist[0]
    [key for key in mwe_extractor.phrases.keys() if key == mwe_extractor.blacklist[0]]

    mwe_extractor.phrases.pop("palo_alto")
    type(mwe_extractor.phrases_model.vocab)
    mwe_extractor.phrases_model.vocab.pop
    


    # mwe_extractor.corpus[0][:20]

    mwe_extractor.extract_mwe()
    # mwe_extractor.phrases_model.vocab
    len(mwe_extractor.phrases_model.vocab) # 138,579
    len([mwe for mwe in mwe_extractor.phrases_model.vocab if "_" in mwe]) # 112,251

    mwe_export = mwe_extractor.export_mwe()
    len(mwe_export) # 2,366
    mwe_blacklist = mwe_extractor.blacklist_mwe(mwe_export)
    len(mwe_blacklist) # 34
    
    # mwe_extractor.remove_blacklisted_mwe(mwe_blacklist)
    mwe_export = mwe_extractor.export_mwe()
    len(mwe_export) # 2,332
    len([mwe for mwe in mwe_extractor.phrases_model.vocab if "_" in mwe]) # 112,217
    

    len([mwe for doc in mwe_extractor.corpus for mwe in doc if "_" in mwe]) # 0
    mwe_extractor.tokenise_mwe()
    len([mwe for doc in mwe_extractor.corpus for mwe in doc if "_" in mwe]) # 52,372
    len([mwe for doc in mwe_extractor.corpus for mwe in doc if "role_and_you" in mwe]) # 0
    
    mwe_extractor.corpus[100:200]
    mwe_extractor.phrases_model.vocab

    ### SECOND ROUND ###
    mwe_extractor.extract_mwe()
    len(mwe_extractor.phrases_model.vocab) # 151,530
    len([mwe for mwe in mwe_extractor.phrases_model.vocab if "_" in mwe]) # 125,227
    mwe_export = mwe_extractor.export_mwe()
    len(mwe_export) # 2,183
    'fish_and_chips' in mwe_export.keys() # False

    mwe_blacklist = mwe_extractor.blacklist_mwe(mwe_export)
    len(mwe_blacklist) # 34

    mwe_extractor.remove_blacklisted_mwe(mwe_blacklist)
    len(mwe_extractor.phrases_model.vocab) # 151,496
    len([mwe for mwe in mwe_extractor.phrases_model.vocab if "_" in mwe]) # 125,193
    mwe_export = mwe_extractor.export_mwe()
    len(mwe_export) # 2,149
    type(mwe_export)
    
    stopwords = ["and", "&"]
    
    def identify_mwe_with_leading_and_trailing_stopwords(mwe_export: dict) -> list[str]:
        mwe_blacklist = []
        for mwe in mwe_export.keys():
            terms = mwe.split("_")
            if len(terms) > 1:
                if terms[0] in stopwords or terms[-1] in stopwords:
                    mwe_blacklist.append(mwe)
        return mwe_blacklist
    
    mwe_blacklist = identify_mwe_with_leading_and_trailing_stopwords(mwe_export)
    mwe_extractor.remove_blacklisted_mwe(mwe_blacklist)
    mwe_export = mwe_extractor.export_mwe()
    len(mwe_export) # 2,069

    final_mwe = []
    for k in mwe_export.keys():
        final_mwe.append(k)

    with open("data/mwe_output.txt", "w") as output_file:
        for mwe in final_mwe:
            print(mwe)
            output_file.append(mwe + "\n")


    len(mwe_extractor.phrases_model.vocab) # 151,496
    mwe_extractor.phrases_model.vocab


    vocab = {}
    for term, freq in mwe_extractor.phrases_model.vocab.items():
        if freq > 10:
            vocab[term] = freq
    
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    len(vocab_sorted) # 5,991
    vocab_sorted
    
    bichars = []
    for k,v in vocab_sorted:
        if len(k) < 3:
            bichars.append(k)
    len(bichars) # 174

    trichars = []
    for k,v in vocab_sorted:
        if len(k) == 3:
            trichars.append(k)
    len(trichars) # 364

    fourchars = []
    for k,v in vocab_sorted:
        if len(k) == 4:
            fourchars.append(k)
    len(fourchars) # 700

    counter = 0
    n = 500
    top_unigrams = []
    for k,v in vocab_sorted:
        if len(k) > 4:
            if "_" not in k:
                top_unigrams.append(k)
                counter += 1
        if counter > n:
            break