from stopwords import stopwords_english

from gensim.models.phrases import Phrases, FrozenPhrases, ENGLISH_CONNECTOR_WORDS

from collections import OrderedDict


class MultiWordExpressionExtractor:

    stopwords: list[str]  = ["and", "&"] 
    connector_words: list[str] = []

    
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.phrases_model: Phrases = None


    def extract_mwe(self, min_count: int=10) -> Phrases:
        phrases_model = Phrases(
            sentences=self.corpus,
            min_count=min_count,
            threshold=0,
            scoring="npmi",
            connector_words=MultiWordExpressionExtractor.connector_words)
        self.phrases_model = phrases_model


    def export_mwe(self) -> dict:
        mwe_export = self.phrases_model.export_phrases()
        mwe_export_sorted = MultiWordExpressionExtractor.sort_mwe_export(mwe_export)
        return mwe_export_sorted


    def identify_mwe_with_leading_and_trailing_stopwords(self, mwe_export: dict) -> list[str]:
        mwe_blacklist = []
        for mwe in mwe_export.keys():
            terms = mwe.split("_")
            if len(terms) > 1:
                if terms[0] in self.stopwords or terms[-1] in self.stopwords:
                    mwe_blacklist.append(mwe)
        return mwe_blacklist


    def remove_blacklisted_mwe(self, mwe_blacklist: list[str]) -> None:
        if len(mwe_blacklist) > 0:
            for mwe in mwe_blacklist:
                try:
                    del self.phrases_model.vocab[mwe]
                except KeyError:
                    pass


    def tokenise_mwe(self) -> None:
        corpus_mwe: list[list[str]] = []
        for doc in self.corpus:
            doc_mwe = self.phrases_model[doc]
            corpus_mwe.append(doc_mwe)
        self.corpus = corpus_mwe  

    
    @staticmethod
    def sort_mwe_export(mwe_export: dict, reversed: bool = True) -> dict:
        mwe_sorted = OrderedDict(
            sorted(
                mwe_export.items(), key=lambda kv: kv[1], reverse=reversed
                )
            )
        return mwe_sorted


if __name__ == "__main__":
    corpus_filepath = "data/test_data.txt"
    corpus = []
    with open(corpus_filepath, "r") as input_file:
        for line in input_file:
            corpus.append(line.strip().split())
    
    mwe_extractor = MultiWordExpressionExtractor(corpus)
    # mwe_extractor.corpus[0][:20]

    mwe_extractor.extract_mwe()
    # mwe_extractor.phrases_model.vocab
    len(mwe_extractor.phrases_model.vocab) # 138,579
    len([mwe for mwe in mwe_extractor.phrases_model.vocab if "_" in mwe]) # 112,251

    mwe_export = mwe_extractor.export_mwe()
    len(mwe_export) # 2,366
    mwe_blacklist = mwe_extractor.identify_mwe_with_leading_and_trailing_stopwords(mwe_export)
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

    mwe_blacklist = mwe_extractor.identify_mwe_with_leading_and_trailing_stopwords(mwe_export)
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