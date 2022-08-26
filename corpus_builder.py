from smart_open import open
import json



class CorpusBuilder:


    text: list[str]


    def __init__(self, data_filepath: str) -> None:
        self.text = []
        self.stream_data(data_filepath)



    def stream_data(self, data_filepath: str) -> list[str]:
        for serialised_record in open(data_filepath):
            deserialised_record = json.loads(serialised_record)
            try:
                locale = deserialised_record["locale"]
                if locale.startswith("en_"):
                    self.text.append(deserialised_record["localized_completed_text"])
            except KeyError:
                record_id = deserialised_record["unique_id"]
                print(f"Record w/o locale -> unique_id: {record_id}")



if __name__ == "__main__":
    data_filepath = "scratchpad/data/common_index_results.json"
    corpus = Corpus(data_filepath)
    corpus.text[:20]


    with open("scratchpad/data/suggestion_names.txt", "w") as out_file:
        for name in corpus.text:
            out_file.write(name + "\n")
