from datasets import load_from_disk
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.collectors import Collector
import os

def load_index(index_path="indexes/hf_last_index"):
    ix = open_dir("indexes/hf_last_index")
    return ix

class HitLimitReached(Exception):
    def __init__(self, hits):
        self.hits = hits
        super().__init__(f"Stopped early after {len(hits)} hits")

class EarlyExitCollector(Collector):
    def __init__(self, max_hits):
        self.max_hits = max_hits
        self.hits = []

    def set_searcher(self, searcher):
        self.searcher = searcher

    def collect(self, docnum):
        if len(self.hits) >= self.max_hits:
            raise HitLimitReached(self.hits)  # Kill the search early
        self.hits.append(docnum)

def create_index(fn="data/preprocessed_data_parsed/last/train.hf",
                 index_dir = "hf_last_index"):
    # Load a Hugging Face dataset (Example: 'ag_news' dataset)
    print("loading dataset")
    dataset = load_from_disk(fn)

    # Define the indexing schema
    schema = Schema(id=ID(stored=True), content=TEXT)  #todo: store true for text

    # Create index directory
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    # Create the index
    index = create_in(index_dir, schema)
    writer = index.writer()

    # Add dataset texts to the index
    for i, row in enumerate(dataset):
        writer.add_document(id=str(i), content=row["program_text"])  # Ensure ID is a string
    writer.commit()

    print("Indexing complete!")
    
def look_up(ix, keyword_str="speed 30\ndot aqua, 1000000", max_hits=500):
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(keyword_str.replace("\n", " "))
        
        collector = EarlyExitCollector(max_hits=max_hits)
        try:
            print(keyword_str.replace("\n", " "))
            searcher.search_with_collector(query, collector)
            results = [searcher.stored_fields(docnum) for docnum in collector.hits]
            print(f"Collected {len(results)} results")
            return len(results)
        except StopIteration:
            print("Stopped early.")
            return max_hits
        except HitLimitReached:
            print("Stopped at limit.")
            return max_hits
                       
#create_index(fn="data/preprocessed_data_parsed/all_downsampled/train.hf",index_dir = "hf_all_downsampled_index")
#create_index(fn="data/preprocessed_data_parsed/synthetic_downsampled/train.hf",index_dir = "hf_synthetic_downsampled_index")
#create_index(fn="data/preprocessed_data_parsed/all/train.hf",index_dir = "hf_all_index")
#create_index(fn="data/preprocessed_data_parsed/synthetic/train.hf",index_dir = "hf_synthetic_index")