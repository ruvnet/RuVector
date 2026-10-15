"""Validate real adapter/native collection with deterministic malformed embedding outputs."""
import numpy as np
import pytest
from ruvector import Collection
from ruvector.collection import CollectionError
from langchain_core.embeddings import Embeddings
from ruvector.integrations.langchain import RuVectorStore

class OutputEmbeddings(Embeddings):
    def __init__(self, output): self.output = output
    def embed_documents(self, texts): return self.output
    def embed_query(self, text): return [1., 0.]

@pytest.mark.parametrize("output", [
    [[1., 0.]],
    [[1., 0.], [0., 1.], [1., 1.]],
    [[1., 0.], [0., 1., 2.]],
    [[1., 0.], [float("nan"), 1.]],
    [[1., 0.], [float("inf"), 1.]],
])
def test_rejected_embedding_outputs_preserve_collection_and_id_maps(output):
    coll = Collection.create(dim=2)
    store = RuVectorStore(coll, OutputEmbeddings([[1., 0.]]))
    store.add_texts(["existing"], ids=["existing"])
    before_count = len(coll)
    before_map = dict(store._id_to_int)
    store._embedding = OutputEmbeddings(output)
    with pytest.raises((ValueError, CollectionError)):
        store.add_texts(["first", "second"], ids=["first", "second"])
    assert len(coll) == before_count
    assert store._id_to_int == before_map
    assert store.similarity_search("existing", k=1)[0].page_content == "existing"

def test_valid_embeddings_preserve_returned_ids_and_metadata():
    coll = Collection.create(dim=2)
    store = RuVectorStore(coll, OutputEmbeddings([[1., 0.], [0., 1.]]))
    assert store.add_texts(["first", "second"], ids=["a", "b"], metadatas=[{"n": 1}, {"n": 2}]) == ["a", "b"]
    assert len(coll) == 2
    assert store.similarity_search("first", k=1)[0].metadata == {"n": 1}
