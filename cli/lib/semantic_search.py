from sentence_transformers import SentenceTransformer
import numpy as np
import os

from .search_utils import CACHE_DIR


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        document_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            document_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(document_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
