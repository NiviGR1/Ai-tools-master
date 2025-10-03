import chromadb
from sentence_transformers import SentenceTransformer

class ResumeDB:
    def __init__(self, persist_directory="./chroma_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="resumes"
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_resume(self, name: str, text: str):
        embedding = self.model.encode([text])[0].tolist()
        self.collection.add(
            ids=[name],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"filename": name}]
        )

    def search(self, query: str, k: int = 3):
        query_emb = self.model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )
        return results
