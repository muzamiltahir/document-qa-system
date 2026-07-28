import os
import chromadb

path = "./vector_store"
print(f"Absolute path: {os.path.abspath(path)}")

client = chromadb.PersistentClient(path=path)
collections = client.list_collections()
print(f"Collections: {collections}")

collection = client.get_collection(name="documents2")
print(f"Count: {collection.count()}")

peek = collection.peek(limit=1)
stored_embedding = peek['embeddings'][0].tolist()
test_result = collection.query(
    query_embeddings=[stored_embedding],
    n_results=3
)
print(f"Test query result ids: {test_result['ids']}")