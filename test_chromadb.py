import chromadb
import numpy as np

client = chromadb.PersistentClient(path="./vector_store")
collection = client.get_or_create_collection(name="documents2")

# Get the actual stored embedding of chunk_0
peek = collection.peek(limit=1)
stored_embedding = peek['embeddings'][0].tolist()

print(f"Stored embedding length: {len(stored_embedding)}")

# Query using that exact embedding — should return chunk_0 as top result
result = collection.query(
    query_embeddings=[stored_embedding],
    n_results=3
)
print(result)