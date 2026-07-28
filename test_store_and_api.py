import os
import httpx
import asyncio
import chromadb

async def test():
    # Get a stored embedding from the collection
    client = chromadb.PersistentClient(path="./vector_store")
    collection = client.get_collection(name="documents2")
    peek = collection.peek(limit=1)
    stored_embedding = peek['embeddings'][0].tolist()
    
    # Generate a fresh embedding for the same text
    async with httpx.AsyncClient() as http:
        api_key = os.environ["OPENAI_API_KEY"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": peek['documents'][0],
            "encoding_format": "float"
        }
        response = await http.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload
        )
        fresh_embedding = response.json()["data"][0]["embedding"]
    
    print(f"Stored first 5: {stored_embedding[:5]}")
    print(f"Fresh first 5:  {fresh_embedding[:5]}")
    
    # Query with fresh embedding
    result = collection.query(
        query_embeddings=[fresh_embedding],
        n_results=3
    )
    print(f"Query with fresh embedding: {result['ids']}")

asyncio.run(test())