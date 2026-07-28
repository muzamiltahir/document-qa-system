import pdfplumber
import tiktoken
import os
import httpx
import asyncio
import json
import chromadb

def store_embeddings(chunk: list[str], embeddings: list[list[float]]):
	chroma_client = chromadb.PersistentClient(path="./vector_store")
	collection = chroma_client.get_or_create_collection(
		name="documents2",
		metadata={"hnsw:space": "cosine"}
	)
	chunk_index_list = []
	check_id_list = []
	for i in range(len(chunk)):
		chunk_index_list.append({"source":f"chunk_{i}"})
		check_id_list.append(f"chunk_{i}")
		
	collection.add(
		documents=chunk,
		embeddings=embeddings,
		metadatas=chunk_index_list,
		ids=check_id_list
	)
	
	print(f'Total count of stored vectors: {collection.count()}')

async def generate_embeddings(chunks: list[str], batch_size:int=50):
	timeout = httpx.Timeout(120.0)   # 120 seconds
	async with httpx.AsyncClient(timeout=timeout) as client:
		uri='https://api.openai.com/v1/embeddings'
		
		api_key = os.environ["OPENAI_API_KEY"]
			
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}
		
		final_embedding_list = []
		for start in range(0,len(chunks),batch_size):			
			end = min(start+batch_size,len(chunks))
			print(f'Running bacth {start}-{end}')
			
			batch_chunk = chunks[start:end]
			payload = {
					"model": "text-embedding-3-small",
					"input": batch_chunk,
					"encoding_format": "float"
				  }
				  
			response = await client.post(uri,json=payload,headers=headers)
			embeddings_data = response.json()
			if "data" not in embeddings_data:
				raise ValueError(f"Embedding API error on batch {start}-{end}: {embeddings_data}")
			for item in embeddings_data["data"]:
				final_embedding_list.append(item["embedding"])			
				
		return final_embedding_list


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
	encoder = tiktoken.encoding_for_model("gpt-4o-mini")
	tokens = encoder.encode(text)
	
	step = 	chunk_size - overlap
	start=0
	final_list = []
	for start in range(0,len(tokens),step):
		end = min(start+chunk_size,len(tokens))
		chunk_tokens = tokens[start:end]
		final_list.append(encoder.decode(chunk_tokens))
		if end == len(tokens):
			break			
	return final_list
		

async def main():
	loc = r'D:\TechLearning\PythonByClaude\Attachments\Complete-PhrasalVerbs-list.pdf'
	with pdfplumber.open(loc) as pdf:	
		page_text = ''
		for page in pdf.pages:
			page_text += page.extract_text()		
		
		chunks_list = chunk_text(page_text)
	
		list_vectors = await generate_embeddings(chunks_list)
				
		print(f'Dimension Count of First Vector : {len(list_vectors[0])}')
		
		print(f'Total Embeddings returned: {len(list_vectors)}')		
		
		print('Now storing the embeddings')
		
		store_embeddings(chunks_list,list_vectors)
				
asyncio.run(main())			
			