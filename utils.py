import os
import httpx
import asyncio
import pdfplumber
import tiktoken
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE = BASE_DIR / "vector_store"

async def generate_embeddings(question: str):	
	timeout = httpx.Timeout(60.0)
	async with httpx.AsyncClient(timeout=timeout) as client:
		uri='https://api.openai.com/v1/embeddings'
		
		api_key = os.environ["OPENAI_API_KEY"]
			
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}
		
		final_embedding_list = []
		payload = {
					"model": "text-embedding-3-small",
					"input": question,
					"encoding_format": "float"
				  }
				  
		retries = 3
		for attempt in range(retries):			  
			response = await client.post(uri,json=payload,headers=headers)
			embeddings_data = response.json()
				
			if response.status_code == 429:
				wait = 2 ** attempt  # 1s, 2s, 4s
				print(f"Rate limited. Waiting {wait}s before retry...")
				await asyncio.sleep(wait)
				continue
					
			if "data" not in embeddings_data:
				raise ValueError(f"Embedding API error: {embeddings_data}")
				
			for item in embeddings_data["data"]:
				final_embedding_list.append(item["embedding"])
			break
		else:
			raise ValueError(f"Embedding API error: {embeddings_data}")
				
		return final_embedding_list[0]
		

async def generate_batch_embeddings(chunks: list[str], batch_size:int=50):
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
				  
			retries = 3
			for attempt in range(retries):	  
				response = await client.post(uri,json=payload,headers=headers)
				embeddings_data = response.json()
				
				if response.status_code == 429:
					wait = 2 ** attempt  # 1s, 2s, 4s
					print(f"Rate limited. Waiting {wait}s before retry...")
					await asyncio.sleep(wait)
					continue
					
				if "data" not in embeddings_data:
					raise ValueError(f"Embedding API error on batch {start}-{end}: {embeddings_data}")
					
				for item in embeddings_data["data"]:
					final_embedding_list.append(item["embedding"])			
				break
			else:
				raise ValueError(f"Batch {start}-{end} failed after {retries} retries")
		return final_embedding_list		


async def fn_extract_text_and_chunk(file:str):
	filename = Path(file).name  # extracts "document.pdf" from full path
	chunk_count = 0
	with pdfplumber.open(file) as pdf:	
		page_text = ''
		for page in pdf.pages:
			page_text += page.extract_text()		
		
		chunks_list = fn_chunk_text(page_text)
	
		list_vectors = await generate_batch_embeddings(chunks_list)

		chunk_count = fn_store_embeddings(chunks_list,list_vectors,filename)

	return chunk_count


def fn_chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
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


def fn_store_embeddings(chunk: list[str], embeddings: list[list[float]], filename:str = "Unknown"):
	chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE))

	# delete existing collection if it exists
	try:
		chroma_client.delete_collection(name="documents3")
	except Exception:
		pass  # collection didn't exist, that's fine

	collection = chroma_client.get_or_create_collection(
		name="documents3",
		metadata={"hnsw:space": "cosine"}
	)
	chunk_index_list = []
	check_id_list = []
	for i, c in enumerate(chunk):
		chunk_index_list.append({"source":f"chunk_{i}", "filename": filename})
		check_id_list.append(f"chunk_{i}")
		
	collection.add(
		documents=chunk,
		embeddings=embeddings,
		metadatas=chunk_index_list,
		ids=check_id_list
	)
	
	#print(f'Total count of stored vectors: {collection.count()}')
	return collection.count()


async def generate_answer(question: str, context_chunks: list[str], conversation_history_lists):				
	context = " ".join(context_chunks)

	timeout = httpx.Timeout(30.0)
	async with httpx.AsyncClient(timeout=timeout) as client:
		uri='https://api.openai.com/v1/chat/completions'
		
		api_key = os.environ["OPENAI_API_KEY"]
		
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}
		

		payload={
			"model":"gpt-4o-mini",
			"messages": [
					{"role": "system", "content":f"You are a helpful assistant. Answer the question using only the context provided. If the answer is not in the context, say so explicitly.\n\n{context}"}					
				]
		}
				
		if len(conversation_history_lists) > 0:	
			for item in conversation_history_lists:
				payload['messages'].append({
				"role": "user", 
				"content": item['question']
				})
				payload['messages'].append({
				"role": "assistant", 
				"content": item['answer']
				})
		
				
		payload['messages'].append({
		"role": "user", 
		"content": question
		})
		
		
		response = await client.post(uri,json=payload,headers=headers)
		
		return response.json()["choices"][0]["message"]["content"]
	