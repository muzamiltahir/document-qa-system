import os
import httpx
import asyncio

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