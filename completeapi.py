import os
import httpx
import asyncio
import json
import chromadb

async def generate_answer(question: str, context_chunks: list[str]):				
	context = " ".join(context_chunks)

	timeout = httpx.Timeout(30.0)
	async with httpx.AsyncClient(timeout=timeout) as client:
		uri='https://api.openai.com/v1/chat/completions'
		
		api_key = os.environ["OPENAI_API_KEY"]
		
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}

		'''
		payload={
			"model":"gpt-4o-mini",
			"messages": [
					#{"role": "system", "content":f"You are a helpful assistant. Answer the question using only the context provided. If the answer is not in the context, say so explicitly.\n\n{context}"},
					{"role": "system", 
	  					"content":f"""you are a helpful assistant answering questions strictly about a phrasal verbs document.
						STRICT RULES:
						- Answer ONLY using the context provided below
						- If the exact answer is not in the context, respond with exactly: "This information is not in the provided context"
						- Do NOT use your general knowledge under any circumstances
						- Do NOT suggest alternatives, synonyms, or related phrases not explicitly mentioned in the context
						- Do NOT apologise or add unnecessary explanation when the answer is not found
						CONTEXT:{context}"""
					},
					{"role": "user", "content": question}
				]
		}
		'''
		payload={
			"model":"gpt-4o-mini",
			"messages": [
					#{"role": "system", "content":f"You are a helpful assistant. Answer the question using only the context provided. If the answer is not in the context, say so explicitly.\n\n{context}"},
					{"role": "system", 
						"content":f"""You are a helpful assistant answering questions about a phrasal verbs document.

							RULES:
							- Answer ONLY using information from the context provided below
							- If the context contains a phrase with a similar or related meaning to what is asked, suggest it
							- If no related phrase exists in the context, respond: "This information is not in the provided context"
							- Do NOT invent phrases not present in the context

						CONTEXT:{context}"""
					},
					{"role": "user", "content": question}
				]
		}
		
		response = await client.post(uri,json=payload,headers=headers)
		
		return response
		
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
				  
		response = await client.post(uri,json=payload,headers=headers)
		embeddings_data = response.json()
		if "data" not in embeddings_data:
			raise ValueError(f"Embedding API error: {embeddings_data}")
			
		for item in embeddings_data["data"]:
			final_embedding_list.append(item["embedding"])
		return final_embedding_list[0]

async def main():
	#question = 'What are the sentences which uses Nail down phrase in them?'
	#question = "Sell something in an auction is related to which phrase?"
	#question = 'Which phrase means Surrender and which sentence is it used in?'
	question = 'Sell something in an auction is related to which phrase?'
	question_embedding = await generate_embeddings(question)
		
	chroma_client = chromadb.PersistentClient(path="./vector_store")
	collection = chroma_client.get_or_create_collection(name="documents")
	
	query_result = collection.query(
		query_embeddings=question_embedding,
		n_results = 3
	)

	retrieved_chunks = query_result['documents'][0]

	print(retrieved_chunks)
	
	resp = await generate_answer(question,retrieved_chunks)
	
	print("Answer:")
	print(resp.json()["choices"][0]["message"]["content"])
	
	metadata = query_result['metadatas'][0]
	distances =  query_result['distances'][0]
	
	print("\nSources:")
	for meta, distance in zip(metadata, distances):
		relevance = 1 - (distance / 2)
		print(f"- {meta['source']} (relevance: {relevance:.2f})")
	
if __name__ == "__main__":
	asyncio.run(main())
