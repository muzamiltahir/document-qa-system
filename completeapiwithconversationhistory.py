import os
import httpx
import asyncio
import json
import chromadb
import sys
import utils

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
		
		return response
		


async def main():
	question = ''
	conversation_history_lists = []	
			
	chroma_client = chromadb.PersistentClient(path="./vector_store")
	collection = chroma_client.get_or_create_collection(name="documents")
		
	while True:
		print('Write down your question. Write quit if you want to stop!')
		question = input('>')
		
		if question.strip() == '':
			continue
		
		if question == 'quit':
			sys.exit()
				
		
		question_embedding = await utils.generate_embeddings(question)
		
		query_result = collection.query(
			query_embeddings=question_embedding,
			n_results = 3
		)

		retrieved_chunks = query_result['documents'][0]
		
		resp = await generate_answer(question,retrieved_chunks,conversation_history_lists)
		
		conversation_history_lists.append({
		'question':question,
		'answer':resp.json()["choices"][0]["message"]["content"]
		})
		
		print("Answer:")
		print(resp.json()["choices"][0]["message"]["content"])
		
		metadata = query_result['metadatas'][0]
		distances =  query_result['distances'][0]
		
		print("\nSources:")
		for meta, distance in zip(metadata, distances):
			relevance = 1 - (distance / 2)
			print(f"- {meta['source']} (relevance: {relevance:.2f})")
		
	
asyncio.run(main())
