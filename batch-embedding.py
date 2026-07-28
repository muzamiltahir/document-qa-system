import pdfplumber
import tiktoken
import os
import httpx
import asyncio
import json
import utils


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
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
	
		list_vectors = await utils.generate_batch_embeddings(chunks_list)
		
		print(f'Dimension Count of First Vector : {len(list_vectors[0])}')
		
		print(f'Total Embeddings returned: {len(list_vectors)}')
				
asyncio.run(main())			
			