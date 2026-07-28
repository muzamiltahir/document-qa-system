import pdfplumber
import tiktoken


loc = r'D:\TechLearning\PythonByClaude\Attachments\Complete-PhrasalVerbs-list.pdf'
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
		

with pdfplumber.open(loc) as pdf:	
	page_text = ''
	for page in pdf.pages:
		page_text += page.extract_text()		
	
	chunks_list = chunk_text(page_text)
		
	print(f"Total chunks: {len(chunks_list)}")
	print(f"Chunk 1 preview: {chunks_list[0]}")
	print(f"Chunk 2 preview: {chunks_list[1]}")
	print(f"Chunk 1 preview: {chunks_list[0][-200:]}")
	print(f"Chunk 2 preview: {chunks_list[1][:200]}")
	
	'''
	encoder = tiktoken.encoding_for_model("gpt-4o-mini")
	
	chunk1_tokens = encoder.encode(chunks_list[0])
	chunk2_tokens = encoder.encode(chunks_list[1])

	print(f"Chunk 1 token count: {len(chunk1_tokens)}")
	print(f"Chunk 2 token count: {len(chunk2_tokens)}")
	print(f"Last 150 tokens of chunk 1:")
	print(encoder.decode(chunk1_tokens[-150:]))
	print(f"First 150 tokens of chunk 2:")
	print(encoder.decode(chunk2_tokens[:150]))
	'''
	