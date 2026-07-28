import pdfplumber

loc = r'D:\TechLearning\PythonByClaude\Attachments\Complete-PhrasalVerbs-list.pdf'
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
	words_list = text.split()
	step = 	chunk_size - overlap
	start=0
	final_list = []
	while start < len(words_list):
		end = min(start+chunk_size,len(words_list))
		chunk = words_list[start:end]
		final_list.append(" ".join(chunk))
		if end == len(words_list):
			break
		start += step
	return final_list
		

with pdfplumber.open(loc) as pdf:
	pages = pdf.pages
	page_text = ''
	for i, page in enumerate(pdf.pages):
		page_text += page.extract_text()		
	
	chunks_list = chunk_text(page_text)
		
	print("=== END OF CHUNK 1 ===")
	print(" ".join(chunks_list[0].split()[-150:]))

	print("=== START OF CHUNK 2 ===")
	print(" ".join(chunks_list[1].split()[:150]))
	
	print(f"Total chunks: {len(chunks_list)}")
	