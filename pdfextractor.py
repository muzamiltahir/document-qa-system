import pdfplumber

loc = r'D:\TechLearning\PythonByClaude\Attachments\Complete-PhrasalVerbs-list.pdf'

with pdfplumber.open(loc) as pdf:
	pages = pdf.pages
	totalPageCount = 0
	for i in range(len(pages)):		
		pageText = pages[i].extract_text()
		print("Page Number " + str(i+1))
		print(pageText)
		totalPageCount = i+1
		if i >= 5:
			break
	print("Total Page Count:" + str(totalPageCount))
	
		
	