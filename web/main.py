from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile
import utils
import chromadb
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE = BASE_DIR / "vector_store"
UPLOADS_DIR = BASE_DIR / "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI()

chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE))
collection = chroma_client.get_or_create_collection(name="documents3")

origins = [    
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_store = {}  # {session_id: [history]}

class UserInput(BaseModel):
    question: str
    sessionid: str

class Citation(BaseModel):
    chunk:str
    relevance:float

class Output(BaseModel):
    answer:str
    citation:list[Citation] = []

class ClearRequest(BaseModel):
    sessionid: str

class ClearResponse(BaseModel):
    status: str
    message: str

@app.post("/upload")
async def upload_file(file: UploadFile):
    if file is None:
        return {"status":"failed", "message" : "Please upload a file", "chunks":0}
    elif file.content_type.lower() != 'application/pdf':
        return {"status":"failed", "message" : "Invalid file type!", "chunks":0}
    
    file_path_name = f"{str(UPLOADS_DIR)}/{file.filename}" 
    with open(file_path_name,"wb") as f:
        content = await file.read()
        f.write(content)

    chunk_count = 0
    chunk_count = await utils.fn_extract_text_and_chunk(file_path_name)    
    return {"status":"success", "message" : "PDF processed", "chunks":chunk_count}

@app.post("/ask", response_model=Output)
async def ask_question(user_input:UserInput):
    question = user_input.question
    sessionid = user_input.sessionid
    
    conversation_history = []

    if sessionid in conversation_store:
        conversation_history = conversation_store[sessionid]

    question_embedding = await utils.generate_embeddings(question)   
    query_result = collection.query(
    query_embeddings=question_embedding,
    n_results = 10
    )

    retrieved_chunks = query_result['documents'][0]

    generated_answer = await utils.generate_answer(question,retrieved_chunks,conversation_history)
    
    conversation_history.append({
    'question':question,
    'answer':generated_answer
    })    

    conversation_store[sessionid] = conversation_history

    metadata = query_result['metadatas'][0]
    distances =  query_result['distances'][0]

    output = Output(answer=generated_answer)    

    for meta, distance in zip(metadata, distances):
        citation = Citation(chunk = meta['source'],relevance = round((1 - (distance / 2)),2))
        output.citation.append(citation)

    return output


@app.post("/clear")
async def clear_conversation_history(input:ClearRequest):
    sessionid = input.sessionid
    if sessionid in conversation_store:
        del conversation_store[sessionid]
        return ClearResponse(status="success",message="Conversation history cleared")

    return ClearResponse(status="failed",message="Conversation history not found")
            
app.mount("/",StaticFiles(directory="web/static",html=True), name="static")
     
