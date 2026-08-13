import chromadb
import utils
import httpx
import os
import json

MAX_ITERATIONS = 5

tools=[
    {
        "type":"function",
        "function":{
            "name":"search_documents",
            "description": "Retrieves an array of relevant chunks from a vector store for a question",
            "strict":True,
            "parameters": {
                "type":"object",
                "properties":{
                    "question":{
                        "type":"string",
                        "description":"Word or phrase or sentence"
                    }
                },
                "required": ["question"],
                "additionalProperties":False
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_document_info",
            "description": "Returns metadata about the uploaded document",
            "strict":True,
            "parameters": {
                "type":"object",
                "properties":{},
                "required": [],
                "additionalProperties":False
            }
        }
    }
    ]

async def search_documents(question:str, collection) -> list[str]:
    if question is None or question == "":
        return []
        
    question_embedding = await utils.generate_embeddings(question)   
    query_result = collection.query(
    query_embeddings=question_embedding,
    n_results = 10
    )
    
    return {
        "chunks": query_result["documents"][0],
        "sources": query_result["metadatas"][0],
        "distances": query_result["distances"][0]
    }

async def get_document_info(collection):
    """
    Returns high-level information about the uploaded document.

    Returns:
        {
            "name": str,
            "chunk_count": int,
            "topics": list[str]
        }
    """
    try:
        # Get all stored records.
        data = collection.get(
            include=["metadatas"]
        )

        metadatas = data.get("metadatas", [])

        if not metadatas:
            return {
                "name": None,
                "chunk_count": 0,
                "topics": []
            }

        # Every chunk may contain the same document metadata.
        first_metadata = metadatas[0] or {}

        document_name = first_metadata.get(
            "filename",
            first_metadata.get("file_name", "Unknown")
        )        

        # Number of chunks
        chunk_count = len(metadatas)

        # Get topics if your upload process stored them.
        topics = set()

        for metadata in metadatas:

            if not metadata:
                continue

            topic = metadata.get("topic")

            if topic:
                topics.add(topic)

        return {
            "name": document_name,
            "chunk_count": chunk_count,
            "topics": sorted(topics)
        }    
    except Exception as ex:
        return {
            "status": "error",
            "message": str(ex)
        }


async def run_agent(question: str, collection, conversation_history: list) -> dict:
    # returns {"answer": str, "tool_calls_made": int, "sources": list}
    iteration = 0
    tool_calls_made = 0
    array_messages = []
    sources_with_distances = []

    array_messages.append({
        "role": "system", 
        "content":"""
        You are a document question-answering agent.

        You have access to tools that can search the uploaded documents
        and retrieve document metadata.

        Use search_documents when you need information contained inside
        the document.

        Use get_document_info when the user is asking about metadata
        or information about the uploaded document itself.

        Do not invent information that is not available from the
        conversation or tool results.
        """
    })

    if len(conversation_history) > 0:	
        for item in conversation_history:
            array_messages.append({
                "role": "user", 
                "content": item['question']
            })

            array_messages.append({
                "role": "assistant", 
                "content": item['answer']
            })
                  
    array_messages.append({
    "role": "user", 
    "content": question
    })


    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while iteration < MAX_ITERATIONS:
            uri='https://api.openai.com/v1/chat/completions'
                    
            api_key = os.environ["OPENAI_API_KEY"]
            
            headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":"application/json"
            }
            

            payload={
                "model":"gpt-4o-mini",
                "messages": array_messages,
                "tools":tools,
                "tool_choice": "auto"
            }
                        
            response = await client.post(uri,json=payload,headers=headers)

            message =  response.json()["choices"][0]["message"]

            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                answer = message.get("content", "")

                return {"answer": answer,
                        "tool_calls_made": tool_calls_made, 
                        "sources": sources_with_distances}

            #["content"]
            array_messages.append(message)
            
            
            for tool_call in tool_calls:
                tool_calls_made += 1

                tool_call_id = tool_call["id"]

                function_name = tool_call["function"]["name"]

                arguments_string = tool_call["function"]["arguments"]

                arguments = json.loads(arguments_string)

                if function_name == "search_documents":
                    search_question = arguments["question"]

                    tool_result = await search_documents(
                        search_question,
                        collection
                    )              

                    #sources.extend(tool_result.get("sources", []))
                    for s, d in zip(tool_result.get("sources", []), tool_result.get("distances", [])):
                        chunk_id = s.get("source")
                        if chunk_id and not any(x["chunk"] == chunk_id for x in sources_with_distances):
                            sources_with_distances.append({
                                "chunk": chunk_id,
                                "distance": d
                            })

                elif function_name == "get_document_info":

                    result = await get_document_info(
                        #document_question,
                        collection
                    )

                    tool_result = result  
                else:
                    tool_result = {
                        "error": f"Unknown tool: {function_name}"
                    }             

                array_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        tool_result,
                    )
                }) 
            iteration += 1

    #if MAX_ITERATIONS reached
    return {
                "answer": "I was unable to find a complete answer after several attempts.",
                "tool_calls_made": tool_calls_made,
                "sources": sources
            }
        

    
                       
