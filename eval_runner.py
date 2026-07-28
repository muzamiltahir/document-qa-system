import os
import httpx
import asyncio
import json
import chromadb
import utils
from completeapi import generate_answer
import numpy as np

async def generate_hypothetical_answer(question:str) -> str:
    # Call gpt-4o-mini with a prompt that asks it to generate
    # what a likely answer from a phrasal verbs document would look like
    # Return the hypothetical answer as a string
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        uri='https://api.openai.com/v1/chat/completions'

        api_key = os.environ['OPENAI_API_KEY'] 

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload ={
            "model":"gpt-4o-mini",
            "messages": [
                {"role":"system",
                "content":f"""Generate a hypothetical entry from a phrasal verbs reference document.
                                Use EXACTLY this format:
                                Phrase — Meaning. Example sentence with PHRASAL VERB in capitals.
                                QUESTION:{question}
                                """}
            ]
        }

        response = await client.post(uri,json=payload,headers=headers)
                
        return response.json()["choices"][0]["message"]["content"]
    

def evaluate_response(expected: str, response: str, category: str):
    if category == "Not Found":
        not_found_signals = ["not in the provided context",
                            "not in the context",
                            "not available",
                            "does not exist",
                            "no phrase",
                            "cannot find",
                            "no information",
                            "not mentioned",
                            "does not appear",
                            "does not contain",
                            "does not provide",
                            "not provided",
                            "cannot answer",
                            "i cannot",
                            "does not explicitly",
                            "is not present",
                            "does not include"]
        return any(signal in response.lower() for signal in not_found_signals)

    if isinstance(expected, list):
        return any(exp.lower() in response.lower() for exp in expected)
    
    return expected.lower() in response.lower()


eval_dataset=[
    {'id':1,'question': 'Which phrase means Surrender and which sentence is it used in?',
    'expected': ['Yield to','Give up','Give in', 'Give way to', 'Give yourself up'],'category':'Exact'},
    {'id':2,'question': 'What does Yammer on mean?',
        'expected': 'Talk continuously','category':'Exact'},
    {'id':3,'question': 'Leave a group by moving in a different direction is meaning of which phrase?',
        'expected': ['Peel away', 'Peel away from'],'category':'Exact'},
    {'id':4,'question': 'Is visit related to come by phrase?',
        'expected': 'Yes','category':'Exact'},
    {'id':5,'question': 'Sell something in an auction is related to which phrase?',
        'expected': 'Auction off','category':'Exact'},
    {'id':6,'question': 'To remain faithful to something or someone and to stand to is closely related to which phrase?',
        'expected': ['abide by', 'stick by', 'stand by', 'stick to'],'category':'Semantic'},
    {'id':7,'question': 'To fall suddenly into a very deep sleep is closely related to which phrase?',
        'expected': 'Zonk out','category':'Semantic'},
    {'id':8,'question': 'zip your lip is similar to which phrase?',
        'expected': ['Zip up', 'Zip it'],'category':'Semantic'},
    {'id':9,'question': 'Hair raiser is closely related to which phrase?',
        'expected': 'nail biter','category':'Semantic'},
    {'id':10,'question': 'I am all ears is closely related to which phrase?',
        'expected': 'None','category':'Not Found'},
    {'id':11,'question': 'which sentence example is given for abashed by?',
        'expected': 'None','category':'Not Found'},
    {'id':12,'question': 'which sentence example is given for above par?',
        'expected': 'None','category':'Not Found'},
    {'id':13,'question': 'ace in the hole is a phrase given in the context, can you find it?',
        'expected': 'None','category':'Not Found'},
    {'id':14,'question': 'Is absent minded phrase available?',
        'expected': 'No','category':'Not Found'},
    {'id':15,'question': 'Is there any phrase that starts with letter x?',
        'expected': 'No','category':'Not Found'}]


async def main():
    chroma_client = chromadb.PersistentClient(path="./vector_store")
    collection = chroma_client.get_or_create_collection(name="documents2")
    output  = ""
    exact_total =0
    exact_pass = 0
    semantic_pass = 0
    semantic_total = 0
    notfound_pass = 0
    notfound_total = 0
    total_pass = 0

    for i in range(len(eval_dataset)):
        id= eval_dataset[i]['id']
        question =eval_dataset[i]['question']    
        category =eval_dataset[i]['category']
        expected_answer = eval_dataset[i]['expected']    
        question_embedding = await utils.generate_embeddings(question)
        #hypothetical = await generate_hypothetical_answer(question)
        #question_embedding = await utils.generate_embeddings(hypothetical)

        query_result = collection.query(
            query_embeddings=question_embedding,
            n_results=10
        )

        if id == 6:
            print("=== Q6 RETRIEVED CHUNKS ===")
            for doc, meta in zip(query_result['documents'][0], 
                                 query_result['metadatas'][0]):
                print(f"\n{meta['source']}: {doc[:200]}")

        retrieved_chunks = query_result['documents'][0]

        resp = await generate_answer(question,retrieved_chunks)

        got_answer =  resp.json()['choices'][0]['message']['content']

        passed = evaluate_response(expected_answer,got_answer,category)

        total_pass += 1 if passed else 0

        if category == 'Exact':
            exact_total += 1
            exact_pass += 1 if passed else 0
        elif category == 'Semantic':
            semantic_total += 1
            semantic_pass += 1 if passed else 0
        elif category == 'Not Found':
            notfound_total += 1
            notfound_pass += 1 if passed else 0

        output += f"Q{id} {category}: {question}\n"
        output += f"Expected: {expected_answer}\n"
        output += f"Got: {got_answer}\n"
        output += f"Pass/Fail: {'Pass' if passed else 'Fail'}\n"
        output += "---\n"

    # add summary at end
    output += f"\nExact:     {exact_pass}/{exact_total}\n"
    output += f"Semantic:  {semantic_pass}/{semantic_total}\n"
    output += f"Not Found: {notfound_pass}/{notfound_total}\n"
    output += f"Overall:   {total_pass}/15\n"    

    with open("eval_result_2.3.txt", "w") as f:
        f.write(output)

    print(output)

asyncio.run(main())


async def manual_run():
    chroma_client = chromadb.PersistentClient(path="./vector_store")
    collection = chroma_client.get_or_create_collection(name="documents2")

    question = "Sell something in an auction is related to which phrase?"
    hypothetical = await generate_hypothetical_answer(question)
    print(f"Hypothetical: {hypothetical}")

    embedding = await utils.generate_embeddings(hypothetical)

    print(f"Query embedding type: {type(embedding)}")
    print(f"Query embedding length: {len(embedding)}")
    print(f"Is nested list: {isinstance(embedding[0], list)}")

    '''
    print(f"Embedding type: {type(embedding)}")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    results = collection.query(query_embeddings=embedding, 
                               n_results=3,
                               include=["documents", "metadatas", "distances"]
                               )
    '''

    results = collection.query(query_embeddings=np.array([embedding]).tolist(),
                                n_results=10
                            )    
        
    print(f"Raw result: {results}")

    print(f"Retrieved chunks:")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"\n{meta['source']}:")
        print(doc[:300])


#asyncio.run(manual_run())     