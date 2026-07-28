import os
import httpx
import asyncio
import json

async def CallOpenAPI():
	async with httpx.AsyncClient() as client:
		uri='https://api.openai.com/v1/chat/completions'
		
		api_key = os.environ["OPENAI_API_KEY"]
			
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}
		
		payload = {
				"model": "gpt-4o-mini",
				"messages": [
					{"role": "user", "content": "What is the capital of USA? Give me just the name of the capital."}
				]
				#,"stream":False
				#,"temperature":0.7
			  }
			  
		respose = await client.post(uri,json=payload,headers=headers)
		print(response.json()["choices"][0]["message"]["content"])
asyncio.run(CallOpenAPI())