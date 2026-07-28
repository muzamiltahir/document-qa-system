import httpx
import asyncio
import json

async def CallOpenAPI():
	async with httpx.AsyncClient() as client:
		data = {
				"model": "llama3.2:3b",
				"messages": [{"role": "user", "content": "How Are You?"}],
				"stream":False
			  }
		respose = await client.post('http://localhost:11434/api/chat',json=data)
		print(respose.json())
asyncio.run(CallOpenAPI())