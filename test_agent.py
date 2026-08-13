import asyncio
import chromadb
from pathlib import Path
from web.agent import run_agent

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE = BASE_DIR / "vector_store"

async def main():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE))
    collection = client.get_or_create_collection(name="documents3")

    result = await run_agent(
        question="Compare Auction off and Sell off",
        collection=collection,
        conversation_history=[]
    )

    print(f"Answer: {result['answer']}")
    print(f"Tool calls made: {result['tool_calls_made']}")
    print(f"Sources: {result['sources']}")

asyncio.run(main())