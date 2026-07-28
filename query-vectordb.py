import pdfplumber
import tiktoken
import os
import httpx
import asyncio
import json
import chromadb

async def generate_embeddings(question: str):	
	timeout = httpx.Timeout(60.0)
	async with httpx.AsyncClient(timeout=timeout) as client:
		uri='https://api.openai.com/v1/embeddings'
		
		api_key = os.environ["OPENAI_API_KEY"]
			
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}
		
		final_embedding_list = []
		payload = {
					"model": "text-embedding-3-small",
					"input": question,
					"encoding_format": "float"
				  }
				  
		response = await client.post(uri,json=payload,headers=headers)
		embeddings_data = response.json()
		if "data" not in embeddings_data:
			raise ValueError(f"Embedding API error: {embeddings_data}")
			
		for item in embeddings_data["data"]:
			final_embedding_list.append(item["embedding"])
		return final_embedding_list[0]
		

async def main():
	question_embedding = await generate_embeddings('What are the sentences which uses Nail down phrase in them?')
		
	chroma_client = chromadb.PersistentClient(path="./vector_store")
	collection = chroma_client.get_or_create_collection(name="documents")
	
	results = collection.query(
		query_embeddings=question_embedding,
		n_results = 3
	)
	
	print(results)
	
#asyncio.run(main())	


async def generate_answer(question: str, context_chunks: list[str]):				
	context = " ".join(context_chunks)

	timeout = httpx.Timeout(30.0)	
	async with httpx.AsyncClient() as client:
		uri='https://api.openai.com/v1/chat/completions'
		
		api_key = os.environ["OPENAI_API_KEY"]
		
		headers={
		"Authorization": f"Bearer {api_key}",
		"Content-Type":"application/json"
		}
		
		
		payload={
			"model":"gpt-4o-mini",
			"messages": [
					{"role": "system", "content":f"You are a helpful assistant. Answer the question using only the context provided. If the answer is not in the context, say so explicitly.\n\n{context}"},
					{"role": "user", "content": question}
				]
		}
		
		response = await client.post(uri,json=payload,headers=headers)
		
		return response
		


async def main_second():
	query_result = {'ids': [['chunk_68', 'chunk_59', 'chunk_2']], 'embeddings': None, 'documents': [[".\nwhere the salary was considerably better.\nThe police MOVED the crowd ON\nMove on Make people move from a place.\nbecause they were holding up the traffic.\nShe's tired of living there and is\nMove out Leave a place you live or work in.\nMOVING OUT.\nThe president said they would MOVE the\nMove out Remove.\ntroops OUT next year.\nMove out Change lane or position to pass a vehicle. I MOVED OUT to overtake the bus.\nMove The government are MOVING\nMake preparations for something.\ntowards TOWARDS free elections.\nCould you MOVE UP and let me sit\nMove up Move to make space.\ndown?\nThey MOVED her UP to senior\nMove up Move to a higher level.\nmanagement.\nMuddle If you're ambitious, you cannot\nContinue without a clear aim or plan.\nalong MUDDLE ALONG.Muddle Do or achieve something without knowing We didn't know what to expect and just\nthrough what is required or having a plan. MUDDLED THROUGH.\nTake things that are ordered or sequenced My cleaner MUDDLED UP my books\nMuddle up\nand mess them up. and I can't find anything.\nMistake a person or thing for someone or\nMuddle up I MUDDLE her and her sister UP.\nsomething else because they look similar.\nMug up Study quickly, revise. I have to MUG UP before the exam.\nI need to MUG UP ON my history for the\nMug up on Study something quickly, revise.\ntest.\nShe said that she'd had to MULL IT\nMull over Think about an issue or problem.\nOVER before deciding what was\nBecome involved in something when your We tried to exclude him of the committee,\nMuscle in\ninvolvement is not wanted. but he MUSCLED IN.\nMuscle in Become involved in something despite She MUSCLED IN ON the deal even\non opposition to your involvement. though we wanted to do it without her.\nThey are MUSCLING INTO our market\nBecome involved even though there is\nMuscle into and there is very little we can do to stop\nopposition to your involvement.\nthem.\nThey used their contacts in local\nUse power, contacts, etc, to force someone\nMuscle out government to MUSCLE the competitors\nout.\nOUT.\nHe was making a fuss, so I told him to\nNaff off Get lost, go away (used as imperative).\nNAFF OFF.\nMy boss is always NAGGING AT me\nNag at Repeatedly criticise someone verbally. about my arriving a few minutes late for\nwork.\nThey are having trouble NAILING\nNail down Succeed in getting, achieve.\nDOWN the contract.\nI can't NAIL DOWN what's wrong with\nNail down Understand fully.\ntheir idea, but I'm sure it won't work.\nI can't NAIL them DOWN about when\nNail down Get full information from someone.\nthey're going to finish the project.\nI NAILED the job DOWN in the first\nNail down Succeed or achieve something.\ninterview.\nGive someone a name to remember I was NAMED AFTER my uncle who\nName after\nanother person. died in the war.\nI am not sure which university to apply\nNarrow Remove less important options to make it\nto, but I have NARROWED my list\ndown easier to choose.\nDOWN to three.\nI'm going to NERD OUT and not go on\nNerd out Play safe and avoid taking a risk.\nthe river trip.\nI had to NERD OUT when they asked\nNerd out Discuss something in great detail.\nabout conditionals and deontic modality.\nNip off Go somewhere quickly. I'm NIPPING OFF to get some milk.\nShe'll be back in a minute- she's just\nNip out Go somewhere quickly.\nNIPPED OUT to the shops.\nNod off Fall asleep.", "CKED OUT the challenger in the\nfourth round of the fight.\nThey're KNOCKING hundreds OUT a\nKnock out Sell, distribute.\nday in the sales.\nKnock They KNOCKED TOGETHER two\nJoin houses that had been separate.\ntogether outbuilding and turned them into a home.\nShe got KNOCKED UP when she was on\nKnock up Become or get someone pregnant..\nholiday.\nThe teams KNOCKED UP for a few\nKnock up Play a bit before a match to get ready.\nminutes before the final.\nThey KNOCKED a model UP over the\nKnock up Produce or create something quickly.\nweekend.\nI've got my exams next week and I\nKnuckle\nMake a great effort. haven't done much work, so I'd better\ndown\nKNUCKLE DOWN.\nThe teacher made the students\nKnuckle\nSubmit to authority. KNUCKLE UNDER and hand their\nunder\nprojects in on time.\nHe LANDED ME IN it when he told\nLand in Get someone into trouble.\nthem what I had done wrong.\nArrive, end a journey in a place, often We set out for Manchester, but LANDED\nLand up in\nwithout planning. UP IN Liverpool.\nHe LANDED ME WITH the job of\nLand with Create a problem for someone.\nproofreading the whole thing.\nLap up Appreciate something. He LAPPED UP their praise.\nLarge it up Have a good time when intoxicated. They were LARGING IT UP in the rave.\nThe children made me angry because they\nLark about Behave in a silly way.\nwere LARKING ABOUT.\nLark The students wouldn't stop LARKING\nBehave in a silly way.\naround AROUND.\nAfter they won, they went to a bar to L\nLark it up Enjoy yourself noisily and exuberantly.\nARK IT UP.\nThe rain was LASHING DOWN all day\nLash down Fall heavily (rain).\nand the roads were flooded.\nWe LASHED the tarpaulin DOWN to\nLash down Secure something with ropes or cords.\nstop the wind blowing it away.\nHe LASHED INTO them for messing\nLash into Criticise someone strongly.\nthins up.He LASHED OUT and broke the man's\nLash out Suddenly become violent.\nnose.\nHe LASHES OUT when things don't go\nLash out React angrily.\nhis way.\nLash out Spend a lot of money on luxuries. I LASHED OUT in the sales last week.\nLash out The press has LASHED OUT AGAINST\nCriticise something strongly.\nagainst the policy.\nHit someone suddenly, usually without He LASHED OUT AT me when I\nLash out at\nwarning, or try to hit them. laughed at him.\nShe LASHED OUT AT her colleagues\nLash out at Criticise someone or shout at them.\nwhen she was sacked.\nLash out on Spend a lot of money buying something. I LASHED OUT a lot ON a new car.\nThey were lying, but it took her ages to\nLatch on Understand, often after a long time.\nLATCH ON.\nUnderstand something, often after a long The police didn't LATCH ON TO what\nLatch on to\ntime. the crooks were doing for years.\nLatch onto Connect to something. The gecko LATCHED ONTO the ceiling.\nDecide or realise that something is good or Oil companies have LATCHED ONTO\nLatch onto\nprofitable. environmental ideas.\nPretend something (an injury, news, etc.) He LAUGHED OFF the sprained finger\nLaugh off\nisn’t important. but it obviously affected his golf game.\nThe rules of the sport were LAID DOWN\nLay down Establish rules or procedures.\nearly in the nineteenth century.\nHe got LAID DOWN in a turf", ".\nOF jail.\nStop supporting someone when they are in Everybody BAILED OUT ON him when\nBail out on\ntrouble. the scandal broke.\nI was late because he BAILED me UP on\nBail up Talk to someone and delay them.\nthe phone and wouldn't shut up.\nHe was BAILED UP by a couple of\nBail up Rob someone at gunpoint.\nmuggers as he came out of the bank.\nThe new project has BALLED me UP- I\nBall up Confuse or make things complicated.\nhave no idea what to do.\nHe BALLED UP his napkin when he had\nBall up Roll or form into a round shape.\nfinished eating.\nBalls up Spoil, ruin. He BALLSED the presentation UP.\nBang about Move in a place making a lot of noise. He's BANGING ABOUT in the kitchen.\nBang I can hear him BANGING ABOUT\nMove in a place making a lot of noise.\naround upstairs.\nHe BANGED ON for half an hour but no\nBang on Talk at great length.\none was listening.\nBang on He's always BANGING ON ABOUT\nKeep talking about something.\nabout football.\nShe BANGED the tune OUT on the\nBang out Play a musical instrument loudly.\npiano.\nThe judge BANGED him UP for eight\nBang up Put someone in prison.\nyears.\nBang up Damage badly. He BANGED his car UP last night.\nI'm BANKING ON your help; I can't do\nBank on Count or rely on.\nit alone.\nBargain Persuade someone to drop the price of I BARGAINED her DOWN to half what\ndown something they're selling. she originally wanted.\nBargain for Expect something to happen (usually I hadn't BARGAINED FOR so manynegative). people coming.\nExpect something to happen (usually\nBargain on I hadn't BARGAINED ON him coming.\nnegative).\nHe keeps BARGING IN and asking\nBarge in Enter a place and interrupt.\nstupid questions when I'm trying to work.\nThey BARGED INTO my office without\nBarge into Enter a place and interrupt people rudely. knocking and started talking even though\nI was on the phone.\nIf you BASH your monitor ABOUT like\nBash about Mistreat physically.\nthat, it won't last long.\nThe burglars BASHED the door IN to\nBash in Break, damage or injure by hitting.\nenter the house.\nWrite something quickly without much I BASHED the essay OUT the night\nBash out\npreparation. before I had to hand it in.\nThey BASHED him UP in the fight in the\nBash up Break, damage or hurt by hitting. pub last week and he had to go to\nhospital.\nShe BAWLED me OUT for coming home\nBawl out Scold, shout at someone.\ndrunk.\nShe BAWLED Raj OUT for getting there\nBawl out Scold.\nlate.\nHe BAWLED OUT our names at the top\nBawl out Sing or shout unpleasantly loudly.\nof his voice.\nThe police ARE AFTER him because of\nBe after Try to find or get.\nthe theft.\nThe next bus should BE ALONG in the\nBe along Arrive.\nnext quarter of an hour or so.\nBe away Be elsewhere; on holiday, etc.. She's AWAY on business for three weeks.\nBe cut out She's not CUT OUT FOR this kind of\nBe suitable, have the necessary qualities.\nfor work.\nShe was very CUT UP about coming\nBe cut up Be upset. second as she thought she deserved to\nwin.\nHe's BEEN DOWN since his partner left\nBe down Be depressed.\nhim.\nThe firm's profits ARE DOWN by ten\nBe down Be reduced or less.\npercent this quarter.\nAfter the argument, James is DOWN ON\nBe down on Have negative feelings toward someone.\nhis boss.\nBe down Gul is DOWN"]], 'uris': None, 'included': ['metadatas', 'documents', 'distances'], 'data': None, 'metadatas': [[{'source': 'chunk_68'}, {'source': 'chunk_59'}, {'source': 'chunk_2'}]], 'distances': [[0.5094848275184631, 0.5343165397644043, 0.5698158740997314]]}

	list = query_result['documents'][0]
	
	resp = await generate_answer('What are the sentences which uses Nail down phrase in them?',list)
	
	print(resp.json()["choices"][0]["message"]["content"])

asyncio.run(main_second())
		
		
		
	
			
			
			