# RAG Pipeline — Failure Log

## F001 — Retrieval Failure: Vocabulary Overlap
**Date:** 26 July 2026  
**Question:** "Sell something in an auction is related to which phrase?"  
**Expected:** Auction off  
**Got:** Context does not mention auction  

**Root Cause:**  
Query contained the word "sell" which appears in dozens of chunks (Sell off, 
Sell out, Sell up etc.). These chunks dominated similarity scores, pushing the 
correct chunk containing "Auction off" out of the top 3 results.

**Retrieved chunks:** chunk_86, chunk_12, chunk_10  
**Correct chunk should have been:** chunk_2 (approx)  

**Impact:** Exact category retrieval failure. System returns wrong chunks 
despite correct answer existing in vector store.  

**Potential Fix:** Reranking — add a second pass that re-scores retrieved 
chunks using a more precise cross-encoder model. This would catch that 
"auction" in the chunk directly matches "auction" in the query despite 
"sell" noise.  

**Status:** Open


## F002 — Retrieval Failure: Chunk Size Too Large for Reference Documents
**Date:** 26 July 2026
**Affected Questions:** Q1, Q2, Q3, Q5, Q6, Q7, Q8, Q9
**Root Cause:** 800 token chunks bundle 15-20 phrasal verb entries together.
Embeddings average across too many concepts, weakening retrieval precision
for any single phrase.
**Potential Fix:** Reduce chunk size to 200 tokens with 50 overlap for
reference-style documents. Consider document-structure-aware chunking.
**Status:** Open


## F003 — Dataset Quality: Multiple Valid Answers
**Date:** 26 July 2026
**Question:** Q1 - Which phrase means Surrender?
**Problem:** Expected answer was "Yield to" but PDF contains multiple 
phrases meaning surrender: Give up, Give in, Give way to, Give yourself up.
Eval dataset must account for multiple valid answers.
**Fix:** Update evaluate_response to accept a list of valid answers,
not just one string.
**Status:** Open

## F004 — Semantic Retrieval Failure: Embedding Distance Too Large
**Date:** 26 July 2026
**Affected:** All 5 Semantic category questions (0/5)
**Pattern:** Questions using semantically related but lexically different 
phrasing consistently fail to retrieve correct chunks.
Example: "zip your lip" should retrieve "Zip up" but retrieves unrelated chunks.
**Root Cause:** text-embedding-3-small embedding distance between 
paraphrased queries and target phrases exceeds retrieval threshold.
**Potential Fix:** Reranking with a cross-encoder model, or HyDE 
(Hypothetical Document Embeddings) — generate a hypothetical answer 
first, embed that, then search.
**Status:** Open


## F005 — Silent Path Error: Wrong Vector Store Connected
**Date:** 26 July 2026
**Cause:** Typo in path "/.vector_store" vs "./vector_store"
**Effect:** Chroma silently created empty collection at wrong path.
Count showed 510 but queries returned empty — no error thrown.
**Lesson:** Always print absolute path on startup. Never trust relative 
paths without verification.
**Fix:** os.path.abspath(path) logged at startup to catch this immediately.

## F006 — HyDE Failed for Reference Documents
**Finding:** HyDE generated hypotheticals using vocabulary not present in 
the document. Generated phrases ("put up for auction") pulled wrong chunks 
("Put up", "Put down") due to invented vocabulary dominating embeddings.
**Conclusion:** HyDE works best for prose documents where hypothetical 
answers share vocabulary with the source. For structured reference lists 
with fixed vocabulary, direct question embedding performs better.
**Decision:** Reverted to direct question embedding with n_results=10.
**Status:** Closed

## F007 — Existence Queries Fail Vector Search
**Question:** "Is Yammer on phrase used anywhere?"
**Problem:** Vector search retrieves semantically similar chunks, not 
chunks containing a specific term. Existence queries ("is X mentioned") 
require keyword search, not semantic search.
**Potential Fix:** Hybrid search — combine vector search with BM25 
keyword search. Use keyword search for existence queries.
**Potential Fix:** Changed the question to "What does Yammer on mean?"
**Finding:** Yammer on" genuinely isn't being retrieved even with n_results=10. This is a real retrieval gap for alphabetically distant, low-frequency terms. 
**Status:** Open

## F008 — get_document_info Returns Unknown Filename
**Date:** 13 August 2026
**Function:** get_document_info in agent.py
**Problem:** Document filename is not stored in ChromaDB chunk metadata 
during ingestion. fn_store_embeddings in utils.py only stores chunk ID 
as source. get_document_info always returns "Unknown" for document name.
**Root Cause:** Metadata schema designed for retrieval only — filename 
not included at write time.
**Fix:** Update fn_store_embeddings to accept and store filename in 
chunk metadata:
    {"source": "chunk_0", "filename": "document.pdf"}
Then get_document_info can retrieve it from any chunk's metadata.
**Status:** Fixed — filename now stored in chunk metadata during ingestion

## F009 — Agent Citations Show 0% Relevance
**Cause:** Agent returns chunk IDs from tool calls but no distance scores.
Distance scores only available from direct ChromaDB query results.
**Fix:** Return distances alongside sources from search_documents tool,
pass them through run_agent return value.
**Status:** Fixed — distances returned from search_documents tool and converted to relevance scores

## Eval Summary — Phase 1 RAG Pipeline
**Date:** 28 July 2026  
**Dataset:** 15 questions across 3 categories (Exact, Semantic, Not Found)  
**Document:** Complete Phrasal Verbs List PDF (510 chunks, 200 tokens each)

**Results:**
- Exact:     4/5  (80%)
- Semantic:  2/4  (50%)
- Not Found: 6/6  (100%)
- Overall:   12/15 (80%)

**What works well:**
- Exact phrase retrieval with direct question embedding
- Hallucination prevention — system correctly refuses to answer 
  when context doesn't contain relevant information
- Cosine similarity with n_results=10 provides adequate coverage 
  for most queries

  **Known failure modes:**
- F001: Vocabulary overlap causes wrong chunk retrieval
- F002: Large chunks reduce retrieval precision for reference documents
- F003: Multiple valid answers exist for single expected answer
- F004: Semantic retrieval fails for lexically distant queries
- F005: Silent path error — wrong vector store connected
- F006: HyDE failed for structured reference documents
- F007: Existence queries require hybrid search, not vector search

**What would improve scores:**
- Hybrid search (BM25 + vector) for existence queries — fixes Q2
- Reranking with cross-encoder — would improve Semantic category
- Larger eval dataset with more diverse query types

**Conclusion:**
Pipeline performs well on direct queries and correctly handles 
out-of-scope questions. Semantic retrieval is the primary gap — 
requires reranking or hybrid search to close.