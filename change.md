Changes from the Basic RAG Code to the Optimized RAG Code

In the previous RAG example, you built a basic RAG pipeline.
The flow was:
Load document
↓
Split into chunks
↓
Create embeddings
↓
Store in Chroma
↓
Retrieve relevant chunks
↓
Send context to the LLM
↓
Generate answer
This is a correct first implementation.
However, in real-world systems, basic RAG is usually not enough.
In this updated version, we improve the RAG system to make it:
•	More reliable
•	Easier to debug
•	Safer against hallucination
•	Better at retrieving useful context
•	Closer to how production RAG systems are designed
The main difference is:
Basic RAG retrieves something.
Optimized RAG checks whether what it retrieved is good enough to answer.
________________________________________
1. Chunk Size and Overlap
In the old code
We used:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15
)
This creates very small chunks.
Small chunks may be useful sometimes, but they can also split important meaning.
For example, a sentence may be split into two chunks:
Chunk 1:
Students can book lessons through the mobile app by selecting the Book button, choosing a teacher,

Chunk 2:
a teacher, and selecting a suitable time.
This is not ideal because the second chunk does not fully make sense by itself.
________________________________________
In the new code
We use configurable values:
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
This gives each chunk more context.
It also makes it easier to experiment with different values.
________________________________________
Why this change matters
Chunking affects retrieval quality.
If chunks are too small, the meaning may be lost.
If chunks are too large, the retrieved context may contain too much irrelevant information.
The goal is to find a good balance.
Important idea:
Better chunks → better retrieval → better context → better answers
________________________________________
2. Metadata
In the old code
The document had only simple metadata:
metadata = {
    "source": "company_policy.txt"
}
This tells us where the document came from, but not much more.
________________________________________
In the new code
We add richer metadata:
metadata = {
    "source": DOCUMENT_PATH,
    "document_type": "policy",
    "department": "support",
    "access_level": "public"
}
Then we also add a chunk index:
chunk.metadata["chunk_index"] = index
________________________________________
Why this change matters
Metadata helps us understand and control retrieval.
It can help with:
•	Debugging
•	Source tracking
•	Citations
•	Filtering
•	Permissions
•	Access control
Example:
If the system retrieves a chunk, we can now know:
Source: company_policy.txt
Chunk Index: 2
Department: support
Access Level: public
This is much better than only seeing the text.
Important idea:
page_content is the knowledge.
metadata tells us where the knowledge came from and how we should use it.
________________________________________
3. Retrieval Mode
In the old code
We used simple top-k retrieval:
retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)
This means:
Return the single most similar chunk.
This is simple, but it may be risky.
If the user asks a question that needs more than one piece of information, k=1 may not retrieve enough context.
Example:
Can I cancel my session and what happens if I miss it?
One chunk may answer cancellation, but not missed sessions.
________________________________________
In the new code
We support two retrieval modes:
retrieve_top_k(question)
and:
retrieve_mmr(question)
This allows us to compare:
Normal top-k retrieval
vs
MMR retrieval
________________________________________
4. MMR Retrieval
In the old code
We did not use MMR.
The retriever simply returned the most similar chunks.
________________________________________
In the new code
We added MMR:
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": TOP_K,
        "fetch_k": FETCH_K
    }
)
MMR means:
Maximal Marginal Relevance
MMR tries to return chunks that are:
Relevant to the question
+
Different from each other
________________________________________
Why this change matters
Normal top-k retrieval may return repeated or very similar chunks.
Example:
Normal top-k:
Chunk A
Chunk A'
Chunk A''
These chunks may all say almost the same thing.
MMR tries to return broader context:
MMR:
Chunk A
Chunk B
Chunk C
Important idea:
Top-k focuses mainly on similarity.
MMR balances similarity and diversity.
Use MMR when the retrieved chunks are too repetitive or when the question may need information from different parts of the document.
________________________________________
5. Refuse Threshold
In the old code
The system always sent the retrieved context to the LLM.
answer = chain.invoke({
    "context": context,
    "question": question
})
Even if the retrieved context was weak or irrelevant, the LLM still tried to answer.
This can lead to hallucination.
________________________________________
In the new code
We retrieve documents with relevance scores:
scored_results = retrieve_with_scores(question)
best_doc, best_score = scored_results[0]
Then we check the score:
if best_score < REFUSE_THRESHOLD:
    return "I don't have enough information in the provided documents."
________________________________________
Why this change matters
A RAG system should not always answer.
Sometimes the correct behavior is to refuse.
Example:
User question:
How much does the premium package cost?

Retrieved context:
Students can cancel a session up to 24 hours before the session starts.
The retrieved context is not about pricing.
So the system should answer:
I don't have enough information in the provided documents.
Important idea:
Refusal is not a failure.
In RAG, refusal is sometimes the correct answer.
________________________________________
6. Stronger Grounding Prompt
In the old code
The prompt was simple:
Answer the question using only the context below.

If the answer is not in the context, say:
"I don't have enough information in the provided documents."
This is good for a basic demo.
________________________________________
In the new code
The prompt is stricter:
You must answer using ONLY the retrieved context below.

Rules:
- If the answer exists in the context, answer clearly and concisely.
- If the answer is not found in the context, say exactly:
  "I don't have enough information in the provided documents."
- Do not guess.
- Do not use general knowledge.
- Do not invent prices, policies, or conditions.
________________________________________
Why this change matters
The new prompt reduces hallucination.
It tells the model clearly:
•	Use only retrieved context
•	Do not guess
•	Do not use general knowledge
•	Do not invent missing information
This is called grounding.
Important idea:
Grounding means the answer is based on the retrieved context.
Hallucination means the answer is not supported by the retrieved context.
________________________________________
7. Better Debugging Output
In the old code
We printed the retrieved context:
print("\n=== Retrieved Context ===")
print(context)
This is useful and should always be done when learning RAG.
________________________________________
In the new code
We print more debugging information:
print(f"Question: {question}")
print(f"Retrieval mode: {retrieval_mode}")
print(f"Best relevance score: {best_score:.4f}")
We also print metadata for each retrieved chunk:
Source
Chunk Index
Department
Access Level
Content
________________________________________
Why this change matters
When a RAG answer is wrong, we need to debug the retrieval step.
We need to ask:
•	Which chunk was retrieved?
•	Was it the correct chunk?
•	Did it contain the answer?
•	What was its relevance score?
•	Was the context noisy?
•	Was the answer grounded?
•	Should the system have refused?
Important idea:
Before blaming the LLM, inspect the retrieved context.
________________________________________
8. Rebuilding the Vector Store
In the old code
The vector store was saved here:
persist_directory="./chroma_db"
But if we changed the document, chunk size, overlap, or embedding model, the old vector store could still remain.
This can create confusion.
You may think you are testing new chunks, but actually Chroma may still contain old vectors.
________________________________________
In the new code
We added:
REBUILD_VECTOR_STORE = True

if REBUILD_VECTOR_STORE and os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
This deletes the old Chroma database before rebuilding.
________________________________________
Why this change matters
Whenever you change:
•	Document content
•	chunk_size
•	chunk_overlap
•	Embedding model
you should rebuild the vector store.
Important idea:
If the knowledge or chunking changes, rebuild the vector store.
________________________________________
9. Interactive Testing
In the old code
The script asked one question:
question = input("Ask a question about the policy: ")
Then it ended.
To ask another question, you had to run the script again.
________________________________________
In the new code
We use an interactive loop:
while True:
    user_input = input("Ask a question: ").strip()
The program supports:
mode top_k
mode mmr
exit
________________________________________
Why this change matters
This makes it easier to test and compare.
You can ask multiple questions in the same run.
You can also switch between retrieval modes:
mode top_k
or:
mode mmr
Then compare the retrieved context and final answer.
________________________________________
10. Summary of Main Differences
Area	Old Code	New Optimized Code
Purpose	Basic RAG demo	More reliable RAG demo
Chunking	Small chunks: 100 / 15	Configurable chunks: 300 / 50
Metadata	Basic source only	Source, type, department, access level, chunk index
Retrieval	Basic top-k only	Top-k and MMR
k value	Usually k=1	Configurable TOP_K=3
Refusal	Only prompt-based	Prompt + relevance threshold
Prompt	Basic grounding	Stronger anti-hallucination rules
Debugging	Prints context	Prints context, score, metadata, retrieval mode
Vector store rebuild	Manual	Optional automatic rebuild
Testing	One question per run	Interactive loop
Production thinking	Beginner-level	Closer to real-world RAG
________________________________________
11. How to Think About the Old Code vs New Code
The old code answers this question:
Can we retrieve context and generate an answer from it?
The new code answers a more professional question:
Can we trust the retrieved context enough to answer?
That is the key difference.
The old code proves that RAG works.
The new code shows how to make RAG safer, more reliable, and easier to debug.
________________________________________
12. Final Teaching Point
Remember this:
Basic RAG retrieves something.
Professional RAG checks whether the retrieved evidence is good enough.
A powerful LLM with poor retrieval still gives poor answers.
A good RAG system should:
•	Retrieve relevant chunks
•	Avoid noisy context
•	Use metadata for traceability
•	Refuse when evidence is weak
•	Answer only from provided context
•	Be easy to debug
•	Rebuild the vector store when documents or chunking change
Final sentence:
Good RAG is not just about answering.
Good RAG is about answering only when the system has the right evidence.
