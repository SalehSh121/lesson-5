Walkthrough: Advanced RAG Strategies in Lesson 5
We have successfully implemented the Advanced RAG strategies to match your lecture objectives on the advanced_rag branch.

What Was Done
1. Corpus Configuration
Created a corpus directory: lesson 5/data/corpus/
Added three text policies:
company_policy.txt (topic: general_policy, access: student_visible)
refund_policy.txt (topic: refund, access: admin_visible)
learning_plan.txt (topic: learning, access: student_visible)
2. Multi-File Loader & Metadata Tagging
Updated 
document_loader.py
 with:
load_corpus(directory_path): Reads all files inside the corpus directory.
infer_metadata_from_filename(file_path): Automatically extracts topic, file source, document type, and access level.
3. Role-Based Access Control (RBAC)
Updated 
retriever.py
 to add:
retrieve_with_role(query, user_role, retrieval_mode): Filters documents by access_level matching the user's role.
Passed the metadata_filter to retrieve_with_scores so that score comparisons and the refuse threshold check are computed on permitted files only.
4. Custom Keyword Reranking
Created 
reranker_service.py
 introducing an overlap-based reranking filter.
Registered the class in the package 
init
.py
.
Integrated the reranker in 
rag_pipeline.py
 as an optional second stage.
5. Interactive Shell Commands
Enhanced 
lesson5.py
 to support:
Toggling user access permissions (role student / role admin).
Activating/deactivating re-ranking (rerank on / rerank off).
Displaying the source file, chunk ID, access level, and topic for every retrieved document.
Verification Results
Interactive Shell Run Logs
1. Student Access Restricted (Refusal Test)
When asking about refund procedures as a student, the system blocks access to refund_policy.txt, fails the threshold check, and safely refuses to answer:

text

[student][rerank=False][top_k] Ask: How do I request a refund?
====================================================
Question: How do I request a refund?
Retrieval mode: top_k
User Role: student
Reranking Active: False
====================================================
Best relevance score: 0.5090
=== Refuse Threshold Triggered or No Context Found ===
=== Final Answer ===
I don't have enough information in the provided documents.
2. Admin Access Permitted (Access Test)
When switching role to admin, the filter is bypassed. The system retrieves refund_policy.txt and answers correctly:

text

[student][rerank=False][top_k] Ask: role admin
User role set to: admin (full access)
[admin][rerank=False][top_k] Ask: Who reviews refund requests?
====================================================
Question: Who reviews refund requests?
Retrieval mode: top_k
User Role: admin
Reranking Active: False
====================================================
Best relevance score: 0.6751
=== Retrieved Context & Sources ===
[1] File: refund_policy.txt | Topic: refund | Access: admin_visible | Chunk: 7
    Content: Refund requests are reviewed manually by the support team.
[2] File: refund_policy.txt | Topic: refund | Access: admin_visible | Chunk: 9
    Content: Refunds are not automatic and must be approved by the support team.
[3] File: refund_policy.txt | Topic: refund | Access: admin_visible | Chunk: 8
    Content: If a teacher misses a session, the student may receive a replacement session...
=== Final Answer ===
Refund requests are reviewed manually by the support team.
3. Reranker Execution
When activating re-ranking, candidate documents are retrieved, re-scored based on search term occurrences, and truncated to the top 2 elements:

text

[admin][rerank=False][top_k] Ask: rerank on
Re-ranking enabled.
[admin][rerank=True][top_k] Ask: What happens if a teacher misses a session?
====================================================
Question: What happens if a teacher misses a session?
Retrieval mode: top_k
User Role: admin
Reranking Active: True
====================================================
Best relevance score: 0.7655
=== Retrieved Context & Sources ===
[1] File: refund_policy.txt | Topic: refund | Access: admin_visible | Chunk: 8
    Content: If a teacher misses a session, the student may receive a replacement session or refund after admin review.
[2] File: company_policy.txt | Topic: general_policy | Access: student_visible | Chunk: 1
    Content: If a student misses a session without cancelling in advance, the session is counted as used.
=== Final Answer ===
If a teacher misses a session, the student may receive a replacement session or refund after admin review.
All system strategies operate seamlessly. Your branch is ready for today's lecture.

