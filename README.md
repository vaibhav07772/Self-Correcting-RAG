# 🧠 Self-Correcting RAG System

A production-grade **Self-Correcting Retrieval-Augmented Generation (RAG)** system that automatically detects hallucinations and corrects itself using **LLM-as-Judge** evaluation with **LangGraph** workflow orchestration.

---

## 📌 Project Overview

This system goes beyond traditional RAG by implementing a **self-correction loop**. When the LLM generates an answer, a separate **LLM-as-Judge** evaluates it for:

- **Faithfulness** — Is the answer based on the provided context?
- **Hallucination** — Did the model create false information?
- **Confidence Score** — How confident is the model in its answer?

If the evaluation fails (hallucination detected, confidence < threshold), the system **automatically retries** up to 3 times with correction prompts — ensuring you always get the most accurate response possible.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **📄 Document Indexing** | Load PDF/TXT documents, split into chunks, and store embeddings in ChromaDB |
| **🔍 Hybrid Retrieval** | BM25 + Vector Search (Ensemble Retriever) for optimal context retrieval |
| **🤖 RAG Generation** | Generate answers using Groq's Llama 3.3 70B model |
| **🧠 LLM-as-Judge** | Evaluate answers for faithfulness, hallucination, and confidence |
| **🔄 Self-Correction Loop** | Automatically retry up to 3 times with correction prompts |
| **🛡️ Guardrails** | PII redaction and prompt injection detection |
| **📊 Observability** | LangFuse integration for tracking, logging, and monitoring |
| **🖥️ Full UI** | Streamlit interface with real-time feedback and evaluation scores |

---

## 🏗️ System Architecture
User Query
↓
[Input Guardrails]
↓
[RAG Retrieval] ← [ChromaDB Vector Store]
↓
[LLM Generation] (Groq Llama 3.3)
↓
[LLM-as-Judge Evaluation]
↓
┌───────────────────────┐
│ Is Answer Valid? │
└───────────────────────┘
↓ Yes ↓ No
[Final Answer] [Self-Correction]
↓
[Retry (max 3)]





---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq (Llama 3.3 70B Versatile) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **RAG Framework** | LangChain (LangChain 1.x) |
| **Orchestration** | LangGraph (Stateful Workflow) |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Evaluation** | LLM-as-Judge (Custom) |
| **Observability** | LangFuse |
| **Language** | Python 3.12+ |

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vaibhav07772/Self-Correcting-RAG.git
cd Self-Correcting-RAG




2. Create Conda Environment
conda create -n self_correcting_rag python=3.12 -y
conda activate self_correcting_rag



3. Install Dependencies
pip install -r requirements.txt


4. Set Up .env File
# Groq API
OPENAI_API_KEY=gsk_your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL_NAME=llama-3.3-70b-versatile

# LangFuse (Observability)
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Evaluation Settings
MAX_RETRIES=3
CONFIDENCE_THRESHOLD=0.7


5. Add Documents
Place your PDF/TXT files in the data/ folder.


6. Index Documents
python core/indexing.py


7. Run Backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000



8. Run Frontend
streamlit run streamlit_app.py


💻 UI & Interaction
📊 System Info Panel
The sidebar displays how the system works:

🔍 RAG retrieves relevant context
🤖 LLM generates answer
🧠 LLM-as-Judge evaluates answer
🔄 If hallucinated → Auto-correct (max 3 tries)
✅ Returns final answer



💬 Chat Interface
User Input: Ask any question related to the indexed documents
Real-time Feedback: See the system "thinking" with a spinner
Evaluation Results: Each answer shows:

✅ Valid answer with confidence score
⚠️ Hallucination warning
🔄 Self-correction count



🧪 Sample Input & Output
✅ Case 1: Valid Answer
User Input:
What is the main topic of this document?

System Response:
The main topic of this document appears to be Artificial Intelligence (AI) and its related concepts, including Natural Language Processing (NLP) and Transformers, with explanations of their definitions, key tasks, and applications.

✅ Valid answer (Confidence: 0.95)



🔄 Case 2: Self-Correction
User Input:
What is the difference between Natural Language Processing (NLP) and Transformers according to this document?

System Response:
The provided context does not mention Transformers, and it only briefly mentions Natural Language Processing (NLP) with a question "What is Natural Language Processing (NLP)?" but does not provide an answer. Therefore, based on the given context, it is not possible to determine the difference between Natural Language Processing (NLP) and Transformers.

🔄 Self-corrected after 3 attempts
⚠️ Answer may contain hallucinations (Confidence: 0.00)




⚠️ Case 3: Hallucination Warning
User Input:
What are the key concepts of Artificial Intelligence discussed in this document?

System Response:
The key concepts of Artificial Intelligence discussed in this document are:

Natural Language Processing (NLP) - the branch of AI focused on enabling computers to understand, interpret, and generate human language.
Confusion Matrix - a table used to describe the performance of a classification model...
Precision, Recall, and F1-Score - metrics used to evaluate the performance of a classification model...

🔄 Self-corrected after 3 attempts
⚠️ Answer may contain hallucinations (Confidence: 0.00)



📁 Project Structure
self_correcting_rag/
├── app/
│   └── main.py                 # FastAPI Backend
├── core/
│   ├── indexing.py             # Document ingestion + ChromaDB
│   ├── retrieval.py            # Hybrid retrieval (BM25 + Vector)
│   ├── generation.py           # RAG generation with Groq LLM
│   ├── evaluation.py           # LLM-as-Judge evaluation
│   └── self_correction.py      # LangGraph self-correction loop
├── guardrails/
│   └── guardrails.py           # PII redaction + Prompt injection detection
├── config/
│   └── settings.py             # Environment variables & config
├── data/                       # Upload your PDF/TXT files here
├── chroma_db/                  # Vector database storage
├── streamlit_app.py            # Streamlit UI
├── requirements.txt            # Python dependencies
├── .env                        # API Keys (ignored)
└── README.md                   # This file



🔑 Key Components
📄 Document Indexing (core/indexing.py)
Loads PDF and TXT files
Splits documents into chunks (500 tokens, 50 overlap)
Generates embeddings using Sentence Transformers
Stores vectors in ChromaDB



🔍 Hybrid Retriever (core/retrieval.py)
Combines BM25 (keyword search) + Vector Search
Returns top-5 most relevant chunks



🤖 RAG Generator (core/generation.py)
Uses Groq's Llama 3.3 70B model via OpenAI-compatible API
Generates answers based on retrieved context
LangChain LCEL chain for seamless orchestration



🧠 LLM-as-Judge (core/evaluation.py)
Evaluates answers for:
Faithfulness: Is the answer fully based on context?
Hallucination: Did the model create false information?
Confidence Score: 0-1 scale (1 = completely faithful)
Returns structured JSON output



🔄 Self-Correction Loop (core/self_correction.py)
LangGraph-based stateful workflow
Max 3 retry attempts with correction prompts
Automatic fallback if threshold not met




🛡️ Guardrails (guardrails/guardrails.py)
PII redaction (email, phone, SSN)
Prompt injection detection



📊 Observability (langfuse)
Every query is traced in LangFuse
Track cost, latency, and quality scores




📊 Evaluation Metrics
Metric	                   Description
Faithfulness	           Is the answer based on provided context? (Yes/No)
Hallucination	           Did the model create false information? (Yes/No)
Confidence Score	       0-1 scale (1 = completely faithful)
Attempts	               Number of retries before final answer



💡 Future Improvements
□ Add RAGAS evaluation (Faithfulness, Answer Relevancy, Context Relevancy)
□ Support for more file formats (DOCX, MD, HTML)
□ Implement Re-ranking with Cross-Encoder
□ Add human-in-the-loop approval
□ Deploy with Docker + Kubernetes
□ Add CI/CD pipeline with GitHub Actions
□ Support for multiple LLM providers (OpenAI, Anthropic, Gemini)




🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.



📝 License
This project is for educational purposes as part of the GenAI learning journey.




📬 Connect with Me
Author: Vaibhav Singh
GitHub: vaibhav07772
LinkedIn: Vaibhav Singh




⭐ Acknowledgments
Groq for the free, fast LLM inference
LangChain for the RAG framework
LangGraph for stateful workflows
LangFuse for observability
Streamlit for the beautiful UI




💡 Why This Matters
Most AI systems confidently give wrong answers. This system is different—it knows its limits, self-corrects, and admits when it doesn't have enough information.

"A system that knows when it doesn't know is smarter than one that always pretends to know."


