# 📄 AI-Powered PDF Question Answering System

An intelligent document-based Question Answering system that allows users to upload PDF files and ask questions in natural language. The system uses *Retrieval-Augmented Generation (RAG)* to generate accurate, context-aware responses strictly from the uploaded document while minimizing hallucinations.

---

## ✨ Features

✅ Upload and process PDF documents  
✅ Extract and analyze document text  
✅ Semantic search using vector embeddings  
✅ Context-aware question answering  
✅ Hallucination reduction mechanisms  
✅ Fast retrieval using FAISS  
✅ Clean interactive UI with Streamlit  
✅ Answers restricted to document content only  

---

## 🛠️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| Frontend/UI | Streamlit |
| PDF Parsing | PyPDF |
| Embedding Model | Sentence Transformers |
| Vector Database | FAISS |
| Large Language Model | Groq API |
| AI Model | Llama 3 |
| AI Architecture | Retrieval-Augmented Generation (RAG) |

---

## 🧠 System Architecture

The application follows a *Retrieval-Augmented Generation (RAG)* pipeline:


PDF Upload
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
Store in FAISS Vector Database
     ↓
User Question
     ↓
Similarity Search
     ↓
Retrieve Relevant Context
     ↓
LLM Response Generation


---

## ⚙️ Workflow

### Step 1: Upload PDF
Users upload one or multiple PDF documents.

### Step 2: Extract Text
Text is extracted using *PyPDF*.

### Step 3: Split into Chunks
Large text is divided into smaller meaningful chunks.

### Step 4: Generate Embeddings
Sentence Transformers converts chunks into vector embeddings.

### Step 5: Store in FAISS
Embeddings are indexed for fast semantic retrieval.

### Step 6: Retrieve Relevant Information
User queries are converted into embeddings and matched with document chunks.

### Step 7: Generate Grounded Response
Only retrieved document context is sent to the LLM.

---

## 🎯 AI Approach Used

This project uses *Retrieval-Augmented Generation (RAG)* to improve answer quality and reduce hallucinations.

Instead of sending the entire document to the language model:

- Relevant content is retrieved first
- Only contextual information is passed
- Answers are generated using document evidence
- External knowledge is restricted

This ensures reliable and document-grounded outputs.

---

## 📝 Prompt Engineering Strategy

A strict system prompt is designed to control model behavior:


Rules:
• Answer only from provided context
• Do not use external knowledge
• Do not guess information
• If answer is unavailable, return:

"Not available in document"


Temperature:


Temperature = 0


This significantly reduces randomness and hallucination.

---

## 🛡️ Hallucination Prevention Techniques

Multiple safeguards were implemented:

### 1. Retrieval-Based Generation
Uses RAG instead of direct prompting.

### 2. Strict Prompt Constraints
Prevents unsupported answers.

### 3. Similarity Threshold Filtering
Filters low-confidence retrieved content.

### 4. Context-Limited Generation
LLM only sees retrieved chunks.

### 5. Temperature Control
Temperature set to 0 for deterministic output.

---

## Example Output

*Question:*


What is the objective of this report?


*Answer:*


The report objective is ...


If information does not exist:


Not available in document


---

## 🚀 Installation

Clone repository:

bash
git clone https://github.com/Venkatesh2033/Task_1.git


Move into project folder:

bash
cd Task_1


Install dependencies:

bash
pip install -r requirements.txt


Run application:

bash
streamlit run app.py


---

## Future Improvements

- Multi-PDF comparison support
- Chat history memory
- OCR support for scanned PDFs
- Citation generation
- Hybrid retrieval techniques
- Better chunking strategies
