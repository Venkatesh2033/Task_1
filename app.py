import os
import tempfile

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* REMOVE FILE UPLOADER CLOSE BUTTON */

button[title="Remove file"] {
    display: none !important;
}

/* REMOVE FILE NAME */

[data-testid="stFileUploaderFileName"] {
    display: none !important;
}

/* APP */

.stApp {
    background:
    linear-gradient(
        135deg,
        #020617 0%,
        #081225 45%,
        #0f172a 100%
    );

    color: white;
}

/* MAIN */

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: rgba(8,12,25,0.98);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* TITLE */

.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
    animation: fadeIn 0.8s ease;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 30px;
}

/* CHAT */

.stChatMessage {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 14px;
    margin-bottom: 14px;
    backdrop-filter: blur(14px);
    animation: fadeUp 0.3s ease;
}

/* CHAT INPUT */

.stChatInput input {
    background: rgba(255,255,255,0.04) !important;
    color: white !important;
}

/* FILE UPLOADER */

.stFileUploader {
    border: 2px dashed rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 12px;
    background: rgba(255,255,255,0.02);
}

/* BUTTONS */

.stButton button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 12px;
    font-weight: 600;

    background:
    linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color: white;
    transition: 0.25s ease;
}

.stButton button:hover {
    transform: scale(1.02);
    opacity: 0.95;
}

/* ANSWER */

.answer-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    line-height: 1.7;
    animation: fadeUp 0.3s ease;
}

/* DOCUMENT STATUS */

.doc-box {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 14px;
    border-radius: 14px;
    color: #bbf7d0;
    margin-top: 12px;
    font-size: 0.92rem;
    animation: fadeUp 0.4s ease;
}

/* WARNING */

.warning-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.18);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    color: #fecaca;
    margin-top: 90px;
    animation: pulseGlow 2s infinite;
}

/* SOURCE BOX */

.source-box {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 12px;
}

/* ANIMATIONS */

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

@keyframes pulseGlow {
    0% {
        box-shadow: 0 0 0px rgba(239,68,68,0.2);
    }
    50% {
        box-shadow: 0 0 25px rgba(239,68,68,0.3);
    }
    100% {
        box-shadow: 0 0 0px rgba(239,68,68,0.2);
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY")
)

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY")
    st.stop()


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are an AI PDF assistant.

STRICT RULES:
- Answer ONLY from provided document context
- Do NOT use external knowledge
- Do NOT hallucinate
- Do NOT guess
- If answer is unavailable in document respond EXACTLY:
Not available in document
"""

SIMILARITY_THRESHOLD = 0.35
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# =========================================================
# INITIALIZE MODELS
# =========================================================

client = Groq(api_key=GROQ_API_KEY)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📄 PDF Assistant")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    st.markdown("")

    if st.button("Reset Chat"):

        st.session_state.chat_history = []

        st.rerun()

    if st.button("Remove Documents"):

        st.session_state.vector_index = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.uploaded_docs = []

        st.session_state.uploader_key += 1

        st.rerun()

    if st.session_state.uploaded_docs:

        st.markdown("### Uploaded Documents")

        for doc in st.session_state.uploaded_docs:

            st.markdown(
                f"""
                <div class="doc-box">
                ✅ {doc}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
PDF AI Assistant
</div>

<div class="main-subtitle">
Ask intelligent questions from multiple PDF documents
</div>
""", unsafe_allow_html=True)


# =========================================================
# PDF PROCESSING
# =========================================================

def extract_text_from_pdf(pdf_path, filename):

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            cleaned_text = text.replace("\n", " ").strip()

            documents.append({
                "page": page_number + 1,
                "text": cleaned_text,
                "source": filename
            })

    return documents


def chunk_documents(documents):

    chunks = []

    for doc in documents:

        text = doc["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": doc["page"],
                "source": doc["source"]
            })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def create_vector_index(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_chunks(question):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = st.session_state.vector_index.search(
        question_embedding.astype("float32"),
        TOP_K
    )

    relevant_chunks = []

    for i, idx in enumerate(indices[0]):

        if idx >= len(st.session_state.chunks):
            continue

        distance = distances[0][i]

        similarity = 1 / (1 + distance)

        if similarity >= SIMILARITY_THRESHOLD:

            relevant_chunks.append({
                "text": st.session_state.chunks[idx]["text"],
                "page": st.session_state.chunks[idx]["page"],
                "source": st.session_state.chunks[idx]["source"]
            })

    return relevant_chunks


# =========================================================
# ANSWER GENERATION
# =========================================================

def generate_answer(question, relevant_chunks):

    if not relevant_chunks:
        return "Not available in document"

    context = "\n\n".join([
        f"""
Document: {chunk['source']}
Page: {chunk['page']}

{chunk['text']}
"""
        for chunk in relevant_chunks
    ])

    # CONVERSATION MEMORY

    previous_messages = []

    recent_history = st.session_state.chat_history[-6:]

    for msg in recent_history:

        previous_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(previous_messages)

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}
"""
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        max_tokens=450
    )

    answer = response.choices[0].message.content.strip()

    if not answer:
        return "Not available in document"

    return answer


# =========================================================
# PROCESS PDFs
# =========================================================

if uploaded_files:

    current_files = sorted(
        [file.name for file in uploaded_files]
    )

    if current_files != st.session_state.uploaded_docs:

        with st.spinner("Processing documents..."):

            all_documents = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.read())

                    temp_pdf_path = temp_file.name

                documents = extract_text_from_pdf(
                    temp_pdf_path,
                    uploaded_file.name
                )

                all_documents.extend(documents)

            chunks = chunk_documents(
                all_documents
            )

            vector_index = create_vector_index(
                chunks
            )

            st.session_state.vector_index = vector_index
            st.session_state.chunks = chunks
            st.session_state.uploaded_docs = current_files


# =========================================================
# EMPTY STATE
# =========================================================

if st.session_state.vector_index is None:

    st.markdown("""
    <div class="warning-box">
    📄 Please upload one or more PDF documents to begin chatting
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your documents..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    if st.session_state.vector_index is None:

        st.warning("Please upload PDF documents first.")

        st.stop()

    # USER MESSAGE

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # ASSISTANT RESPONSE

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            relevant_chunks = retrieve_chunks(
                question
            )

            answer = generate_answer(
                question,
                relevant_chunks
            )

            st.markdown(
                f"""
                <div class="answer-box">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            # SOURCE REFERENCES

            if answer != "Not available in document":

                unique_sources = sorted(
                    list(set([
                        f"{chunk['source']} (Page {chunk['page']})"
                        for chunk in relevant_chunks
                    ]))
                )

                st.caption(
                    "📄 Sources: " +
                    ", ".join(unique_sources)
                )

                # SOURCE CONTEXT

                with st.expander("View Source Context"):

                    for chunk in relevant_chunks:

                        st.markdown(
                            f"""
                            <div class="source-box">

                            <b>{chunk['source']}</b>
                            <br>
                            Page {chunk['page']}

                            <br><br>

                            {chunk['text'][:500]}...

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # SAVE ASSISTANT MESSAGE

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })