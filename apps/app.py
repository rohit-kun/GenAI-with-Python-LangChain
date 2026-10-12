import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

st.set_page_config(page_title="RAG Agent", page_icon="📄", layout="wide")

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using retrieved context.
My knowledge base consists of the details from the uploaded documents.
ALWAYS use the `retrieve_context` tool for questions requiring external knowledge.
If the retrieved context does not contain the answer, say so instead of guessing.
"""


# ---------- cached, expensive-to-create objects ----------
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="qwen3-embedding:8b")


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-20b")


# ---------- pipeline pieces ----------
def build_vector_db(uploaded_files):
    """Load every uploaded PDF, split into chunks, embed, and index."""
    all_docs = []
    for f in uploaded_files:
        # PyPDFLoader needs a real path, so write the upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        finally:
            os.remove(tmp_path)

        for page in pages:
            page.metadata["source"] = f.name  # show the real filename, not the temp path
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    return InMemoryVectorStore.from_documents(
        documents=chunks, embedding=get_embeddings()
    )


def build_agent(vector_db):
    """The tool is created here so it is bound to the current vector DB."""

    @tool
    def retrieve_context(query: str):
        """Retrieve documents relevant to a query from the knowledge base."""
        docs = vector_db.similarity_search(query=query, k=4)
        parts = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            label = f"{src}, page {page + 1}" if isinstance(page, int) else src
            parts.append(f"[Source: {label}]\n{doc.page_content}")
        return "\n\n".join(parts)

    return create_agent(
        model=get_llm(),
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


# ---------- session state ----------
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.indexed_files = []


# ---------- sidebar: upload ----------
with st.sidebar:
    st.header("Knowledge source")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading, chunking and embedding..."):
            vector_db = build_vector_db(uploaded_files)
            st.session_state.agent = build_agent(vector_db)
            st.session_state.thread_id = str(uuid.uuid4())  # fresh conversation
            st.session_state.messages = []
            st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.success(f"Indexed {len(uploaded_files)} file(s)")

    if st.session_state.indexed_files:
        st.caption("Currently indexed:")
        for name in st.session_state.indexed_files:
            st.write(f"📄 {name}")

    if st.button("Clear chat", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ---------- main: chat ----------
st.title("📄 Chat with your PDFs")

if st.session_state.agent is None:
    st.info("Upload one or more PDFs in the sidebar and click **Process documents**.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": st.session_state.thread_id}},
            )
            answer = response["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})