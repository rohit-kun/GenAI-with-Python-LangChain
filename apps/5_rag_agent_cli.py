from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  

## load the documents
loader = PyPDFLoader("../data/medical_report.pdf")
docs = loader.load()

## split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
docs = splitter.split_documents(documents=docs)

## embedings and vector DB
embeddings = OllamaEmbeddings(
    model = "qwen3-embedding:8b"
)

vector_db = InMemoryVectorStore.from_documents(
    documents = docs,
    embedding=embeddings
)

## create a agent - tool, llm, prompt
llm = ChatGroq(
    model = "openai/gpt-oss-20b"
)

@tool
def retrive_context(query:str):
    """
    Retrive documents relevent to a query from the knowledge base.
    """
    context = ""
    docs = vector_db.similarity_search(query = query, k=3)
    for doc in docs:
        context = doc.page_content + "\n\n"
    return context

system_prompt = """
You are a helpful assistant that answers questions using retrieved context. 
My knowledge base consists of the details from the uploaded document. 
ALWAYS use the `retrieve_context` tool for questions requiring external knowledge.
"""

memory = InMemorySaver()

agent = create_agent(
    model = llm,
    tools = [retrive_context],
    system_prompt = system_prompt,
    checkpointer = memory
)

while True:
    query = input("User: ")
    if query.lower() in ["quit", "exit", "bye"]:
        print("GoodBye 👋")
        break

    response = agent.invoke(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id":"1"}}
        )
    result = response["messages"][-1].content

    print("AI: ", result)
