from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatGroq(
    model = "openai/gpt-oss-20b"
)

search = GoogleSerperAPIWrapper()

memory = InMemorySaver()

agent = create_agent(
    model = llm,
    tools = [search.run],
    checkpointer = memory,
    system_prompt = "You are an agent and can search for any question on google.",
    
)

while True:
    query = input("User: ")
    if query.lower() in ["quit", "exit", "bye"]:
        print("GoodBye 👋")
        break

    response = agent.invoke(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable":{"thread_id":"1"}}
    )

    print("AI: ", response["messages"][-1].content)