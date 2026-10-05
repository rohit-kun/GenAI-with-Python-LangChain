## LLM
## TOOL - Google Search Tool
## Agent
## Memory
## Streamlit
## Web Interface


from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

llm = ChatGroq(
    model = "openai/gpt-oss-20b"
)

search = GoogleSerperAPIWrapper()

tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.history = []
#memory = InMemorySaver()

agent = create_agent(
    model = llm,
    tools = tools,
    checkpointer = st.session_state.memory,
    system_prompt = "You are an amazing ai agent and can search on google as well."
)

### Building Web Interface..
st. subheader("QuickAnswer - Answers at the speed of input")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything ?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user", "content":query})

    # response = agent.invoke(
    #     {"messages":[{"role":"user", "content":query}]},
    #     {"configurable":{"thread_id":"1"}}
    # )

    response = agent.stream(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":"1"}},
            stream_mode = "messages"
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)

#print(response["messages"][-1].content)
    # answer = response["messages"][-1].content
    # st.chat_message("ai").markdown(answer)

    #st.session_state.history.append({"role":"ai", "content":answer})

        st.session_state.history.append({"role":"ai", "content":message})
        