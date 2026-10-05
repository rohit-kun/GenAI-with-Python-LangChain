import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Initialize Agent
# --------------------------------------------------
@st.cache_resource
def create_ai_agent():

    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    search = GoogleSerperAPIWrapper()

    tools = [search.run]

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=(
            "You are an amazing AI agent. "
            "You can search Google when needed to answer questions. "
            "Give accurate, helpful and concise answers."
        )
    )

    return agent


agent = create_ai_agent()


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.title("🤖 AI Agent")

    st.write(
        "This AI agent can answer questions and "
        "search Google when necessary."
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("🤖 AI Search Agent")

st.caption("Powered by Groq + LangChain + LangGraph + Google Search")


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------
if prompt := st.chat_input("Ask me anything..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                answer = response["messages"][-1].content

                st.markdown(answer)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong:\n\n{str(e)}"
                )
