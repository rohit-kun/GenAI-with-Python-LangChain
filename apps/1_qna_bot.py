from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
import streamlit as st

llm = ChatGroq(
    model = "openai/gpt-oss-20b"
)

st.title("🤖 AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Groq !")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)



query = st.chat_input("Ask anything")

if query:
    st.session_state.messages.append({"role":"user", "content":"query"})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

# while True:
#     query = input("User: ")

#     if query.lower() in ["quit", "exit", "bye"]:
#         print("GoodBye 👋")
#         break

#     res = llm.invoke(query)
#     print(f"AI: {res.content}\n")

# que = "Who is PM of India ?"

# result = llm.invoke(que)

# print(result.content)