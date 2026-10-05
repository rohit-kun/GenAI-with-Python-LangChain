import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Task Manager AI",
    page_icon="✅",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    .task-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .task-subtitle {
        color: #777;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Database
# --------------------------------------------------

@st.cache_resource
def get_database():
    db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

    db.run(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT CHECK (
                status IN ('pending', 'in_progress', 'completed')
            ) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    return db


# --------------------------------------------------
# Agent
# --------------------------------------------------

@st.cache_resource
def get_agent():

    db = get_database()

    llm_model = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    toolkit = SQLDatabaseToolkit(
        db=db,
        llm=llm_model
    )

    tools = toolkit.get_tools()

    memory = InMemorySaver()

    system_prompt = """
    You are a task management assistant that interacts with a SQL database
    containing a 'tasks' table.

    TASK RULES:

    1. Limit SELECT queries to 10 results max with
       ORDER BY created_at DESC.

    2. After CREATE/UPDATE/DELETE, confirm the operation with
       a SELECT query.

    3. If the user requests a list of tasks, present the output
       in a structured Markdown table.

    CRUD OPERATIONS:

    CREATE:
        INSERT INTO tasks(title, description, status)

    READ:
        SELECT * FROM tasks
        WHERE ...
        ORDER BY created_at DESC
        LIMIT 10

    UPDATE:
        UPDATE tasks
        SET status=?
        WHERE id=? OR title=?

    DELETE:
        DELETE FROM tasks
        WHERE id=? OR title=?

    Table schema:

    id
    title
    description
    status
    created_at

    Valid status values:
    - pending
    - in_progress
    - completed

    Always use the SQL tools to interact with the database.
    Do not invent task information.
    """

    agent = create_agent(
        model=llm_model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
    )

    return agent


# --------------------------------------------------
# Initialize Agent
# --------------------------------------------------

agent = get_agent()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Task Manager")

    st.markdown("---")

    st.markdown(
        """
        ### What can I do?

        - ➕ Create tasks
        - 📋 List tasks
        - 🔄 Update task status
        - 🗑️ Delete tasks
        - 🔎 Search tasks
        - 💬 Ask questions about your tasks
        """
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="task-header">✅ AI Task Manager</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="task-subtitle">'
    'Manage your tasks using natural language.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

query = st.chat_input(
    "Ask me to create, update, delete, or list tasks..."
)


# --------------------------------------------------
# Process User Query
# --------------------------------------------------

if query:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "streamlit-task-manager"
                        }
                    }
                )

                result = response["messages"][-1].content

                st.markdown(result)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result
                    }
                )

            except Exception as e:

                error_message = (
                    "❌ Something went wrong.\n\n"
                    f"`{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )
