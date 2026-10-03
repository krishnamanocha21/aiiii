from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

#CREATING A SQLITE DATABASE LOCALLY
db =SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL ,
        description TEXT ,
        status TEXT CHECK (status IN ('pending','in_progress', 'completed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

## llm, tools, memory, system_prompt
model =ChatGroq(model="openai/gpt-oss-20b")

#tools
toolkit =SQLDatabaseToolkit(db=db,llm=model)
tools =toolkit.get_tools()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:

1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC.
2. After CREATE, UPDATE, or DELETE, confirm with a SELECT query.
3. If the user requests a list of tasks, present the output as a clean table.

CRUD OPERATIONS:

CREATE: INSERT INTO tasks(title, description, status)
READ: SELECT * FROM tasks WHERE ... LIMIT 10
UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema:
id, title, description, status (pending/in_progress/completed), created_at.
"""

#see i do not want to create the agent every time the code runs so store the agetn in the cache
@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=MemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.subheader("📜 TaskBot - Manage Your Tasks")

#storing the history and printing the old messages
if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt =st.chat_input("Ask me to manage your task")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt })

    with st.chat_message("ai"):
        # pahle ai container banega then spinner chlta rhega and all the process will execute inside it 
        with st.spinner("Processing"):
            #when all the process execute then the spinner stops and the ai chat containers get complete
            response =agent.invoke(
                {"messages":[{"role":"user", "content":prompt}]},
                {"configurable":{"thread_id":"1"}}
            )

            #len of the response willl contain alll the agent tools  call and user input and the final result is in the last index 
            result =response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai","content":result})

