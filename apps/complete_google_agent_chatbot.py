from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

llm =ChatGroq(model="openai/gpt-oss-20b",streaming=True)
# Create Google search tool
search=GoogleSerperAPIWrapper(serper_api_key="12fe0a2d3739456b643598ad0911cf48e56e65c3")
tools=[search.run]

if "memory" not in st.session_state:
    #this is used by the agent in this way in the streamlit (directly managed by the langgraph)
    st.session_state.memory=MemorySaver()
    #this is used for priting the chathistory
    st.session_state.history=[]

agent = create_agent(
    model=llm, 
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are a amazing ai agent and can search on google as well"
)

#now building the web interface
st.subheader("QuickAnswer - Answers at the speed of thought")   

#printing the old messages
for message in st.session_state.history:
    role=message["role"]
    content =message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})

    response =agent.stream(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"1"}},
        #does not stream the user's question to the LLM. It controls how the agent's output/messages are streamed back to your Python code.
        stream_mode="messages"
    )

    # Create space for AI response
    ai_container =st.chat_message("ai")
    #everthing inside the with comes under the ai answer part 
    with ai_container:
        #just initalizing the ai answer space
        space =st.empty()
        # Store complete AI response
        message=""

         # Process each streamed chunk
        for chunk in response:
            message =message+chunk[0].content
            space.write(message)

        # the chat is only appended once 
        st.session_state.history.append({"role":"ai","content":message})    