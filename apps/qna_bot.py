from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ---------------- STRUCTURED OUTPUT ----------------
class QnaResponse(BaseModel):
    answer:str =Field(
        description ="Answer to the user Question"
    )
    topic:str =Field(
        description="Main Topic of the user question"
    )


# ---------------- MODEL ----------------

llm =ChatOpenAI(model="gpt-5-nano")
structured_llm =llm.with_structured_output(QnaResponse)


# ---------------- STREAMLIT ----------------

st.title("🤖 AskBuddy – AI QnA Bot")
st.markdown("My QnA Bot with LangChain and OpenAI !")

# ---------------- MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages =[]

# display the revious message 
for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)


# ---------------- USER INPUT ----------------

query =st.chat_input("Ask Anything")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    result= structured_llm.invoke(st.session_state.messages)
    # Display structured response (with is used to create a context in which everything you write inside the block will appear inside the same Streamlit chat message.)
    with st.chat_message("ai"):
        st.markdown(result.answer)

        st.caption(f"Topic: {result.topic}")

    st.session_state.messages.append({"role":"ai","content":result.answer})