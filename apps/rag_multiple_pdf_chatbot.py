from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader ,PyPDFDirectoryLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

# data in the st session (so that it does not get created again)

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded =False

if "agent" not in st.session_state:
    st.session_state.agent =None

if "vector_store" not in st.session_state:
    st.session_state.vector_store =None

if "messages" not in st.session_state:
    st.session_state.messages = []
    

def process_document(path):

    #load the document (for the file use the pypdf loader but for the folder use this )
    loader=PyPDFDirectoryLoader(path)
    docs =loader.load()

    # onlyfor checking ->print("Number of pages loaded:", len(docs))

    #spilt the docs
    splitter =RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)

    print("\nNumber of chunks:", len(docs))
    #embeddding and the vector db
    embeddings =OpenAIEmbeddings(model="text-embedding-3-large")

    #creating the vectoronly if it is not created earlier
    if st.session_state.vector_store is None:
        
        st.session_state.vector_store =InMemoryVectorStore.from_documents(
            documents=docs,
            embedding=embeddings
        )

    #i have to create this because otherwise it si giving the error because we are using it inside the @tool 
    #so the st.sessionstate is not access by the tool directly so we have to create a variable
    vector_store = st.session_state.vector_store  

    #create the agent -llm,tool,prompt
    llm =ChatGroq(model="openai/gpt-oss-20b")

    @tool 
    def retrieve_context(query:str):
        """Retrieve documents relevent to a query from the knowledge base."""

        context="" 
        docs =vector_store.similarity_search(query=query,k=3)
        for doc in docs:
            context +=doc.page_content +"\n\n"

        return context    


    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
    My knowledge base consists of the details from the uploaded document. 
    ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory =MemorySaver()

    #here i create the agent as a part of the function so it is a local variable 
    agent =create_agent(
        model=llm, 
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    #that's why we are storing it in the session state
    st.session_state.agent=agent 
    st.session_state.document_uploaded =True


#uploaded ui

if not st.session_state.document_uploaded:
    uploaded =st.file_uploader(label="Select file s to Upload" ,type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing"):
            path="./data/"
            for file in uploaded:
                # saving the PDF uploaded through Streamlit onto your computer/server.
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            #it will run the ui
            st.rerun()

#chat ui
if st.session_state.document_uploaded and st.session_state.agent:
    #looading the previous messages
    for message in st.session_state.messages:
        role =message.get("role")
        content =message.get("content")
        st.chat_message(role).markdown(content)
    query =st.chat_input("Ask anything related to the Document...")
    if query:
        st.session_state.messages.append({"role":"user","content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})
    