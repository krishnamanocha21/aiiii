from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# this is the model
model =ChatOpenAI(model="gpt-5-nano")
#this is the search tool
search =GoogleSerperAPIWrapper(serper_api_key="12fe0a2d3739456b643598ad0911cf48e56e65c3")
#this is the memory /checkpoint -which stores the state after whcih the next chat starts
memory =MemorySaver()

#creating the agent 
agent =create_agent(
    model =model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a agent and can search for any question on google."
)

while True:
    query=input("User:")
    if query.lower()=="quit":
        print("Good Bye")
        break

    response =agent.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"1"}},
    )
    print("AI:",response["messages"][-1].content)