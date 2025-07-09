from typing_extensions import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOpenAI(model="gpt-4o")


def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\n AI: {response.content}")
    return state


graph_builder = StateGraph(AgentState)
graph_builder.add_node("process_node", process_node)
graph_builder.add_edge(START, "process_node")
graph_builder.add_edge("process_node", END)

agent = graph_builder.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
