from typing_extensions import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(model="gpt-4o")


def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI {response.content}")
    # print("CURRENT STATE", state["messages"])

    return state


graph_builder = StateGraph(AgentState)
graph_builder = StateGraph(AgentState)
graph_builder.add_node("process_node", process_node)
graph_builder.add_edge(START, "process_node")
graph_builder.add_edge("process_node", END)


app = graph_builder.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    app.invoke({"messages": conversation_history})
    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your conversation log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Your: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("End of conversation")

print("Conversation save to logging.txt")
