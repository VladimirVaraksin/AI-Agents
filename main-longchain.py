# Initialize our LLM
import os
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
#API Key from OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

#to evaluate math expressions
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

search = DuckDuckGoSearchRun()

#memory in RAM
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next: str

#Agents
def create_planner(state: AgentState) -> AgentState:
    prompt = """
    You are a planning agent that decides which tool to use next.
    Available tools:
        - search: Use DuckDuckGo to search the internet
        - calculator: Evaluate mathematical expressions
        - finish: Complete the task and return the final answer
        
    Based on the conversation history, decide which tool to use next or if we should finish.
    Respond with just one word: 'search', 'calculator', or 'finish'.
    """
    question = HumanMessage(content=prompt)
    response = llm.invoke(state["messages"] + [question])
    next = response.content.strip().lower()
    reply = AIMessage(content=next)
    return {
        "next": next,
        "messages": [question, reply]
    }

def create_tool_executor(state: AgentState) -> AgentState:
    tools = {
        "calculator": calculator,
        "search": search
    }
    tool_name = state["next"]
    if tool_name not in tools:
        return state.copy()

    tool = tools[tool_name]

    #Generate tool input

    prompt  = f"Given the conversation, what should we use the {tool_name} tool for? Respond with just the input for the tool."
    question = HumanMessage(content=prompt)
    response = llm.invoke(state["messages"] + [question])
    tool_input = response.content.strip()
    reply = AIMessage(content=tool_input)

    #Execute Tool
    tool_output = tool.invoke(tool_input)
    tool_reply = AIMessage(content=f"Using {tool_name}: {tool_output}")

    return {
        "next": "",
        "messages": [question, reply, tool_reply]
    }

def create_final_answer(state: AgentState) -> AgentState:
    prompt = "Please provide a final answer based on the conversation history."
    question = HumanMessage(content=prompt)
    response = llm.invoke(state["messages"] + [question])
    reply = AIMessage(content=response.content)
    return {
        "next": "",
        "messages": [question, reply]
    }


def build_graph():
    workflow = StateGraph(AgentState)
    #add nodes
    workflow.add_node("planner", create_planner)
    workflow.add_node("executor", create_tool_executor)
    workflow.add_node("final_answer", create_final_answer)

    # add edges
    workflow.add_edge("executor", "planner")
    workflow.add_conditional_edges(
        "planner",
        lambda state: state["next"],
        {
            "finish": "final_answer",
            "calculator": "executor",
            "search": "executor"
        }
    )
    # set entry point
    workflow.set_entry_point("planner")
    workflow.set_finish_point("final_answer")

    return workflow.compile()

if __name__ == "__main__":
    graph = build_graph()
    graph.get_graph().draw_png("langgraph_visualization.png")

    query = "What is the population of Augsburg divided by 2"

    state = {"messages": [HumanMessage(content=query)], "next": ""}
    result = graph.invoke(state)
    print("Final Answer: ", result["messages"][-1].content)

    print("\nHistory")
    for message in result["messages"]:
        message.pretty_print()
        print()