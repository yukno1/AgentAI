from typing import Dict, Any

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools
from langchain_core.messages import HumanMessage


# class State(MessagesState):
#     pass

graph = StateGraph(MessagesState)
# graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)


graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def event_loop(state: Dict[str, Any]) -> str:
    messages = state.get("messages", [])
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in messages)
    num_iterations = count_tool_visits
    if num_iterations >= MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())

response = app.invoke(
    {"messages": [HumanMessage(content="Write about how small business can leverage AI to grow")]}
)

# Print the final response - look for answer in the last AI message that has tool calls
final_messages = response["messages"]
for msg in reversed(final_messages):
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(msg.tool_calls[0]["args"]["answer"])
        break

print(response, "response")