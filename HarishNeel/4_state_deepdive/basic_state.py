from typing import TypedDict
from langgraph.graph import StateGraph, END


class SimpleState(TypedDict):
    count: int


def increment(state: SimpleState) -> SimpleState:
    return {
        "count": state["count"] + 1
    }

def should_continue(state: SimpleState):
    if state["count"] < 5:
        return "continue"
    else:
        return "stop"

graph = StateGraph(SimpleState) # offer blueprint

graph.set_entry_point("increment")
graph.add_node("increment", increment)
graph.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment", # if return "continue", go to "increment
        "stop": END,
    }
)

app = graph.compile()
print(app.get_graph().draw_mermaid())

result = app.invoke({
    "count": 0
})
print(result)
