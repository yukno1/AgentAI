from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
import operator


class SimpleState(TypedDict):
    count: int
    # give langgraph metadate tell how to update
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]


def increment(state: SimpleState) -> SimpleState:
    return {
        "count": state["count"] + 1,
        # "sum": state["sum"] + state["count"] + 1,
        "sum": state["count"] + 1,
        "history": [state["count"] + 1]
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

# state = {
#     "count": 0,
#     "sum": 0
# }
state:SimpleState = {'count': 0, 'sum': 0, 'history': []}

result = app.invoke(state)
print(result)
