import operator
from typing import Annotated, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish

class AgentState(TypedDict):
    input: str  # initial human problem
    agent_outcome: Union[AgentAction, AgentFinish, None] # hold the output of create_react_agent
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

