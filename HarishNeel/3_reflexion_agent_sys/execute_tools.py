import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults

# Create DuckDuckGo search tool
duckduckgo_tool = DuckDuckGoSearchResults(max_results=5, output_format="json")

# function to execute search queries from AnswerQuestion tool calls
def execute_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    # Extract messages from state
    messages = state.get("messages", [])
    
    # Get the last AI message with tool calls
    last_ai_message = None
    for msg in reversed(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message:
        return state

    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            # Execute each search query using search tool
            query_results = {}
            for query in search_queries:
                result = duckduckgo_tool.invoke(query)
                query_results[query] = result

            # Create a tool message with the results
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id,
                )
            )

    # Return updated state with new messages
    updated_messages = messages + tool_messages
    return {"messages": updated_messages}

# example usage (commented out to avoid execution on import)
# test_state = [
#     HumanMessage(
#         content="Write about how small business can leverage AI to grow",
#     ),
#     AIMessage(
#         content="",
#         tool_calls=[
#             {
#                 "name": "AnswerQuestion",
#                 "args": {
#                     'answer': '',
#                     'search queries': [
#                         'AI tools for small business',
#                         'AI in small business marketing',
#                         'AI automation for small business',
#                     ],
#                     'reflection': {
#                         'missing': '',
#                         'superfulous': '',
#                    }
#                 },
#                 "id": "call_kpthichFFEmlitHFvFhKy1Ra",
#             }
#         ],
#     )
# ]
# 
# # Execute the tools
