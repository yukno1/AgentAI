from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()
endpoint = "https://models.github.ai/inference"
token = os.environ["GITHUB_TOKEN"]

# llm = ChatOllama(
#     model="qwen3:4b",
# )
llm = ChatOpenAI(
    base_url=endpoint,
    api_key=token,
    model="openai/gpt-4.1"
)

search_tool = TavilySearch(search_depth='basic')

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time in specified format."""

    formatted_time = datetime.datetime.now().strftime(format)
    return formatted_time

tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(llm, tools, react_prompt)
