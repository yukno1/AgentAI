import langchain.agents
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# result = llm.invoke('Give me a fact about cats')
#
# print(result)

search_tools = TavilySearchResults(search_depth='basic')

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time in specified format."""

    formatted_time = datetime.datetime.now().strftime(format)
    return formatted_time

tools = [search_tools, get_system_time]

agent = initialize_agent(tools=tools, llm=llm, agent=langchain.agents.AgentType('zero-shot-react-description'), verbose=True)

agent.invoke("When was SpaceX's last launch and how many days are from this instant?")