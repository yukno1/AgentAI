from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
import os
from schema import AnswerQuestion, ReivseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

load_dotenv()
endpoint = "https://models.github.ai/inference"
token = os.environ["GITHUB_TOKEN"]

llm = ChatOpenAI(
    base_url=endpoint,
    api_key=token,
    model="openai/gpt-4.1"
)

# actor prompt
actor_prompt_templates = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries separately** for
            researching improvements. Do not include them inside the reflection. 
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_templates.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice='AnswerQuestion') | pydantic_parser

# revisor section

revise_instruction = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" sections to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words. 
"""

revisor_chain = actor_prompt_templates.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(
    tools=[ReivseAnswer], tool_choice='ReviseAnswer') | pydantic_parser

response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write me a blog post on how small business can leverage AI to grow")]
})

print(response)