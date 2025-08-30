import asyncio
import os

from mcp import ClientSession
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1"

        self.client = OpenAI(
            api_key=self.github_token,
            base_url=self.base_url,
        )

    async def process_query(self, query: str):
        """调用 OpenAI API 处理用户查询"""
        messages = [
            {
                "role": "system",
                "content": "你是一个智能助手，帮助用户回答问题。",
            },
            {
                "role": "user",
                "content": query,
            }
        ]

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            )
            return response.choice[0].message.content
        except Exception as e:
            return f"调用 OpenAI API 错误: {str(e)}"

    async def connect_to_mock_server(self):
        """模拟 MCP 服务器连接"""
        print("MCP client init, no connected server")

    async def chat_loop(self):
        """交互式聊天循环"""
        print("MCP client chat loop. 'quit' to quit")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                print(f"\n [Mock response] 你说的是: {query}")
            except Exception as e:
                print(f"\n error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_mock_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())