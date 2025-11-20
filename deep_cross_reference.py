import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from schemas import State

load_dotenv()


async def deep_cross_reference(state: State):
    llm = ChatOpenAI(model="gpt-5.1", temperature=0, reasoning_effort="high")
    client = MultiServerMCPClient(
        {
            "bright_data": {
                "url": f"https://mcp.brightdata.com/sse?token={os.getenv('SCRAPER_MCP')}&pro=1",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.
    For each claim, search multiple authoritative sources automatically
    Check news sites, fact-checkers, government sources, academic sources
    For media content, perform reverse image/video searches
    Document all sources and their credibility levels
    """

    agent = create_deep_agent(
        tools=tools, system_prompt=research_instructions, model=llm
    )

    result = await agent.ainvoke({"messages": state["messages"]})

    return {"messages": [result["messages"][-1]]}
