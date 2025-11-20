from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import MessageLike
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from openai.types.shared.reasoning_effort import ReasoningEffort

from schemas import State

load_dotenv()


def claim_identifier(state: State):
    llm = ChatOpenAI(model="gpt-5.1", temperature=0, reasoning_effort="high")

    system_prompt = """
    Parse extracted content to find specific, verifiable factual claims.
    Ignore opinions, jokes, satire, and subjective statements.
    Extract key facts: statistics, events, quotes, dates, locations.
    Focus on claims that could mislead people if false.
    """

    extracted_info = state["messages"]
    final_messages = [SystemMessage(content=system_prompt), *extracted_info]
    claims = llm.invoke(final_messages)

    return {"messages": claims}
