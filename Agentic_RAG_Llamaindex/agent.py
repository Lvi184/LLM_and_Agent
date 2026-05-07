import os
from dotenv import load_dotenv

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike

from tools import ALL_TOOLS


load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("请在 .env 设置 DASHSCOPE_API_KEY")


SYSTEM_PROMPT = """
You are Alfred, a sophisticated gala assistant and Agentic RAG agent.

Rules:
1. If the user asks about guest name, relation, biography, interests, or email,
   call guest_info_retriever first.
2. If the user asks about weather, fireworks, or outdoor planning,
   call weather_info.
3. If the user asks about recent facts, companies, AI models, or public news,
   call web_search or model_stats_search.
4. Do not invent guest emails, relationships, or facts not found in the local guest database.
5. Clearly distinguish local guest database information from external web information.
6. Answer in Chinese by default unless the user asks otherwise.
"""


def build_llm():
    return OpenAILike(
        model="glm-5.1",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=API_KEY,
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0.2,
    )


def build_agent():
    llm = build_llm()

    return FunctionAgent(
        tools=ALL_TOOLS,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )


alfred = build_agent()