# ==============================================================================
# ✅ 更接近 HuggingFace agents.ipynb 完整功能的 LlamaIndex 0.14.x 版本
# ✅ 包含：
#    1) AgentWorkflow.from_tools_or_functions
#    2) stream_events() 事件流输出
#    3) Context 多轮对话
#    4) QueryEngineTool RAG Agent
#    5) Multi-Agent Workflow
# ✅ 保留：
#    - glm-5.1
#    - 离线 BGE
#    - 本地 Chroma
# ==============================================================================

import os
import asyncio
import chromadb

# ------------------------------------------------------------------------------
# 环境设置
# ------------------------------------------------------------------------------
os.environ["PHOENIX_DISABLE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ------------------------------------------------------------------------------
# LlamaIndex 核心导入
# ------------------------------------------------------------------------------
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
    ToolCallResult,
    AgentStream,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope


# ------------------------------------------------------------------------------
# 1) LLM + Embedding
# ------------------------------------------------------------------------------
llm = DashScope(
    model_name="glm-5.1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    local_files_only=True,
)

Settings.llm = llm
Settings.embed_model = embed_model


# ------------------------------------------------------------------------------
# 2) 数学工具（比你原来更接近网页：补齐四则运算）
# ------------------------------------------------------------------------------
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


# ------------------------------------------------------------------------------
# 3) 基础工具 Agent（对应 notebook 第一部分）
#    网页里用 AgentWorkflow.from_tools_or_functions(...)
# ------------------------------------------------------------------------------
math_agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[subtract, multiply, divide, add],
    llm=llm,
    system_prompt=(
        "你是一个数学助手。"
        "涉及加减乘除时，必须优先使用提供的工具，"
        "并尽量逐步完成计算。"
    ),
)


# ------------------------------------------------------------------------------
# 4) 构建本地 RAG 知识库 + QueryEngineTool
#    更接近 notebook：先有 vector store，再生成 query engine tool
# ------------------------------------------------------------------------------
RAG_DB_PATH = "./llama_agents_chroma_db"
RAG_COLLECTION = "llama_index_docs"

db = chromadb.PersistentClient(path=RAG_DB_PATH)
chroma_collection = db.get_or_create_collection(RAG_COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 为避免每次运行无限重复写入，这里做一个简易判断
# 如果库里没有数据，则初始化写入；否则直接从向量库加载
try:
    existing_count = chroma_collection.count()
except Exception:
    existing_count = 0

seed_documents = [
    Document(text="LlamaIndex是用于构建大模型应用的框架，支持RAG、智能体、工具调用能力。"),
    Document(text="LlamaIndex支持基于工具的代理系统，可以把函数、查询引擎等封装成工具供智能体调用。"),
    Document(text="LlamaIndex支持Workflow式Agent，可以处理多轮对话、工具调用、流式事件输出，以及多Agent协作。"),
]

if existing_count == 0:
    index = VectorStoreIndex.from_documents(
        seed_documents,
        vector_store=vector_store,
        embed_model=embed_model,
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

query_engine = index.as_query_engine(llm=llm)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="knowledge_base",
    description="用于查询 LlamaIndex 框架、RAG、Agent、工具调用等相关知识。",
    return_direct=False,
)

rag_agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[query_engine_tool],
    llm=llm,
    system_prompt=(
        "你是一个知识库助手。"
        "当用户询问 LlamaIndex、RAG、Agent、工具调用相关内容时，"
        "优先使用知识库工具检索后再回答。"
    ),
)


# ------------------------------------------------------------------------------
# 5) 多 Agent 系统（对应 notebook 最后一部分）
# ------------------------------------------------------------------------------
calculator_agent = ReActAgent(
    name="calculator",
    description="执行基础数学运算",
    system_prompt="你是计算助手。遇到数学运算时必须使用你的工具。",
    tools=[
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=divide),
    ],
    llm=llm,
)

info_lookup_agent = ReActAgent(
    name="info_lookup",
    description="查询与 LlamaIndex 相关的知识",
    system_prompt="你是知识查询助手。遇到知识检索问题时必须使用你的查询工具。",
    tools=[query_engine_tool],
    llm=llm,
)

multi_agent_system = AgentWorkflow(
    agents=[calculator_agent, info_lookup_agent],
    root_agent="calculator",
)


# ------------------------------------------------------------------------------
# 6) 通用：流式打印事件
#    对应 notebook 的 handler.stream_events()
# ------------------------------------------------------------------------------
async def run_and_trace(handler, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print("\n[工具调用]")
            print(f"工具名: {ev.tool_name}")
            print(f"参数:   {ev.tool_kwargs}")
            print(f"结果:   {ev.tool_output}")
        elif isinstance(ev, AgentStream):
            # notebook 里用这个显示 thought process / 流式输出
            print(ev.delta, end="", flush=True)

    resp = await handler
    print("\n\n[最终回答]")
    print(str(resp))
    return resp


# ------------------------------------------------------------------------------
# 7) 演示主函数
# ------------------------------------------------------------------------------
async def main():
    # --------------------------------------------------------------------------
    # A. 基础工具 Agent：与 notebook 第一部分对应
    # --------------------------------------------------------------------------
    handler = math_agent.run("What is (2 + 2) * 2?")
    await run_and_trace(handler, "A. 基础工具 Agent：四则运算 + 事件流")

    # 中文测试
    handler = math_agent.run("20 + (2 * 4) 等于多少？请使用工具分步计算。")
    await run_and_trace(handler, "A2. 中文数学测试")

    # --------------------------------------------------------------------------
    # B. Context 多轮对话：与 notebook 第二部分对应
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("B. Context 多轮对话")
    print("=" * 80)

    ctx = Context(math_agent)

    resp1 = await math_agent.run("My name is Bob.", ctx=ctx)
    print("第1轮：", str(resp1))

    resp2 = await math_agent.run("What was my name again?", ctx=ctx)
    print("第2轮：", str(resp2))

    # 你也可以加一轮中文测试
    resp3 = await math_agent.run("请用中文回答：我刚才叫什么名字？", ctx=ctx)
    print("第3轮：", str(resp3))

    # --------------------------------------------------------------------------
    # C. RAG Agent：与 notebook 第三部分对应
    # --------------------------------------------------------------------------
    handler = rag_agent.run(
        "Search the knowledge base for information about LlamaIndex agent capabilities and summarize them."
    )
    await run_and_trace(handler, "C. RAG Agent：知识检索 + 事件流")

    handler = rag_agent.run("LlamaIndex 是什么？它支持哪些核心能力？")
    await run_and_trace(handler, "C2. 中文 RAG 测试")

    # --------------------------------------------------------------------------
    # D. Multi-Agent：与 notebook 最后一部分对应
    # --------------------------------------------------------------------------
    handler = multi_agent_system.run(user_msg="Can you add 5 and 3?")
    await run_and_trace(handler, "D. Multi-Agent：数学任务")

    handler = multi_agent_system.run(user_msg="请查询知识库：LlamaIndex 支持什么能力？")
    await run_and_trace(handler, "D2. Multi-Agent：知识检索任务")

    handler = multi_agent_system.run(
        user_msg="先算出 12 * 3，再告诉我 LlamaIndex 是否支持工具调用。"
    )
    await run_and_trace(handler, "D3. Multi-Agent：混合任务")


if __name__ == "__main__":
    asyncio.run(main())