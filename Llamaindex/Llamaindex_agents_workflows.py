# ==============================================================================
# ✅ 更接近 HuggingFace workflows.ipynb 完整功能的 LlamaIndex 0.14.x 实现版
# ✅ 保留：
#    - glm-5.1
#    - 离线 BGE
#    - 本地 Chroma
# ✅ 包含：
#    1) Basic Workflow
#    2) Multi-step Workflow
#    3) Loops and Branches
#    4) Draw Workflows
#    5) Context State Management
#    6) Multi-Agent Workflow
# ==============================================================================

import os
import asyncio
import random
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
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Event,
    Context,
    step,
)
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope

# 画 workflow 图
from llama_index.utils.workflow import draw_all_possible_flows


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
# 2) 本地知识库（给 QueryEngineTool / Agent 用）
# ------------------------------------------------------------------------------
RAG_DB_PATH = "./workflow_demo_chroma_db"
RAG_COLLECTION = "workflow_demo_docs"

db = chromadb.PersistentClient(path=RAG_DB_PATH)
chroma_collection = db.get_or_create_collection(RAG_COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

try:
    existing_count = chroma_collection.count()
except Exception:
    existing_count = 0

documents = [
    Document(text="LlamaIndex是用于构建大模型应用的框架，支持RAG、智能体、工具调用和工作流。"),
    Document(text="LlamaIndex Workflow 通过 StartEvent、StopEvent、Event 和 @step 来编排多步骤流程。"),
    Document(text="LlamaIndex 支持 Context 状态管理，可以跨步骤保存与读取信息。"),
    Document(text="LlamaIndex 支持多 Agent Workflow，可将不同能力的 Agent 组合起来完成复杂任务。"),
]

if existing_count == 0:
    index = VectorStoreIndex.from_documents(
        documents,
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
    description="用于查询 LlamaIndex、Workflow、Agent、RAG 等相关知识。",
)


# ------------------------------------------------------------------------------
# 3) 一些基础工具
# ------------------------------------------------------------------------------
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)


# ==============================================================================
# Part A. Basic Workflow
# 对应 notebook: 最简单的 StartEvent -> StopEvent
# ==============================================================================
class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")


# ==============================================================================
# Part B. Multi-step Workflow
# 对应 notebook: 事件在步骤之间传递
# ==============================================================================
class ProcessingEvent(Event):
    intermediate_result: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


# ==============================================================================
# Part C. Loops and Branches
# 对应 notebook: 用返回不同 Event 实现循环与分支
# ==============================================================================
class LoopEvent(Event):
    loop_output: str


class BranchingWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


# ==============================================================================
# Part D. Context / State Management
# 对应 notebook: 在 step 之间用 ctx.store 保存状态
# notebook 已改成 await ctx.store.set(...) / get(...)
# ==============================================================================
class StatefulWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent, ctx: Context) -> ProcessingEvent:
        await ctx.store.set("query", "What is the capital of France?")
        await ctx.store.set("stage", "initialized")
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent, ctx: Context) -> StopEvent:
        query = await ctx.store.get("query")
        stage = await ctx.store.get("stage")
        print(f"Query: {query}")
        print(f"Stage: {stage}")
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


# ==============================================================================
# Part E. 一个“更实用”的 Workflow：把 Workflow + LLM/检索结合
# 这部分不是 notebook 原样代码，但很适合你当前代码体系
# ==============================================================================
class KnowledgeQueryEvent(Event):
    question: str


class KnowledgeWorkflow(Workflow):
    @step
    async def prepare_query(self, ev: StartEvent, ctx: Context) -> KnowledgeQueryEvent:
        user_question = ev.get("user_question", "LlamaIndex 是什么？")
        await ctx.store.set("user_question", user_question)
        return KnowledgeQueryEvent(question=user_question)

    @step
    async def query_knowledge(self, ev: KnowledgeQueryEvent, ctx: Context) -> StopEvent:
        response = query_engine.query(ev.question)
        await ctx.store.set("last_answer", str(response))
        return StopEvent(result=str(response))


# ==============================================================================
# Part F. Multi-Agent Workflow
# 对应 notebook 最后一部分
# 一个 agent 负责乘法，一个 agent 负责加法
# ==============================================================================
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="能够计算两个整数的乘积",
    system_prompt="你是乘法助手。遇到乘法计算时优先使用你的工具。",
    tools=[multiply_tool],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="能够计算两个整数的和",
    system_prompt="你是加法助手。遇到加法计算时优先使用你的工具。",
    tools=[add_tool],
    llm=llm,
)

knowledge_agent = ReActAgent(
    name="knowledge_agent",
    description="能够查询 LlamaIndex / Workflow / Agent 相关知识",
    system_prompt="你是知识查询助手。遇到框架知识问题时优先使用知识库工具。",
    tools=[query_engine_tool],
    llm=llm,
)

multi_agent_workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent, knowledge_agent],
    root_agent="multiply_agent",
)


# ------------------------------------------------------------------------------
# 演示函数
# ------------------------------------------------------------------------------
async def demo_basic_workflow():
    print("\n" + "=" * 80)
    print("A. Basic Workflow")
    print("=" * 80)
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


async def demo_multistep_workflow():
    print("\n" + "=" * 80)
    print("B. Multi-step Workflow")
    print("=" * 80)
    w = MultiStepWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


async def demo_branching_workflow():
    print("\n" + "=" * 80)
    print("C. Loops and Branches")
    print("=" * 80)
    w = BranchingWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


async def demo_draw_workflow():
    print("\n" + "=" * 80)
    print("D. Draw Workflow")
    print("=" * 80)
    w = BranchingWorkflow(timeout=10, verbose=False)
    draw_all_possible_flows(w, filename="workflow_all_flows.html")
    print("已生成: workflow_all_flows.html")


async def demo_stateful_workflow():
    print("\n" + "=" * 80)
    print("E. Stateful Workflow / Context")
    print("=" * 80)
    w = StatefulWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


async def demo_knowledge_workflow():
    print("\n" + "=" * 80)
    print("F. Knowledge Workflow")
    print("=" * 80)
    w = KnowledgeWorkflow(timeout=20, verbose=False)
    result = await w.run(user_question="LlamaIndex Workflow 是什么？")
    print(result)


async def demo_multi_agent_workflow():
    print("\n" + "=" * 80)
    print("G. Multi-Agent Workflow")
    print("=" * 80)

    response1 = await multi_agent_workflow.run(user_msg="Can you add 5 and 3?")
    print("加法测试：", response1)

    response2 = await multi_agent_workflow.run(user_msg="What is 6 multiplied by 7?")
    print("乘法测试：", response2)

    response3 = await multi_agent_workflow.run(
        user_msg="请查询知识库：LlamaIndex 支持哪些工作流能力？"
    )
    print("知识查询测试：", response3)


# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
async def main():
    await demo_basic_workflow()
    await demo_multistep_workflow()
    await demo_branching_workflow()
    await demo_draw_workflow()
    await demo_stateful_workflow()
    await demo_knowledge_workflow()
    await demo_multi_agent_workflow()


if __name__ == "__main__":
    asyncio.run(main())