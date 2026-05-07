# ==============================================================================
# ✅ 适配 LlamaIndex 0.14.x / 0.14.21 风格
# ✅ 保留：glm-5.1 + 离线 BGE + Chroma + FunctionTool + QueryEngineTool
# ✅ 使用：workflow 版 ReActAgent
# ==============================================================================
import os
import asyncio

# 关闭追踪 + 强制离线（不联网）
os.environ["PHOENIX_DISABLE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# ----------------------
# ✅ 保留你的模型：glm-5.1
# ----------------------
from llama_index.llms.dashscope import DashScope

llm = DashScope(
    model_name="glm-5.1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1
)

# ----------------------
# ✅ 离线加载已下载 BGE
# ----------------------
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    local_files_only=True
)

Settings.llm = llm
Settings.embed_model = embed_model

# ----------------------
# 1. FunctionTool
# ----------------------
def multiply(a: float, b: float) -> float:
    """用于计算两数乘积"""
    return a * b

def add(a: float, b: float) -> float:
    """用于计算两数之和"""
    return a + b

multiply_tool = FunctionTool.from_defaults(
    fn=multiply,
    name="multiply",
    description="用于计算两个数字的乘积。"
)

add_tool = FunctionTool.from_defaults(
    fn=add,
    name="add",
    description="用于计算两个数字的和。"
)

# ----------------------
# 2. 构建 RAG 知识库 + QueryEngineTool
# ----------------------
documents = [
    Document(text="LlamaIndex是用于构建大模型应用的框架，支持RAG、智能体、工具调用能力。")
]

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("llama_index")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 若你每次运行都 from_documents，可能会重复写入；
# 先保留你的写法，后面我会告诉你更稳的方式
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model,
)

query_engine = index.as_query_engine(llm=llm)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="data_query",
    description="提供数据查询功能。使用详细的纯文本问题作为工具输入。"
)

# ----------------------
# 3. 创建 ReActAgent（0.14.x 正确写法）
# ----------------------
agent = ReActAgent(
    tools=[multiply_tool, add_tool, query_engine_tool],
    llm=llm,
    verbose=True,
)

# 可选：如果你希望多轮对话共享上下文，可以创建 Context
ctx = Context(agent)

# ----------------------
# 4. 调用工具 + 测试
# ----------------------
async def main():
    print("\n===== 测试：数学工具计算 =====")
    response = await agent.run(
        user_msg="20+(2*4)等于多少? 使用工具计算每一步",
        ctx=ctx,
    )
    print("最终回答：", str(response))

    print("\n===== 测试：RAG数据查询 =====")
    response2 = await agent.run(
        user_msg="LlamaIndex是什么？",
        ctx=ctx,
    )
    print("最终回答：", str(response2))

    # 某些版本/返回对象里可能有 tool_calls
    # 为了稳妥，先做兼容判断
    print("\n===== 查看工具调用记录 =====")
    if hasattr(response, "tool_calls"):
        print("工具调用：", response.tool_calls)
    else:
        print("当前返回对象没有直接暴露 tool_calls，可改用 stream_events() 观察工具调用。")

if __name__ == "__main__":
    asyncio.run(main())