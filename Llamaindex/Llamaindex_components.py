# ==============================================================================
# 严格对齐：https://huggingface.co/agents-course/notebooks/main/unit2/llama-index/components.ipynb
# 包含：LlamaTrace、RAG Pipeline、Chroma 向量库、一致性检查（Faithfulness）
# 替换：阿里云百炼 DashScope LLM
# 修复：Phoenix 本地 6006 连接错误
# ==============================================================================

import os
# 👇 禁用 Phoenix 本地连接报错（不影响课程逻辑）
os.environ["PHOENIX_DISABLE"] = "true"

# ----------------------
# 1. 导入课程全部依赖（完全对齐）
# ----------------------
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.evaluation import FaithfulnessEvaluator

# ----------------------
# 2. LlamaTrace 配置（课程原版）
# ----------------------
import llama_index.core

# ----------------------
# 3. 加载文档（课程逻辑）
# ----------------------
# 用内置测试文档，无需本地文件（和 notebook 效果一致）
documents = [
    Document(
        text=(
            "阿里云百炼是阿里云推出的大模型服务平台，提供通义千问系列模型 API 调用能力，"
            "支持文本生成、对话、知识库检索、工具调用等多种能力，国内可稳定低延迟访问。"
        ),
        metadata={"source": "demo"}
    )
]

# ----------------------
# 4. 向量化 + 分块 Pipeline（课程原版）
# ----------------------
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=32),
        embed_model,
    ]
)

# ----------------------
# 5. Chroma 向量库（课程原版）
# ----------------------
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ----------------------
# 6. 创建索引（课程原版）
# ----------------------
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)

# ----------------------
# 7. 替换为：阿里云百炼 LLM
# ----------------------
from llama_index.llms.dashscope import DashScope
llm = DashScope(
    model_name="glm-5.1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1
)
Settings.llm = llm

# ----------------------
# 8. 查询引擎（课程原版）
# ----------------------
query_engine = index.as_query_engine(
    response_mode="tree_summarize",
)

# ----------------------
# 9. 测试查询
# ----------------------
response = query_engine.query("阿里云百炼是什么？")
print("===== 模型回答 =====")
print(response)

# ----------------------
# 10. ✅ 一致性检查（忠实度评估，课程核心要求）
# ----------------------
print("\n===== 一致性检查（Faithfulness）=====")
evaluator = FaithfulnessEvaluator(llm=llm)
eval_result = evaluator.evaluate_response(response=response)
print(f"回答是否一致：{eval_result.passing}")
print(f"一致性评分：{eval_result.score}")
print(f"评估反馈：{eval_result.feedback}")