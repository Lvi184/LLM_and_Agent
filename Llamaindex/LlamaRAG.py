# ======================
# 全量可运行：百炼 LLM + RAG 检索
# 无报错、不空返回、国内直连
# ======================
import os
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# 关闭 LlamaTrace，避免 401
# os.environ["PHOENIX_DISABLE"] = "true"

# ======================
# 全局统一配置（关键！解决空返回）
# ======================
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 256
Settings.chunk_overlap = 32

# ======================
# 测试文档
# ======================
documents = [
    Document(
        text=(
            "阿里云百炼是阿里云推出的大模型服务平台，提供通义千问系列模型 API 调用能力，"
            "支持文本生成、对话、知识库检索、工具调用等多种能力，国内可稳定低延迟访问。"
        ),
        metadata={"source": "demo"}
    )
]

# ======================
# 向量库
# ======================
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("llama_index_demo")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ======================
# 建立索引（简化版，最稳）
# ======================
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)

# ======================
# 百炼模型（稳定版）
# ======================
from llama_index.llms.dashscope import DashScope

llm = DashScope(
    model_name="glm-5.1",  # 最稳、最快、不会空返回
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
)

# 全局设置 LLM（关键！）
Settings.llm = llm

# ======================
# 查询引擎
# ======================
query_engine = index.as_query_engine()

# ======================
# 测试提问
# ======================
if __name__ == "__main__":
    print("正在提问：阿里云百炼是什么？")
    response = query_engine.query("阿里云百炼是什么？")
    
    print("\n===== 最终回答 =====")
    print(response)