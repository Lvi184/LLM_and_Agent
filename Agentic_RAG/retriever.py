from datasets import load_from_disk
from smolagents import Tool
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "离线宾客信息检索: 输入姓名/描述返回宾客详情。"
    )
    inputs = {
        "query": {"type": "string", "description": "搜索条件"},
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.retriever = BM25Retriever.from_documents(docs)
        self.retriever.k = 5

    def forward(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "未找到匹配的宾客信息。"
        return "\n\n".join(doc.page_content for doc in docs)

def load_guest_dataset():
    # dataset = load_dataset("agents-course/unit3-invitees", split="train")
    # dataset.save_to_disk(r"D:\BaiduNetdiskWorkspace\大模型\Hugging_Face_AI_Agents_Course\Agentic_RAG/unit3-invitees")
    dataset = load_from_disk("./unit3-invitees")
    docs = []
    for row in dataset:
        content = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
        docs.append(Document(page_content=content, metadata=dict(row)))
    return GuestInfoRetrieverTool(docs)

