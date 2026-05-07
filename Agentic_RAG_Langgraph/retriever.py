import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def load_invitee_docs(path: str = "invitees.json") -> list[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return [
        Document(
            page_content="\n".join(
                [
                    f"Name: {x['name']}",
                    f"Relation: {x['relation']}",
                    f"Description: {x['description']}",
                    f"Email: {x['email']}",
                ]
            ),
            metadata={"name": x["name"]},
        )
        for x in data
    ]


def build_guest_retriever(path: str = "invitees.json", k: int = 3):
    docs = load_invitee_docs(path)
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever