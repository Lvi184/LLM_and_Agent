import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())


def load_invitees(path: str = "invitees.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class GuestRetriever:
    def __init__(self, path: str = "invitees.json", top_k: int = 3):
        self.guests = load_invitees(path)
        self.top_k = top_k

        self.documents = [
            "\n".join(
                [
                    f"Name: {guest['name']}",
                    f"Relation: {guest['relation']}",
                    f"Description: {guest['description']}",
                    f"Email: {guest['email']}",
                ]
            )
            for guest in self.guests
        ]

        self.tokenized_docs = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str) -> str:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_k]

        results = [
            self.documents[index]
            for index, score in ranked
            if score > 0
        ]

        if not results:
            return "No matching guest information found."

        return "\n\n".join(results)


_guest_retriever = GuestRetriever()


def guest_info_retriever(query: str) -> str:
    """
    Retrieve detailed information about gala guests based on name, relation,
    biography, interests, or email.
    """
    return _guest_retriever.search(query)