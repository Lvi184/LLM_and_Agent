from langchain_core.messages import HumanMessage
from agent import graph


config = {
    "configurable": {
        "thread_id": "gala-demo-user-001"
    }
}


def ask(question: str):
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    print("\nUSER:", question)
    print("\nALFRED:\n", result["messages"][-1].content)


if __name__ == "__main__":
    ask("Tell me about Lady Ada Lovelace and give me her email.")
    ask("今晚巴黎适合放烟花吗？")
    ask("我要和 Dr. Nikola Tesla 聊无线能量传输，帮我准备一些话题。")
    ask("刚才那位嘉宾还有什么适合闲聊的话题？")