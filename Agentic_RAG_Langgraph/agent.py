import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    guest_info_retriever,
    weather_info,
    web_search,
    organization_profile,
)

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("请在 .env 设置 DASHSCOPE_API_KEY")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tools = [
    guest_info_retriever,
    weather_info,
    web_search,
    organization_profile,
]

llm = ChatOpenAI(
    model="glm-5.1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
    temperature=0.2,
)

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
你是 Alfred，一个晚会 Agentic RAG 助手。

规则：
1. 涉及嘉宾姓名、关系、邮箱、背景、兴趣时，调用 guest_info_retriever。
2. 涉及天气、烟花、户外安排时，调用 weather_info。
3. 涉及最新信息、公司、组织、公开背景时，调用 web_search 或 organization_profile。
4. 回答时区分本地嘉宾库信息和外部搜索信息。
5. 不要编造嘉宾库里没有的信息。
6. 默认中文回答。
"""


def assistant_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "agentic-rag-demo-001"}}

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="帮我查一下 Lady Ada Lovelace 的背景和邮箱")
            ]
        },
        config=config,
    )

    print(result["messages"][-1].content)