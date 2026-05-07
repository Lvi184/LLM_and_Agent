import random
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from retriever import build_guest_retriever


_guest_retriever = build_guest_retriever()
_search = DuckDuckGoSearchRun()


@tool
def guest_info_retriever(query: str) -> str:
    """根据嘉宾姓名、关系、背景或兴趣检索本地嘉宾信息。"""
    docs = _guest_retriever.invoke(query)
    if not docs:
        return "No matching guest information found."
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def weather_info(location: str) -> str:
    """查询指定地点的模拟天气，用于判断户外活动或烟花是否适合。"""
    data = random.choice(
        [
            {"condition": "Rainy", "temp_c": 15, "fireworks": "不适合"},
            {"condition": "Clear", "temp_c": 25, "fireworks": "适合"},
            {"condition": "Windy", "temp_c": 20, "fireworks": "需要谨慎"},
        ]
    )
    return (
        f"Weather in {location}: {data['condition']}, "
        f"{data['temp_c']}°C. Fireworks suitability: {data['fireworks']}."
    )


@tool
def web_search(query: str) -> str:
    """搜索公开网络信息。适合获取嘉宾外部背景、最新新闻或通用事实。"""
    return _search.invoke(query)


@tool
def organization_profile(name: str) -> str:
    """查询组织/公司概况。这里不连接 Hugging Face Hub，只做普通网络搜索。"""
    return _search.invoke(f"{name} company organization AI model latest")