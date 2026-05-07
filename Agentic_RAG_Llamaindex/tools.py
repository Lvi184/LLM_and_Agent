import random
from duckduckgo_search import DDGS
from llama_index.core.tools import FunctionTool

from retriever import guest_info_retriever


def weather_info(location: str) -> str:
    """
    Fetch dummy weather information for a given location.
    Useful for outdoor planning and fireworks scheduling.
    """
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15, "fireworks": "not suitable"},
        {"condition": "Clear", "temp_c": 25, "fireworks": "suitable"},
        {"condition": "Windy", "temp_c": 20, "fireworks": "use caution"},
    ]

    data = random.choice(weather_conditions)

    return (
        f"Weather in {location}: {data['condition']}, "
        f"{data['temp_c']}°C. Fireworks suitability: {data['fireworks']}."
    )


def web_search(query: str) -> str:
    """
    Search the public web for recent or external information.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No web search results found."

    return "\n\n".join(
        [
            f"Title: {item.get('title')}\n"
            f"URL: {item.get('href')}\n"
            f"Snippet: {item.get('body')}"
            for item in results
        ]
    )


def model_stats_search(author_or_org: str) -> str:
    """
    Search public web for a company's or author's popular AI models.
    This replaces Hugging Face Hub API usage.
    """
    query = f"{author_or_org} most popular AI model downloads"
    return web_search(query)


guest_info_tool = FunctionTool.from_defaults(
    fn=guest_info_retriever,
    name="guest_info_retriever",
    description=(
        "Retrieves detailed information about gala guests from the local "
        "guest database. Use this for guest name, relation, biography, "
        "interests, or email lookup."
    ),
)

weather_info_tool = FunctionTool.from_defaults(
    fn=weather_info,
    name="weather_info",
    description=(
        "Fetches weather information for a location. Use this for outdoor "
        "planning, fireworks, or event scheduling."
    ),
)

web_search_tool = FunctionTool.from_defaults(
    fn=web_search,
    name="web_search",
    description=(
        "Searches the public web for recent or external information."
    ),
)

model_stats_tool = FunctionTool.from_defaults(
    fn=model_stats_search,
    name="model_stats_search",
    description=(
        "Searches public information about an AI company or author's popular "
        "models. This does not use Hugging Face Hub."
    ),
)


ALL_TOOLS = [
    guest_info_tool,
    weather_info_tool,
    web_search_tool,
    model_stats_tool,
]