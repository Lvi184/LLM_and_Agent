# ==============================================================================
# ✅ 对齐 HuggingFace agent.ipynb
# ✅ 模型改为：阿里云百炼 qwen3.5-omni-flash
# ✅ OpenAI 兼容方式调用
# ==============================================================================

import os
import base64
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请先设置 DASHSCOPE_API_KEY")

# ------------------------------------------------------------------------------
# qwen3.5-omni-flash：文本 + 多模态统一模型
# ------------------------------------------------------------------------------
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.5-omni-flash",
    temperature=0.0,
)

vision_llm = llm


# ------------------------------------------------------------------------------
# 工具 1：图片文字提取
# ------------------------------------------------------------------------------
def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using qwen3.5-omni-flash.

    Args:
        img_path: local image file path.

    Returns:
        Extracted text.
    """
    try:
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract all the text from this image. Return only the extracted text, no explanations.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        response = vision_llm.invoke(message)
        return response.content.strip()

    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


# ------------------------------------------------------------------------------
# 工具 2：除法
# ------------------------------------------------------------------------------
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    if b == 0:
        raise ValueError("b cannot be zero")
    return a / b


tools = [divide, extract_text]

# 注意：如果你运行时报 tool calling 不支持，
# 说明该模型/接口当前没有完整兼容 LangChain bind_tools。
llm_with_tools = llm.bind_tools(
    tools,
    parallel_tool_calls=False,
)


# ------------------------------------------------------------------------------
# LangGraph State
# ------------------------------------------------------------------------------
class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


# ------------------------------------------------------------------------------
# Assistant 节点
# ------------------------------------------------------------------------------
def assistant(state: AgentState):
    textual_description_of_tool = """
extract_text(img_path: str) -> str:
    Extract text from an image file using qwen3.5-omni-flash.

divide(a: int, b: int) -> float:
    Divide a and b.
"""

    image = state["input_file"]

    sys_msg = SystemMessage(
        content=(
            "You are a helpful ReAct agent. "
            "You can analyze images and run computations using the provided tools.\n\n"
            f"Available tools:\n{textual_description_of_tool}\n"
            f"Currently loaded image path: {image}\n\n"
            "When the user asks about the image, use extract_text with the current image path. "
            "When the user asks for arithmetic division, use divide."
        )
    )

    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    return {
        "messages": [response],
        "input_file": state["input_file"],
    }


# ------------------------------------------------------------------------------
# 构建 LangGraph
# ------------------------------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

builder.add_edge("tools", "assistant")

react_graph = builder.compile()


# ------------------------------------------------------------------------------
# 测试 1：数学工具
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("测试 1：Divide 6790 by 5")
    print("=" * 60)

    messages = [
        HumanMessage(content="Divide 6790 by 5")
    ]

    result = react_graph.invoke({
        "messages": messages,
        "input_file": None,
    })

    for m in result["messages"]:
        m.pretty_print()

    print("\n" + "=" * 60)
    print("测试 2：图片文字提取")
    print("=" * 60)

    image_messages = [
        HumanMessage(
            content=(
                "According to the note provided by Mr. Wayne in the provided image, "
                "what is the list of items I should buy for the dinner menu?"
            )
        )
    ]

    image_result = react_graph.invoke({
        "messages": image_messages,
        "input_file": "LangGraph/Batman_training_and_meals.png",
    })

    for m in image_result["messages"]:
        m.pretty_print()