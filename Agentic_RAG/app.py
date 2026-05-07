import os
from dotenv import load_dotenv


load_dotenv()

from smolagents import GradioUI, CodeAgent
from tools import WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset
from smolagents import OpenAIServerModel

# === 百炼 OpenAI 兼容模型配置 ===
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("请在 .env 设置 DASHSCOPE_API_KEY")

model = OpenAIServerModel(
    model_id="glm-5.1",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
)

guest_info_tool = load_guest_dataset()
weather_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()

agent = CodeAgent(
    tools=[guest_info_tool, weather_tool, hub_stats_tool],
    model=model,
    add_base_tools=False,
    planning_interval=2,
    instructions=(
        "这是一个嘉宾问答助手，"
        "遇到有关嘉宾信息的问题时优先使用 guest_info_retriever。"
    )
)

# 启动 UI
if __name__ == "__main__":
    GradioUI(agent).launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
    )