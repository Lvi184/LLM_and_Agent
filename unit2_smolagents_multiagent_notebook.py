# %% [markdown]
# # Solving a complex task with a multi-agent hierarchy
# 
# This notebook is part of the [Hugging Face Agents Course](https://www.hf.co/learn/agents-course), a free Course from beginner to expert, where you learn to build Agents.
# 
# ![Agents course share](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png)
# 
# The reception is approaching! With your help, Alfred is now nearly finished with the preparations.
# 
# But now there's a problem: the Batmobile has disappeared. Alfred needs to find a replacement, and find it quickly.
# 
# Fortunately, a few biopics have been done on Bruce Wayne's life, so maybe Alfred could get a car left behind on one of the movie set, and re-engineer it up to modern standards, which certainly would include a full self-driving option.
# 
# But this could be anywhere in the filming locations around the world - which could be numerous.
# 
# So Alfred wants your help. Could you build an agent able to solve this task?
# 
# > 👉 Find all Batman filming locations in the world, calculate the time to transfer via a cargo plane to there, and represent them on a map, with a color varying by a cargo plane transfer time. Also represent some supercar factories with the same cargo plane transfer time.
# 
# Let's build this!

# %%
!pip install 'smolagents[litellm]' plotly geopandas shapely kaleido -q

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
# We first make a tool to get the cargo plane transfer time.
import math
from typing import Optional, Tuple

from smolagents import tool


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,  # Average speed for cargo planes
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)


print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))

# %% [markdown]
# For the model provider, we use Together AI, one of the new [inference providers on the Hub](https://huggingface.co/blog/inference-providers)!
# 
# Regarding the GoogleSearchTool: this requires either having setup env variable `SERPAPI_API_KEY` and passing `provider="serpapi"` or having `SERPER_API_KEY` and passing `provider=serper`.
# 
# If you don't have any Serp API provider setup, you can use `WebSearchTool` but beware that it has a rate limit.

# %%
import os
from PIL import Image
from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel, VisitWebpageTool


model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")

# %% [markdown]
# We can start with creating a baseline, simple agent to give us a simple report.

# %%
task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time."""

# %%
from google.colab import userdata
import os
os.environ["SERPAPI_API_KEY"] = userdata.get('SERPAPI_API_KEY')

# %%
agent = CodeAgent(
    model=model,
    tools=[GoogleSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

# %%
result = agent.run(task)

# %%
result

# %% [markdown]
# We could already improve this a bit by throwing in some dedicated planning steps, and adding more prompting.

# %%
agent.planning_interval = 4

detailed_report = agent.run(f"""
You're an expert analyst. You make comprehensive reports after visiting many websites.
Don't hesitate to search for many queries at once in a for loop.
For each data point that you find, visit the source url to confirm numbers.

{task}
""")

print(detailed_report)

# %%
detailed_report

# %% [markdown]
# Thanks to these quick changes, we obtained a much more concise report by simply providing our agent a detailed prompt, and giving it planning capabilities!
# 
# 💸 But as you can see, the context window is quickly filling up. So **if we ask our agent to combine the results of detailed search with another, it will be slower and quickly ramp up tokens and costs**.
# 
# ➡️ We need to improve the structure of our system.

# %% [markdown]
# ## ✌️ Splitting the task between two agents
# 
# Multi-agent structures allow to separate memories between different sub-tasks, with two great benefits:
# - Each agent is more focused on its core task, thus more performant
# - Separating memories reduces the count of input tokens at each step, thus reducing latency and cost.
# 
# Let's create a team with a dedicated web search agent, managed by another agent.
# 
# The manager agent should have plotting capabilities to redact its final report: so let us give it access to additional imports, including `plotly`, and `geopandas` + `shapely` for spatial plotting.

# %%
model = InferenceClientModel(
    "Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=8096
)

web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)

# %% [markdown]
# The manager agent will need to do some mental heavy lifting.
# 
# So we give it the stronger model [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), and add a `planning_interval` to the mix.

# %%
from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# %%
from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel


def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


manager_agent = CodeAgent(
    model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)

# %% [markdown]
# Let us inspect what this team looks like:

# %%
manager_agent.visualize()

# %%
manager_agent.run("""
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
""")

# %% [markdown]
# I don't know how that went in your run, but in mine, the manager agent skilfully divided tasks given to the web agent in `1. Search for Batman filming locations`, then `2. Find supercar factories`, before aggregating the lists and plotting the map.
# 
# Let's see what the map looks like by inspecting it directly from the agent state:

# %%
manager_agent.python_executor.state["fig"]

# %% [markdown]
# ![output map](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/output_map.png)


