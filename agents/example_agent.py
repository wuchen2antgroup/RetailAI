from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END

# 1. Define tools and model
# 使用ollama
# model = init_chat_model(
#     model="qwen3:1.7b",
#     model_provider="ollama",
#     base_url="http://localhost:11434",
# )
# 使用qwen
model = init_chat_model(
    "qwen3-max",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-1234567890"
)
checkpointer = MemorySaver()

@tool
def get_city_weather(city: str) -> str:
    """获取城市`city`当天的天气情况

    :argument
        city: 城市名称
    """
    if city == "杭州" or city == "杭州市":
        return "晴天"
    else:
        return "雨天"

tools = [get_city_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# 2. Define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# 3. Define model node
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(state["messages"])
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# 4. Define tool node
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# 5. Define end logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# 6. Build and compile the agent
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile(checkpointer=checkpointer)

# 7. Simulate conversation
config = {"configurable": {"thread_id": "1"}}
while True:
    query = input("user:")
    if len(query) > 0:
        print("AI:", end='')
        for chunk in agent.stream(input={"messages": [{"role": "user", "content": query}]},
                                  config=config, stream_mode="messages"):
            print(chunk[0].content, end='')
        print()
    else:
        break