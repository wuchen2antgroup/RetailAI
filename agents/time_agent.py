"""
时间Agent模块
用于回答用户关于全球各地时间的问题
"""
import os
import json
from datetime import datetime
import pytz
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
# 移除不需要的导入，使用字典格式的消息
from pydantic import BaseModel, Field

# 内置的时间工具功能
def get_current_time(timezone: str = "Asia/Shanghai", format: Optional[str] = None) -> Dict[str, Any]:
    """
    获取指定时区的当前时间
    
    Args:
        timezone: IANA时区标识符，默认为Asia/Shanghai
        format: 输出格式，可以是'time'（仅时间）、'date'（仅日期）或'both'（时间和日期）
        
    Returns:
        包含当前时间信息的字典
    """
    try:
        # 验证时区有效性
        if timezone not in pytz.all_timezones:
            return {
                "error": f"无效的时区标识符: {timezone}",
                "valid_timezones": ["Asia/Shanghai", "America/New_York", "Europe/London", "Asia/Tokyo", "Australia/Sydney"]
            }
        
        # 获取指定时区的当前时间
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        # 根据格式要求返回不同信息
        result = {
            "timezone": timezone,
            "datetime": current_time.isoformat(),
            "year": current_time.year,
            "month": current_time.month,
            "day": current_time.day,
            "hour": current_time.hour,
            "minute": current_time.minute,
            "second": current_time.second,
            "weekday": current_time.strftime("%A"),
            "formatted_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone_abbreviation": current_time.strftime("%Z")
        }
        
        # 如果指定了format参数，添加相应的格式化信息
        if format == "time" or format == "both":
            result["time"] = current_time.strftime("%H:%M:%S")
        
        if format == "date" or format == "both":
            # 获取中文星期名称
            weekdays = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", 
                       "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", 
                       "Sunday": "星期日"}
            en_weekday = current_time.strftime("%A")
            zh_weekday = weekdays.get(en_weekday, en_weekday)
            result["date"] = current_time.strftime("%Y年%m月%d日 ") + zh_weekday
        
        return result
        
    except Exception as e:
        return {
            "error": f"获取时间失败: {str(e)}"
        }

# 设置工具可用性标志
HAS_TOOL = True
time_tool_instance = None

# 定义Agent状态
class TimeAgentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    next: str = Field(default="decide")
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

# 工具注册表
class TimeToolRegistry:
    def __init__(self):
        self.tools = {}
        # 直接使用内置的get_current_time函数
        self.tools["get_current_time"] = {
            "name": "get_current_time",
            "description": "获取当前的时间和日期信息",
            "function": get_current_time
        }
    
    def get_tool_schema(self):
        """获取工具的JSON Schema"""
        schemas = []
        if HAS_TOOL:
            schemas.append({
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前的时间和日期信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "description": "输出格式，可以是'time'（仅时间）、'date'（仅日期）或'both'（时间和日期）"
                            }
                        },
                        "required": []
                    }
                }
            })
        return schemas

# 创建工具注册表实例
time_tool_registry = TimeToolRegistry()

# 获取模型
client = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME", "qwen3-max"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 更新系统提示，使其更适合处理空映射情况
TIME_AGENT_SYSTEM_PROMPT = """
你是一个专业的全球时间助手，专注于回答用户关于世界各地时间和日期的问题。

你采用清晰的三阶段工作流程回答用户问题：

1. 地理位置提取阶段：
   - 从用户问题中精确识别所有提到的城市和国家名称
   - 支持多个地点的同时识别
   - 处理模糊或不完整的地点描述

2. 时区映射阶段：
   - 将每个识别出的城市或国家精确映射到对应的IANA时区标识符
   - 对于城市，直接映射到其所在时区
   - 对于国家，映射到其主要或首都时区
   - 验证时区的有效性

3. 时间获取阶段：
   - 为每个映射后的时区调用工具获取当前时间
   - 整合多个时区的时间信息
   - 以自然、友好的方式呈现给用户

工作原则：
- 精确性：确保地理位置识别和时区映射的准确性
- 效率：优化工作流程，减少不必要的步骤
- 兼容性：处理各种用户提问方式和格式
- 透明性：清晰展示工作流程的每个步骤

请严格按照上述工作流程，为用户提供准确、全面的时间信息。
"""

# 不再需要从提示词中提取映射数据，现在使用模型的内置知识进行地理位置识别和时区映射

# 决策节点
def decide_next(state: Dict[str, Any]) -> Dict[str, Any]:
    """决定下一步操作：调用工具或直接回答，实现清晰的工作流程"""
    messages = state.get("messages", [])
    
    # 检查是否有工具结果
    if messages and messages[-1].get("role") == "tool":
        # 如果有工具结果，总结回答
        return {"next": "summarize", "messages": messages}
    
    try:
        # 获取用户的问题
        user_question = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_question = msg.get("content", "")
                break
        
        # 完整工作流程：
        # 1. 判断是否是时间相关问题
        time_keywords = ['几点', '时间', '现在', '日期', '几号', '星期', '今天', '明天', '昨天', '时刻', '几点钟', '时区']
        is_time_related = any(keyword in user_question for keyword in time_keywords)
        
        if is_time_related:
            # 完整工作流程：
            # 1. 提取地理位置名称
            locations = extract_locations_from_question(user_question)
            print(f"工作流程步骤1: 从问题中提取到的地理位置 - {locations}")
            
            # 2. 映射时区信息
            locations_and_timezones = []
            for location in locations:
                timezone = map_location_to_timezone(location)
                locations_and_timezones.append((location, timezone))
                print(f"工作流程步骤2: 将{location}映射到时区{timezone}")
            
            # 如果没有提取到位置，使用默认值
            if not locations_and_timezones:
                locations_and_timezones.append(("默认", "Asia/Shanghai"))
                print("工作流程步骤2: 未提取到位置，使用默认时区Asia/Shanghai")
            
            # 3. 获取时间信息（准备工具调用）
            tool_calls = []
            for location, timezone in locations_and_timezones:
                tool_call = {
                    "name": "get_current_time",
                    "arguments": json.dumps({"format": "both", "timezone": timezone}, ensure_ascii=False)
                }
                tool_calls.append(tool_call)
                print(f"工作流程步骤3: 准备获取{location}时区({timezone})的时间信息")
            
            # 添加助手消息（工具调用）
            assistant_message = {
                "role": "assistant",
                "tool_calls": tool_calls
            }
            new_messages = messages.copy()
            new_messages.append(assistant_message)
            
            return {
                "messages": new_messages,
                "requested_locations": locations_and_timezones,
                "next": "call_tool"
            }
        else:
            # 直接回答非时间问题
            new_messages = messages.copy()
            new_messages.append({"role": "assistant", "content": "抱歉，我是时间助手，只能回答与时间和日期相关的问题。"})
            return {
                "messages": new_messages,
                "next": "end"
            }
    except Exception as e:
        # 处理错误
        error_message = f"处理请求时出错: {str(e)}"
        new_messages = messages.copy()
        new_messages.append({"role": "assistant", "content": error_message})
        return {
            "messages": new_messages,
            "next": "end"
        }

# 旧的extract_locations函数已被新的工作流程替代，但保留用于总结节点的回退方案
def extract_locations(user_question: str) -> List[tuple]:
    """
    从用户问题中提取地理位置信息并映射到时区
    结合位置提取和时区映射的完整过程
    """
    try:
        # 步骤1: 提取地理位置名称
        locations = extract_locations_from_question(user_question)
        
        # 步骤2: 为每个位置映射时区
        locations_and_timezones = []
        for location in locations:
            timezone = map_location_to_timezone(location)
            locations_and_timezones.append((location, timezone))
        
        # 如果没有提取到位置，尝试默认使用上海时区
        if not locations_and_timezones:
            locations_and_timezones.append(("默认", "Asia/Shanghai"))
        
        return locations_and_timezones
    except Exception:
        # 出错时返回默认时区
        return [("默认", "Asia/Shanghai")]

# 简单实现位置提取函数
def extract_locations_from_question(question: str) -> List[str]:
    """
    从问题中提取地理位置
    这是一个基础实现，实际应用中可能需要更复杂的NLP处理
    """
    # 简单的位置关键词匹配示例
    locations = []
    
    # 主要城市和国家关键词
    city_keywords = {
        "北京": "北京", "上海": "上海", "广州": "广州", "深圳": "深圳",
        "纽约": "纽约", "洛杉矶": "洛杉矶", "芝加哥": "芝加哥",
        "伦敦": "伦敦", "巴黎": "巴黎", "东京": "东京", "悉尼": "悉尼"
    }
    
    # 检查问题中是否包含已知城市
    for keyword, city in city_keywords.items():
        if keyword in question and city not in locations:
            locations.append(city)
    
    return locations

# 简单实现时区映射函数
def map_location_to_timezone(location: str) -> str:
    """
    将位置映射到对应的IANA时区
    """
    timezone_mapping = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "广州": "Asia/Shanghai",
        "深圳": "Asia/Shanghai",
        "纽约": "America/New_York",
        "洛杉矶": "America/Los_Angeles",
        "芝加哥": "America/Chicago",
        "伦敦": "Europe/London",
        "巴黎": "Europe/Paris",
        "东京": "Asia/Tokyo",
        "悉尼": "Australia/Sydney",
        "默认": "Asia/Shanghai"
    }
    
    return timezone_mapping.get(location, "Asia/Shanghai")

# 调用工具节点
def call_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """调用工具"""
    messages = state.get("messages", [])
    
    # 找到最后一个助手消息中的工具调用
    if not messages or messages[-1].get("role") != "assistant" or "tool_calls" not in messages[-1]:
        return {
            "messages": messages + [{"role": "assistant", "content": "抱歉，无法识别工具调用。"}],
            "next": "end"
        }
    
    tool_calls = messages[-1].get("tool_calls", [])
    new_messages = messages.copy()
    
    for tool_call in tool_calls:
        try:
            # 获取工具名称和参数
            tool_name = tool_call.get("name")
            arguments = json.loads(tool_call.get("arguments", "{}"))
            
            # 调用对应的工具
            if tool_name in time_tool_registry.tools:
                tool_func = time_tool_registry.tools[tool_name]["function"]
                result = tool_func(**arguments)
                
                # 添加工具结果
                tool_result_message = {
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "name": tool_name
                }
                new_messages.append(tool_result_message)
            else:
                # 工具不存在
                tool_result_message = {
                    "role": "tool",
                    "content": f"未知工具: {tool_name}",
                    "name": tool_name
                }
                new_messages.append(tool_result_message)
        except Exception as e:
            # 处理工具调用错误
            tool_result_message = {
                "role": "tool",
                "content": f"工具调用失败: {str(e)}",
                "name": tool_call.get("name", "unknown_tool")
            }
            new_messages.append(tool_result_message)
    
    return {"messages": new_messages, "next": "summarize"}

# 总结节点
def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """总结工具执行结果并生成回答"""
    messages = state.get("messages", [])
    
    # 获取用户问题
    user_question = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_question = msg.get("content", "")
            break
    
    # 获取请求的所有地点
    requested_locations = state.get("requested_locations", [])
    
    try:
        # 构建总结回答
        # 如果有多个地点请求，获取所有地点的时间
        if requested_locations:
            # 构建包含多个地点时间信息的回答
            summaries = []
            
            # 对于所有地点，直接调用fallback_get_current_time获取时间
            for location, timezone in requested_locations:
                location_time_result = fallback_get_current_time(format="both", timezone=timezone)
                if "time" in location_time_result and "date" in location_time_result:
                    summaries.append(f"{location}当前时间是 {location_time_result['time']}，今天是 {location_time_result['date']}")
                elif "time" in location_time_result:
                    summaries.append(f"{location}当前时间是 {location_time_result['time']}")
                elif "date" in location_time_result:
                    summaries.append(f"{location}今天是 {location_time_result['date']}")
            
            # 组合所有总结，使用更自然的连接词
            if len(summaries) == 1:
                summary = summaries[0] + "。"
            elif len(summaries) == 2:
                summary = "；".join(summaries) + "。"
            else:
                # 多个地点时，使用更清晰的格式
                summary = "<br>• ".join([""] + summaries) + "。"
        else:
            # 尝试重新提取地点信息（防止初始提取失败）
            locations = extract_locations(user_question)
            if locations:
                summaries = []
                for location, timezone in locations:
                    location_time_result = fallback_get_current_time(format="both", timezone=timezone)
                    if "time" in location_time_result and "date" in location_time_result:
                        summaries.append(f"{location}当前时间是 {location_time_result['time']}，今天是 {location_time_result['date']}")
                summary = "；".join(summaries) + "。"
            else:
                # 默认使用上海时区
                timezone = "Asia/Shanghai"
                tool_result = fallback_get_current_time(format="both", timezone=timezone)
                
                # 构建回答
                if "time" in tool_result and "date" in tool_result:
                    summary = f"当前时间是 {tool_result['time']}，今天是 {tool_result['date']}。"
                elif "time" in tool_result:
                    summary = f"当前时间是 {tool_result['time']}。"
                elif "date" in tool_result:
                    summary = f"今天是 {tool_result['date']}。"
                else:
                    summary = f"当前时间信息：{str(tool_result)}"
        
        # 添加总结回答
        new_messages = messages.copy()
        new_messages.append({"role": "assistant", "content": summary})
        
        return {
            "messages": new_messages,
            "ai_response": summary,
            "next": "end"
        }
    except Exception as e:
        # 处理总结错误
        error_message = f"无法总结工具结果: {str(e)}"
        new_messages = messages.copy()
        new_messages.append({"role": "assistant", "content": error_message})
        
        return {
            "messages": new_messages,
            "ai_response": error_message,
            "next": "end"
        }

# 创建Agent图
def create_time_agent():
    """创建时间Agent的LangGraph图"""
    # 创建图构建器
    builder = StateGraph(dict)
    
    # 添加节点
    builder.add_node("decide", decide_next)
    builder.add_node("call_tool", call_tool_node)
    builder.add_node("summarize", summarize_node)
    
    # 设置条件边
    builder.set_entry_point("decide")
    
    # 决策边
    builder.add_conditional_edges(
        "decide",
        lambda state: state.get("next", "end"),
        {
            "call_tool": "call_tool",
            "summarize": "summarize",
            "end": END
        }
    )
    
    # 工具调用后的边
    builder.add_edge("call_tool", "summarize")
    
    # 总结后的边
    builder.add_edge("summarize", END)
    
    # 编译图
    graph = builder.compile()
    
    return graph

# 保留fallback_get_current_time函数，确保向后兼容
def fallback_get_current_time(format="both", timezone="Asia/Shanghai"):
    """获取指定时区的当前时间的备选实现（保持向后兼容）"""
    # 直接使用内置的get_current_time函数
    return get_current_time(timezone=timezone, format=format)

# 导出Agent
time_agent = create_time_agent()