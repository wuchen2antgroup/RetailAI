import json
import time
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
from config.config import config
from tools import tool_registry, handle_tool_call
# 导入时间Agent和意图识别Agent
from agents.time_agent import time_agent
from agents.intent_agent import is_time_related, recognize_intent, process_intent, IntentType

# 初始化OpenAI客户端，使用配置类中的参数
client = OpenAI(**config.get_openai_client_kwargs())

# 定义Agent状态类
class AgentState:
    """Agent的状态类"""
    messages: List[Dict[str, Any]] = add_messages()
    user_input: Optional[str] = None
    tool_responses: List[Dict[str, Any]] = []
    next: str = "direct_answer"
    
    def __init__(self, messages=None, user_input=None, tool_responses=None, next=None):
        if messages is not None:
            self.messages = messages
        else:
            self.messages = []
        if user_input is not None:
            self.user_input = user_input
        else:
            self.user_input = None
        if tool_responses is not None:
            self.tool_responses = tool_responses
        else:
            self.tool_responses = []
        if next is not None:
            self.next = next
        else:
            self.next = "direct_answer"
    
    def __setitem__(self, key, value):
        """允许使用字典方式设置属性"""
        setattr(self, key, value)

# 创建工具调用节点
def execute_tool(state: AgentState) -> Dict[str, Any]:
    """执行工具调用的节点"""
    # 创建一个新的消息列表副本
    updated_messages = state.messages.copy()
    
    # 获取最后一条消息（应该包含工具调用）
    last_message = updated_messages[-1]
    
    if last_message.get("tool_calls"):
        tool_call = last_message["tool_calls"][0]
        
        # 正确获取工具名称和参数（从可序列化的字典中）
        try:
            # 从字典格式中获取工具名称和参数
            tool_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]
            
            try:
                # 解析参数
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}
            
            # 构建工具调用字典
            tool_call_dict = {
                "name": tool_name,
                "arguments": arguments
            }
            
            # 调用工具
            result = handle_tool_call(tool_call_dict, tool_registry)
            
            # 将工具结果添加到消息中
            updated_messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result)
            })
        except Exception as e:
            # 处理工具调用错误
            tool_name = "unknown_tool"
            updated_messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps({"success": False, "error": str(e)})
            })
    
    # 返回字典格式的状态更新
    return {"messages": updated_messages}

# 定义决策节点
def should_use_tool(state: AgentState) -> Dict[str, Any]:
    """决定是否使用工具"""
    # 获取系统提示
    system_prompt = {
        "role": "system",
        "content": "你是一个智能助手，可以使用以下工具来回答用户问题：\n\n1. 当用户询问当前时间、日期或星期几等相关信息时，请使用get_current_time工具。\n2. 当用户询问天气信息时，请使用get_weather工具。\n3. 当用户询问股票信息时，请使用get_stock_info工具。\n4. 当用户需要数学计算时，请使用calculate工具。\n\n请根据用户问题选择合适的工具调用，或者直接回答。"
    }
    
    # 构建消息列表 - 如果有user_input，需要添加到messages中
    messages = [system_prompt]
    
    # 如果有历史消息，添加到消息列表
    if hasattr(state, 'messages') and state.messages:
        # 只添加非系统消息到历史中
        for msg in state.messages:
            if msg.get('role') != 'system':  # 避免重复添加系统消息
                messages.append(msg)
    
    # 如果有新的user_input，添加到消息列表
    if hasattr(state, 'user_input') and state.user_input:
        # 检查是否已经有相同的用户消息
        has_same_user_message = False
        for msg in messages:
            if msg.get('role') == 'user' and msg.get('content') == state.user_input:
                has_same_user_message = True
                break
        
        if not has_same_user_message:
            messages.append({"role": "user", "content": state.user_input})
    
    # 获取工具定义
    tools = []
    for tool in tool_registry.tools.values():
        tools.append({
            "type": "function",
            "function": tool.to_function_schema()
        })
    
    # 调用OpenAI API进行决策
    try:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
    except Exception as e:
        # 返回错误响应
        error_message = {"role": "assistant", "content": f"抱歉，处理你的请求时出错: {str(e)}"}
        updated_messages = state.messages.copy() if hasattr(state, 'messages') and state.messages else []
        if hasattr(state, 'user_input') and state.user_input:
            updated_messages.append({"role": "user", "content": state.user_input})
        updated_messages.append(error_message)
        return {"messages": updated_messages, "next": "direct_answer", "ai_response": error_message['content']}
    
    # 确保state有messages属性
    if not hasattr(state, 'messages') or state.messages is None:
        updated_messages = []
    else:
        updated_messages = state.messages.copy()
    
    # 如果有新的user_input，确保已添加到消息列表
    if hasattr(state, 'user_input') and state.user_input:
        # 检查是否已经有相同的用户消息
        has_same_user_message = False
        for msg in updated_messages:
            if msg.get('role') == 'user' and msg.get('content') == state.user_input:
                has_same_user_message = True
                break
        
        if not has_same_user_message:
            updated_messages.append({"role": "user", "content": state.user_input})
    
    # 如果有工具调用，将调用信息添加到消息中（转换为可序列化的格式）
    if hasattr(message, 'tool_calls') and message.tool_calls:
        # 将工具调用对象转换为可序列化的字典
        serializable_tool_calls = []
        for tool_call in message.tool_calls:
            # 创建可序列化的工具调用字典
            serializable_tool_call = {
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            serializable_tool_calls.append(serializable_tool_call)
        
        # 添加可序列化的工具调用信息
        updated_messages.append({
            "role": "assistant",
            "tool_calls": serializable_tool_calls
        })
        return {"messages": updated_messages, "next": "tool"}
    else:
        # 如果没有工具调用，将回答添加到消息中
        content = getattr(message, 'content', '抱歉，我无法提供回答。')
        updated_messages.append({"role": "assistant", "content": content})
        # 返回ai_response以确保测试脚本能获取到回答
        return {"messages": updated_messages, "next": "direct_answer", "ai_response": content}

# 定义回答节点
def generate_answer(state: AgentState) -> Dict[str, Any]:
    """生成最终回答"""
    # 获取最后一条消息
    last_message = None
    if state.messages:
        last_message = state.messages[-1]
    
    # 如果最后一条消息是工具结果，需要总结
    if last_message and last_message["role"] == "tool":
        # 构建消息列表来生成总结
        messages = [
            {"role": "system", "content": "请根据对话历史和工具执行结果，给用户提供一个友好、自然的总结回答。"},
            {"role": "user", "content": "总结以下对话和工具结果：\n" + json.dumps(state.messages)}
        ]
        
        # 调用OpenAI API生成总结
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            max_tokens=200
        )
        
        summary = response.choices[0].message.content
        
        # 将总结添加到消息中
        updated_messages = state.messages.copy()
        updated_messages.append({"role": "assistant", "content": summary})
        
        return {"messages": updated_messages, "ai_response": summary}
    else:
        # 如果最后一条消息是助手回答，直接返回
        if last_message and last_message["role"] == "assistant":
            return {"ai_response": last_message.get("content", "抱歉，我无法提供回答。")}
        else:
            return {"ai_response": "抱歉，我无法提供回答。"}



# 创建Agent图
def create_agent_graph():
    """创建Agent图"""
    # 初始化状态图
    builder = StateGraph(AgentState)
    
    # 添加节点
    builder.add_node("decision", should_use_tool)
    builder.add_node("tool", execute_tool)
    builder.add_node("answer", generate_answer)
    
    # 添加边和条件
    builder.set_entry_point("decision")
    
    # 决策节点的条件分支 - 使用状态中的next字段来决定下一步
    def condition(state):
        # 检查state是否有next属性，如果没有则使用默认值
        return getattr(state, "next", "direct_answer")
    
    builder.add_conditional_edges(
        "decision",
        condition,
        {
            "tool": "tool",
            "direct_answer": "answer"
        }
    )
    
    # 工具调用后直接到回答节点
    builder.add_edge("tool", "answer")
    
    # 创建内存保存器
    memory = MemorySaver()
    
    # 编译图，使用checkpointer保存状态
    graph = builder.compile(checkpointer=memory)
    return graph

# 实现打字机效果的流式输出函数
def typewriter_print(text: str, delay: float = 0.03, prefix: str = "助手: ") -> None:
    """
    以打字机效果流式输出文本
    
    Args:
        text: 要输出的文本
        delay: 每个字符之间的延迟时间（秒）
        prefix: 输出前缀
    """
    print(f"\n{prefix}", end="", flush=True)
    
    # 分段处理文本，保持段落结构
    paragraphs = text.split('\n')
    for i, paragraph in enumerate(paragraphs):
        # 逐字输出每个段落
        for char in paragraph:
            print(char, end="", flush=True)
            # 根据字符类型调整延迟，让输出更自然
            if char in [',', '，', ';', '；', ':', '：']:
                time.sleep(delay * 2)
            elif char in ['.', '。', '!', '！', '?', '？']:
                time.sleep(delay * 3)
            else:
                time.sleep(delay)
        
        # 在段落之间添加换行
        if i < len(paragraphs) - 1:
            print()
    
    # 确保最后有一个换行
    print()

# 运行Agent
def run_agent():
    """运行Agent"""
    print("欢迎使用智能助手！")
    print("我可以回答问题，还支持以下工具：")
    for name, tool in tool_registry.tools.items():
        print(f"- {name}: {tool.description}")
    print("例如: '现在几点了？', '今天几号？', '北京今天天气怎么样？'")
    print("输入'退出'结束对话。")
    
    # 创建Agent图
    agent = create_agent_graph()
    
    # 会话ID（用于记忆）
    thread_id = "user_thread_1"
    print(f"正在使用会话ID: {thread_id}，对话历史将保存在内存中")
    print("您可以在对话中引用之前的问题或信息")
    

    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ")
            
            # 检查是否退出
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("再见！")
                break
            
            # 创建用户消息
            user_message = {"role": "user", "content": user_input}
            
            try:
                # 使用意图识别Agent处理用户输入
                intent_type, clarification_question = process_intent(user_input)
                
                # 调试信息
                # print(f"调试：意图识别结果 = {intent_type}")
                
                # 如果需要澄清，直接向用户提问
                if clarification_question:
                    typewriter_print(clarification_question)
                    # 继续下一轮循环，等待用户提供更多信息
                    continue
                
                # 判断是否为时间相关意图
                is_time_intent = intent_type == IntentType.TIME_RELATED
                
            except Exception as e:
                # 如果意图识别失败，默认使用主Agent处理
                print(f"意图识别出错: {str(e)}，使用主Agent处理")
                is_time_intent = False
            
            if is_time_intent:
                # 使用时间Agent处理
                try:
                    initial_state = {"messages": [user_message], "next": "decide"}
                    result = time_agent.invoke(initial_state)
                    
                    # 输出回答
                    if isinstance(result, dict):
                        if 'ai_response' in result:
                            typewriter_print(result['ai_response'])
                        elif 'messages' in result:
                            messages = result['messages']
                            for msg in reversed(messages):
                                if msg.get('role') == 'assistant' and 'content' in msg:
                                    typewriter_print(msg['content'])
                                    break
                            else:
                                typewriter_print("抱歉，我无法提供时间回答。")
                        else:
                            typewriter_print("抱歉，我无法提供时间回答。")
                    else:
                        typewriter_print("抱歉，时间服务出错。")
                except Exception as e:
                    typewriter_print(f"时间服务出错: {str(e)}")
            else:
                # 使用主Agent处理
                # 不要创建全新的messages列表，而是让LangGraph自动从checkpointer加载历史
                initial_state = {
                    "user_input": user_input,
                    "next": "direct_answer"
                }
                
                # 执行Agent - LangGraph会自动从checkpointer加载历史并保存新状态
                try:
                    result = agent.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
                    
                    # 输出回答
                    if isinstance(result, dict) and 'ai_response' in result:
                        typewriter_print(result['ai_response'])
                    else:
                        # 如果没有ai_response，尝试从消息中获取
                        if isinstance(result, dict) and 'messages' in result:
                            messages = result['messages']
                            for msg in reversed(messages):
                                if msg.get('role') == 'assistant' and 'content' in msg:
                                    typewriter_print(msg['content'])
                                    break
                            else:
                                typewriter_print("抱歉，我无法提供回答。")
                        else:
                            typewriter_print("抱歉，我无法提供回答。")
                except Exception as e:
                    typewriter_print(f"发生错误: {str(e)}")
        
        except KeyboardInterrupt:
            print("\n程序已终止。")
            break
        except Exception as e:
            typewriter_print(f"程序发生错误: {str(e)}")

if __name__ == "__main__":
    run_agent()