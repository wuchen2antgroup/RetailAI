#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
意图识别Agent
负责识别用户输入的意图类型，如时间相关或非时间相关
支持当无法判断用户意图时向用户提问以获取更多信息
"""

from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from config.config import config

# 创建意图识别的系统提示
INTENT_RECOGNITION_SYSTEM_PROMPT = """
你是一个专业的意图识别器，负责判断用户问题的类型。

你的任务是：
1. 分析用户的问题
2. 判断该问题是否与时间或日期相关
3. 如果与时间/日期相关，请回复：时间相关
4. 如果不相关，请回复：非时间相关
5. 如果无法明确判断，请回复：无法判断

时间相关的问题包括但不限于：
- 询问当前时间、日期、星期几
- 询问特定事件的时间或日期
- 询问时区转换
- 询问时间差计算
- 询问日期推算（如：今天之后3天是几号）

请只回复"时间相关"、"非时间相关"或"无法判断"，不要添加任何其他内容。
"""

# 定义意图类型枚举
class IntentType:
    """意图类型枚举类"""
    TIME_RELATED = "时间相关"
    NON_TIME_RELATED = "非时间相关"
    UNKNOWN = "未知"
    CANNOT_DETERMINE = "无法判断"

# 创建澄清问题的系统提示
CLARIFICATION_SYSTEM_PROMPT = """
你是一个专业的对话助手，当无法确定用户意图时，负责向用户提问以获取更多信息。

你的任务是：
1. 分析用户的模糊问题
2. 生成一个简洁、礼貌的问题，帮助用户明确表达他们的意图
3. 特别关注时间相关的可能性，引导用户确认是否与时间有关

例如：
- 用户输入: "今天怎么样？"
  你的回复: "您是想了解今天的时间，还是其他方面的信息呢？"

- 用户输入: "明天安排？"
  你的回复: "您是想查询明天的具体日期和时间，还是关于明天的日程安排呢？"

请保持问题简洁友好，不要直接提供答案，而是引导用户提供更明确的信息。
"""

class IntentAgent:
    """
    意图识别Agent类
    负责识别用户输入的意图类型，并在需要时向用户提问以获取更多信息
    """
    
    def __init__(self):
        """
        初始化意图识别Agent
        """
        # 初始化OpenAI客户端
        self.client = OpenAI(**config.get_openai_client_kwargs())
        self.system_prompt = INTENT_RECOGNITION_SYSTEM_PROMPT
        self.clarification_prompt = CLARIFICATION_SYSTEM_PROMPT
    
    def recognize_intent(self, user_input: str) -> str:
        """
        识别用户输入的意图
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            意图类型，可能的值为："时间相关"、"非时间相关"、"无法判断"或"未知"
        """
        try:
            # 构建消息列表
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # 调用OpenAI API进行意图识别
            response = self.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0
            )
            
            # 获取并清理结果
            intent_result = response.choices[0].message.content.strip()
            
            # 验证结果格式是否符合预期
            if intent_result in [IntentType.TIME_RELATED, IntentType.NON_TIME_RELATED, IntentType.CANNOT_DETERMINE]:
                return intent_result
            else:
                # 如果结果不符合预期，返回未知
                print(f"警告：意图识别返回非预期结果: {intent_result}")
                return IntentType.UNKNOWN
                
        except Exception as e:
            # 处理识别过程中的错误
            print(f"意图识别错误: {str(e)}")
            return IntentType.UNKNOWN
    
    def generate_clarification_question(self, user_input: str) -> str:
        """
        为模糊的用户输入生成澄清问题
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            用于向用户澄清意图的问题
        """
        try:
            # 构建消息列表
            messages = [
                {"role": "system", "content": self.clarification_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # 调用OpenAI API生成澄清问题
            response = self.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=50,
                temperature=0.7  # 设置适当的temperature以生成更自然的问题
            )
            
            # 获取并返回澄清问题
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # 处理生成过程中的错误，返回默认问题
            print(f"生成澄清问题错误: {str(e)}")
            return "请问您是想了解与时间相关的信息，还是其他方面的内容呢？"
    
    def is_time_related(self, user_input: str) -> bool:
        """
        判断用户输入是否与时间相关
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            True表示与时间相关，False表示与时间无关或识别失败
        """
        intent = self.recognize_intent(user_input)
        return intent == IntentType.TIME_RELATED
    
    def process_intent(self, user_input: str) -> Tuple[str, Optional[str]]:
        """
        处理用户输入的意图，返回意图类型和可能的澄清问题
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            一个元组，第一个元素是意图类型，第二个元素是澄清问题（如果需要的话）
        """
        intent = self.recognize_intent(user_input)
        
        # 如果无法判断意图，生成澄清问题
        if intent == IntentType.CANNOT_DETERMINE or intent == IntentType.UNKNOWN:
            clarification_question = self.generate_clarification_question(user_input)
            return intent, clarification_question
        
        # 如果可以明确判断意图，返回意图类型和None
        return intent, None

# 创建意图识别Agent实例
intent_agent = IntentAgent()

# 辅助函数，方便直接调用
def recognize_intent(user_input: str) -> str:
    """
    直接识别用户输入的意图
    
    Args:
        user_input: 用户输入的文本
        
    Returns:
        意图类型
    """
    return intent_agent.recognize_intent(user_input)

def is_time_related(user_input: str) -> bool:
    """
    直接判断用户输入是否与时间相关
    
    Args:
        user_input: 用户输入的文本
        
    Returns:
        是否与时间相关
    """
    return intent_agent.is_time_related(user_input)

def process_intent(user_input: str) -> Tuple[str, Optional[str]]:
    """
    处理用户输入的意图，返回意图类型和可能的澄清问题
    
    Args:
        user_input: 用户输入的文本
        
    Returns:
        一个元组，第一个元素是意图类型，第二个元素是澄清问题（如果需要的话）
    """
    return intent_agent.process_intent(user_input)