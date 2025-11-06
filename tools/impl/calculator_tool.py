from .. import ExternalTool
from typing import Dict, Any

class CalculatorTool(ExternalTool):
    """计算器工具"""
    
    def __init__(self):
        name = "calculate"
        description = "执行数学计算"
        parameters = {
            "expression": {
                "type": "string",
                "description": "数学表达式，如'2+3*4'、'sin(30)'等",
                "required": True
            }
        }
        super().__init__(name=name, description=description, parameters=parameters)
    
    def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行数学计算"""
        # 实际应用中，这里会调用真实的计算服务或库
        # 为了演示，使用Python内置的eval函数（注意：实际应用中需要进行安全验证）
        expression = params.get("expression", "")
        
        try:
            # 安全检查：只允许基本的数学运算
            allowed_chars = "0123456789+-*/(). "
            for char in expression:
                if char not in allowed_chars:
                    return {
                        "success": False,
                        "error": f"不允许的字符: {char}"
                    }
            
            # 执行计算
            result = eval(expression)
            
            return {
                "success": True,
                "data": {
                    "expression": expression,
                    "result": result
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"计算错误: {str(e)}"
            }