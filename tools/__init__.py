import os
import requests
from typing import Dict, Any, List, Optional, Callable

class ExternalTool:
    """外部工具基类，用于封装HTTP API调用"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 base_url: str = None, headers: Dict[str, str] = None):
        """
        初始化外部工具
        
        Args:
            name: 工具名称
            description: 工具描述（用于LLM理解）
            parameters: 参数描述（JSON Schema格式）
            base_url: API基础URL
            headers: 请求头
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.base_url = base_url or os.getenv("EXTERNAL_API_BASE_URL", "http://localhost:8000")
        self.headers = headers or {
            "Content-Type": "application/json"
        }
        # 如果环境变量中有API_KEY，则添加到请求头
        api_key = os.getenv("EXTERNAL_API_KEY")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def to_function_schema(self) -> Dict[str, Any]:
        """转换为OpenAI function schema格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() if v.get("required", False)]
            }
        }
    
    def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用外部API
        
        Args:
            endpoint: API端点路径
            params: 请求参数
            
        Returns:
            API响应结果
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.post(url, json=params, headers=self.headers)
            response.raise_for_status()  # 如果响应状态码不是200，抛出异常
            return {
                "success": True,
                "data": response.json()
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }

class ToolRegistry:
    """工具注册表，用于管理所有可用的外部工具"""
    
    def __init__(self):
        self.tools: Dict[str, ExternalTool] = {}
    
    def register_tool(self, tool: ExternalTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的function schema列表"""
        return [tool.to_function_schema() for tool in self.tools.values()]
    
    def get_tool(self, name: str) -> Optional[ExternalTool]:
        """根据名称获取工具"""
        return self.tools.get(name)

# 工具调用处理器
def handle_tool_call(tool_call: Dict[str, Any], registry: ToolRegistry) -> Dict[str, Any]:
    """
    处理工具调用
    
    Args:
        tool_call: 工具调用信息
        registry: 工具注册表
        
    Returns:
        调用结果
    """
    tool_name = tool_call.get("name")
    tool_params = tool_call.get("arguments", {})
    
    # 获取工具实例
    tool = registry.get_tool(tool_name)
    if not tool:
        return {
            "success": False,
            "error": f"工具 {tool_name} 不存在"
        }
    
    # 调用工具（这里假设endpoint与工具名称相同，实际可能需要映射）
    result = tool.call(endpoint=tool_name, params=tool_params)
    return result

# 从impl模块导入并注册工具
def create_tool_registry() -> ToolRegistry:
    """创建并初始化工具注册表"""
    registry = ToolRegistry()
    
    # 动态导入实现目录中的工具
    # 这里可以根据需要导入不同的工具实现
    try:
        from .impl.weather_tool import WeatherTool
        from .impl.stock_tool import StockTool
        from .impl.calculator_tool import CalculatorTool
        
        # 注册工具实例
        registry.register_tool(WeatherTool())
        registry.register_tool(StockTool())
        registry.register_tool(CalculatorTool())
        # 注意：TimeTool已被移除，时间功能现在直接内置在time_agent中
    except ImportError as e:
        print(f"警告: 无法导入工具实现: {e}")
    
    return registry

# 创建全局工具注册表实例
tool_registry = create_tool_registry()