from .. import ExternalTool
from typing import Dict, Any

class WeatherTool(ExternalTool):
    """天气查询工具"""
    
    def __init__(self):
        name = "get_weather"
        description = "获取指定城市的天气信息"
        parameters = {
            "city": {
                "type": "string",
                "description": "城市名称",
                "required": True
            },
            "date": {
                "type": "string",
                "description": "查询日期，格式为YYYY-MM-DD，默认为今天",
                "required": False
            }
        }
        super().__init__(name=name, description=description, parameters=parameters)
    
    def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用天气查询API"""
        # 实际应用中，这里会调用真实的天气API
        # 为了演示，返回模拟数据
        city = params.get("city", "")
        date = params.get("date", "今天")
        
        # 模拟API调用成功
        return {
            "success": True,
            "data": {
                "city": city,
                "date": date,
                "temperature": 25,
                "weather": "晴朗",
                "humidity": 60,
                "wind_speed": 15
            }
        }