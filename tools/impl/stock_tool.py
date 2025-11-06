from .. import ExternalTool
from typing import Dict, Any

class StockTool(ExternalTool):
    """股票信息查询工具"""
    
    def __init__(self):
        name = "get_stock_info"
        description = "获取指定股票代码的股票信息"
        parameters = {
            "symbol": {
                "type": "string",
                "description": "股票代码，如600000（浦发银行）",
                "required": True
            },
            "exchange": {
                "type": "string",
                "description": "交易所，可选'sh'（上海）或'sz'（深圳），默认为'sh'",
                "required": False
            }
        }
        super().__init__(name=name, description=description, parameters=parameters)
    
    def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用股票信息查询API"""
        # 实际应用中，这里会调用真实的股票API
        # 为了演示，返回模拟数据
        symbol = params.get("symbol", "")
        exchange = params.get("exchange", "sh")
        full_symbol = f"{exchange.upper()}{symbol}"
        
        # 模拟API调用成功
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "exchange": exchange,
                "full_symbol": full_symbol,
                "name": "示例股票",
                "current_price": 123.45,
                "open_price": 120.00,
                "high_price": 125.00,
                "low_price": 119.50,
                "volume": 1000000,
                "change": 3.45,
                "change_percent": 2.85
            }
        }