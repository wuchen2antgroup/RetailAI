import os
from dotenv import load_dotenv

class Config:
    """配置管理类，用于管理LLM相关配置"""
    
    def __init__(self):
        """初始化配置，加载环境变量"""
        load_dotenv()
        self._load_config()
    
    def _load_config(self):
        """加载配置项"""
        # OpenAI/LLM相关配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model_name = os.getenv("MODEL_NAME", "qwen3-max")
    
    def get_openai_client_kwargs(self):
        """获取OpenAI客户端配置参数字典"""
        client_kwargs = {
            "api_key": self.openai_api_key,
        }
        
        # 如果设置了BASE_URL环境变量，则添加到客户端配置中
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        return client_kwargs
    
    def reload(self):
        """重新加载配置"""
        self._load_config()

# 创建全局配置实例
config = Config()