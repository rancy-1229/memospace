from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMBase(ABC):
    """LLM 抽象基类"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        生成响应（同步）
        
        Args:
            messages: 消息列表，每个消息包含 'role' 和 'content'
            **kwargs: 其他参数
            
        Returns:
            生成的响应文本
        """
        pass

    async def agenerate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        生成响应（异步）
        
        Args:
            messages: 消息列表，每个消息包含 'role' 和 'content'
            **kwargs: 其他参数
            
        Returns:
            生成的响应文本
        """
        # 默认实现是在执行器中运行同步版本
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_response(messages, **kwargs)
        )
