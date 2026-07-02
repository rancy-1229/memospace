from abc import ABC, abstractmethod
from typing import List


class EmbeddingBase(ABC):
    """Embedding 抽象基类"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        获取单个文本的 embedding（同步）
        
        Args:
            text: 输入文本
            
        Returns:
            embedding 向量
        """
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的 embedding（同步，默认实现，子类可覆盖）
        
        Args:
            texts: 输入文本列表
            
        Returns:
            embedding 向量列表
        """
        return [self.embed(text) for text in texts]

    async def aembed(self, text: str) -> List[float]:
        """
        获取单个文本的 embedding（异步）
        
        Args:
            text: 输入文本
            
        Returns:
            embedding 向量
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embed(text))

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的 embedding（异步，默认实现，子类可覆盖）
        
        Args:
            texts: 输入文本列表
            
        Returns:
            embedding 向量列表
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embed_batch(texts))
