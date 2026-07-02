import os
from typing import List

from openai import OpenAI, AsyncOpenAI

from memospace.embeddings.base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI Embedding 实现"""

    def __init__(self, config):
        super().__init__(config)
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key)
        self._async_client = None

    @property
    def async_client(self):
        """延迟初始化异步客户端"""
        if self._async_client is None:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            self._async_client = AsyncOpenAI(api_key=api_key)
        return self._async_client

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.config.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.config.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    async def aembed(self, text: str) -> List[float]:
        response = await self.async_client.embeddings.create(
            model=self.config.model,
            input=text
        )
        return response.data[0].embedding

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.async_client.embeddings.create(
            model=self.config.model,
            input=texts
        )
        return [item.embedding for item in response.data]
