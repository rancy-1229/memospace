import os
from typing import List, Dict

from openai import OpenAI, AsyncOpenAI

from memospace.llms.base import LLMBase


class OpenAILLM(LLMBase):
    """OpenAI LLM 实现"""

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

    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        params.update(kwargs)
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    async def agenerate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        params.update(kwargs)
        
        response = await self.async_client.chat.completions.create(**params)
        return response.choices[0].message.content
