from pydantic import BaseModel, Field
from typing import Optional


class LlmConfig(BaseModel):
    """LLM 配置基类"""
    provider: str = Field(default="openai", description="LLM 提供商")
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    api_key: Optional[str] = Field(None, description="API 密钥")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=1000, description="最大 token 数")
