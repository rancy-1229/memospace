from pydantic import BaseModel, Field
from typing import Optional


class EmbedderConfig(BaseModel):
    """Embedding 配置基类"""
    provider: str = Field(default="openai", description="Embedding 提供商")
    model: str = Field(default="text-embedding-3-small", description="模型名称")
    api_key: Optional[str] = Field(None, description="API 密钥")
