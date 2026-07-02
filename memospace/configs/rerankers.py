from typing import Optional
from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    """Reranker 配置基类"""
    provider: str = Field(default="cohere", description="Reranker 提供商")
    model: str = Field(default="rerank-english-v3.0", description="Reranker 模型")
    api_key: Optional[str] = Field(None, description="API 密钥")
    top_n: int = Field(default=10, description="重排序后返回的结果数量")
