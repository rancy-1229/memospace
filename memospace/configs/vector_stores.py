from pydantic import BaseModel, Field
from typing import Optional


class VectorStoreConfig(BaseModel):
    """向量存储配置基类"""
    provider: str = Field(default="qdrant", description="向量存储提供商")
    host: str = Field(default="localhost", description="主机地址")
    port: int = Field(default=6333, description="端口号")
    collection_name: str = Field(default="memospace", description="集合名称")
    api_key: Optional[str] = Field(None, description="API 密钥（如果需要）")
