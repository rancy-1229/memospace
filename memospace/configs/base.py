import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from memospace.configs.llms import LlmConfig
from memospace.configs.embeddings import EmbedderConfig
from memospace.configs.vector_stores import VectorStoreConfig
from memospace.configs.rerankers import RerankerConfig
from memospace.search.hybrid_search import HybridStrategy

# 设置默认目录
home_dir = os.path.expanduser("~")
memospace_dir = os.environ.get("MEMOSPACE_DIR") or os.path.join(home_dir, ".memospace")


class MemoryItem(BaseModel):
    """单条记忆的数据模型"""
    id: str = Field(..., description="记忆的唯一标识符")
    memory: str = Field(..., description="记忆内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加的元数据")
    score: Optional[float] = Field(None, description="相关性得分")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")


class MemoryConfig(BaseModel):
    """记忆模块的整体配置"""
    vector_store: VectorStoreConfig = Field(
        description="向量存储配置",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="LLM 配置",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Embedding 配置",
        default_factory=EmbedderConfig,
    )
    history_db_path: Optional[str] = Field(
        description="历史记录数据库路径",
        default=None,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Reranker 配置",
        default_factory=RerankerConfig,
    )
    enable_entities: bool = Field(
        description="是否启用实体提取和存储",
        default=True,
    )
    entity_db_path: Optional[str] = Field(
        description="实体数据库路径",
        default=None,
    )
    entity_boost_factor: float = Field(
        description="搜索时实体匹配的提升因子",
        default=1.5,
    )
    # 混合搜索配置
    enable_hybrid_search: bool = Field(
        description="是否启用混合搜索",
        default=True,
    )
    hybrid_strategy: HybridStrategy = Field(
        description="混合搜索策略",
        default=HybridStrategy.LINEAR,
    )
    bm25_weight: float = Field(
        description="BM25 搜索结果的权重",
        default=0.5,
    )
    vector_weight: float = Field(
        description="向量搜索结果的权重",
        default=0.5,
    )
    rrf_k: int = Field(
        description="RRF 混合策略的 k 参数",
        default=60,
    )
    normalize_hybrid_scores: bool = Field(
        description="混合搜索时是否归一化分数",
        default=True,
    )
    # BM25 配置
    bm25_k1: float = Field(
        description="BM25 的 k1 参数",
        default=1.5,
    )
    bm25_b: float = Field(
        description="BM25 的 b 参数",
        default=0.75,
    )


class HistoryItem(BaseModel):
    """历史记录项"""
    id: str = Field(..., description="历史记录 ID")
    memory_id: str = Field(..., description="记忆 ID")
    old_memory: Optional[str] = Field(None, description="旧记忆内容")
    new_memory: Optional[str] = Field(None, description="新记忆内容")
    event: str = Field(..., description="事件类型（add, update, delete）")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    is_deleted: bool = Field(False, description="是否已删除")
    actor_id: Optional[str] = Field(None, description="操作人 ID")
    role: Optional[str] = Field(None, description="角色")


class EntityItem(BaseModel):
    """实体项"""
    id: str = Field(..., description="实体 ID")
    text: str = Field(..., description="实体文本")
    entity_type: str = Field(..., description="实体类型")
    ner_label: Optional[str] = Field(None, description="NER 标签")
    canonical: str = Field(..., description="规范化文本")
    confidence: Optional[float] = Field(None, description="置信度")
    memory_count: Optional[int] = Field(None, description="关联的记忆数量")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
