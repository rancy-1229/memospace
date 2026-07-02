from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class RerankerBase(ABC):
    """Reranker 抽象基类"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的结果数量（可选）
            
        Returns:
            重排序后的结果列表，每个结果包含 index、text 和 relevance_score
        """
        pass
