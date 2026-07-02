import os
from typing import List, Dict, Any, Optional

from memospace.rerankers.base import RerankerBase


class CohereReranker(RerankerBase):
    """Cohere Reranker 实现"""

    def __init__(self, config):
        super().__init__(config)
        self._api_key = config.api_key or os.environ.get("COHERE_API_KEY")
        if not self._api_key:
            raise ValueError("Cohere API key is required. Please provide it through config or set COHERE_API_KEY environment variable.")
        
        try:
            import cohere
            self._client = cohere.ClientV2(self._api_key)
        except ImportError:
            raise ImportError("Cohere SDK is required. Please install it with: pip install cohere")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 Cohere 重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的结果数量（可选）
            
        Returns:
            重排序后的结果列表，包含 index、text 和 relevance_score
        """
        if top_n is None:
            top_n = self.config.top_n
        
        results = self._client.rerank(
            model=self.config.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        
        # 格式化返回结果
        formatted_results = []
        for result in results.results:
            formatted_results.append({
                "index": result.index,
                "text": documents[result.index],
                "relevance_score": result.relevance_score,
            })
        
        return formatted_results
