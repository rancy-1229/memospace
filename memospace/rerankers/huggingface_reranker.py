from typing import List, Dict, Any, Optional

from memospace.rerankers.base import RerankerBase


class HuggingFaceReranker(RerankerBase):
    """HuggingFace Reranker 实现"""

    def __init__(self, config):
        super().__init__(config)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(config.model)
        except ImportError:
            raise ImportError("sentence-transformers is required. Please install it with: pip install sentence-transformers")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 HuggingFace CrossEncoder 重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的结果数量（可选）
            
        Returns:
            重排序后的结果列表，包含 index、text 和 relevance_score
        """
        if top_n is None:
            top_n = self.config.top_n
        
        # 构建 query-document 对
        pairs = [[query, doc] for doc in documents]
        
        # 获取分数
        scores = self._model.predict(pairs)
        
        # 排序并选择 top_n
        scored_results = list(enumerate(scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 格式化返回结果
        formatted_results = []
        for idx, score in scored_results[:top_n]:
            formatted_results.append({
                "index": idx,
                "text": documents[idx],
                "relevance_score": float(score),
            })
        
        return formatted_results
