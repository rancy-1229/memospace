import json
from typing import List, Dict, Any, Optional

from memospace.rerankers.base import RerankerBase
from memospace.llms.base import LLMBase


class LLMReranker(RerankerBase):
    """LLM-based Reranker 实现"""

    _RERANK_PROMPT_TEMPLATE = """You are a helpful assistant that evaluates the relevance of documents to a query.

Query: {query}

Documents to evaluate:
{documents}

Please evaluate each document and provide a relevance score from 0 to 10, where 0 means completely irrelevant and 10 means perfectly relevant.

Return your response in JSON format:
{{
  "scores": [
    {{"index": 0, "score": X}},
    {{"index": 1, "score": Y}}
  ]
}}
"""

    def __init__(self, config, llm: LLMBase):
        super().__init__(config)
        self.llm = llm

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的结果数量（可选）
            
        Returns:
            重排序后的结果列表，包含 index、text 和 relevance_score
        """
        if top_n is None:
            top_n = self.config.top_n
        
        # 构建带序号的文档列表
        formatted_documents = []
        for idx, doc in enumerate(documents):
            formatted_documents.append(f"Document {idx}: {doc}")
        
        documents_str = "\n".join(formatted_documents)
        
        # 构建提示词
        prompt = self._RERANK_PROMPT_TEMPLATE.format(
            query=query,
            documents=documents_str,
        )
        
        # 调用 LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates document relevance."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate_response(messages)
        
        # 解析 JSON 响应
        try:
            # 清理响应
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            result = json.loads(response.strip())
            scores = result.get("scores", [])
            
            # 格式化结果并排序
            scored_results = []
            for score_item in scores:
                idx = score_item.get("index", 0)
                score = score_item.get("score", 0)
                
                # 规范化分数到 0-1
                normalized_score = min(max(score / 10.0, 0.0), 1.0)
                
                scored_results.append({
                    "index": idx,
                    "text": documents[idx],
                    "relevance_score": normalized_score,
                })
            
            # 排序并返回 top_n
            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored_results[:top_n]
            
        except json.JSONDecodeError:
            # 如果解析失败，返回原顺序
            return [
                {
                    "index": idx,
                    "text": doc,
                    "relevance_score": 1.0 / (idx + 1),
                }
                for idx, doc in enumerate(documents[:top_n])
            ]
