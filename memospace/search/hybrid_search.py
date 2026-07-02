"""
混合搜索模块

结合 BM25 关键词搜索和向量相似度搜索，提供更好的搜索结果
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math

from memospace.search.bm25 import BM25Result, BM25Index


class HybridStrategy(Enum):
    """混合搜索策略"""
    LINEAR = "linear"  # 线性加权融合
    RRF = "rrf"  # Reciprocal Rank Fusion
    MIN_MAX = "min_max"  # Min-Max 归一化后融合


@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    id: str
    memory: str
    metadata: Dict[str, Any]
    score: float
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class HybridSearch:
    """混合搜索管理器"""
    
    def __init__(
        self,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        strategy: HybridStrategy = HybridStrategy.LINEAR,
        rrf_k: int = 60,
        normalize_scores: bool = True
    ):
        """
        初始化混合搜索管理器
        
        Args:
            bm25_weight: BM25 结果的权重
            vector_weight: 向量搜索结果的权重
            strategy: 混合策略
            rrf_k: RRF 策略的 k 参数
            normalize_scores: 是否对分数进行归一化
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.strategy = strategy
        self.rrf_k = rrf_k
        self.normalize_scores = normalize_scores
        
        # 验证权重
        total_weight = self.bm25_weight + self.vector_weight
        if total_weight <= 0:
            raise ValueError("权重之和必须大于 0")
    
    def _normalize_min_max(
        self,
        scores: List[float],
        eps: float = 1e-10
    ) -> List[float]:
        """Min-Max 归一化"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < eps:
            return [1.0 for _ in scores]
        
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    def _linear_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """
        线性加权融合策略
        
        Args:
            bm25_results: BM25 搜索结果列表，每个结果包含 doc_id, score, text, metadata
            vector_results: 向量搜索结果列表
        
        Returns:
            融合后的 HybridSearchResult 列表
        """
        # 创建分数字典
        bm25_scores: Dict[str, float] = {}
        vector_scores: Dict[str, float] = {}
        
        bm25_docs: Dict[str, Dict[str, Any]] = {}
        vector_docs: Dict[str, Dict[str, Any]] = {}
        
        # 处理 BM25 结果
        for result in bm25_results:
            doc_id = result['doc_id']
            bm25_scores[doc_id] = result['score']
            bm25_docs[doc_id] = {
                'text': result['text'],
                'metadata': result['metadata']
            }
        
        # 处理向量结果
        for result in vector_results:
            doc_id = result['id']
            vector_scores[doc_id] = result['score']
            vector_docs[doc_id] = {
                'memory': result['memory'],
                'metadata': result['metadata'],
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at')
            }
        
        # 收集所有文档 ID
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        # 提取分数进行归一化
        if self.normalize_scores and all_doc_ids:
            bm25_score_list = [bm25_scores.get(doc_id, 0.0) for doc_id in all_doc_ids]
            vector_score_list = [vector_scores.get(doc_id, 0.0) for doc_id in all_doc_ids]
            
            norm_bm25 = self._normalize_min_max(bm25_score_list)
            norm_vector = self._normalize_min_max(vector_score_list)
            
            # 更新分数为归一化后的
            bm25_scores = {
                doc_id: score
                for doc_id, score in zip(all_doc_ids, norm_bm25)
            }
            vector_scores = {
                doc_id: score
                for doc_id, score in zip(all_doc_ids, norm_vector)
            }
        
        # 计算融合分数
        fused_results = []
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            
            # 加权融合
            final_score = (
                self.bm25_weight * bm25_score +
                self.vector_weight * vector_score
            )
            
            # 优先从向量结果获取文档信息
            if doc_id in vector_docs:
                doc_info = vector_docs[doc_id]
            else:
                doc_info = {
                    'memory': bm25_docs[doc_id]['text'],
                    'metadata': bm25_docs[doc_id]['metadata'],
                    'created_at': None,
                    'updated_at': None
                }
            
            fused_results.append(HybridSearchResult(
                id=doc_id,
                memory=doc_info['memory'],
                metadata=doc_info['metadata'],
                score=final_score,
                bm25_score=bm25_score if doc_id in bm25_docs else None,
                vector_score=vector_score if doc_id in vector_docs else None,
                created_at=doc_info.get('created_at'),
                updated_at=doc_info.get('updated_at')
            ))
        
        # 按融合分数降序排序
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
    
    def _rrf_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """
        Reciprocal Rank Fusion (RRF) 策略
        
        公式：score = sum(1 / (k + rank))
        
        Args:
            bm25_results: BM25 搜索结果列表
            vector_results: 向量搜索结果列表
        
        Returns:
            融合后的 HybridSearchResult 列表
        """
        # 收集文档信息
        doc_info: Dict[str, Dict[str, Any]] = {}
        
        # 处理 BM25 结果
        bm25_doc_ids = []
        for i, result in enumerate(bm25_results):
            doc_id = result['doc_id']
            bm25_doc_ids.append(doc_id)
            doc_info[doc_id] = {
                'text': result['text'],
                'metadata': result['metadata'],
                'bm25_score': result['score']
            }
        
        # 处理向量结果
        vector_doc_ids = []
        for i, result in enumerate(vector_results):
            doc_id = result['id']
            vector_doc_ids.append(doc_id)
            doc_info[doc_id] = {
                'memory': result['memory'],
                'metadata': result['metadata'],
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at'),
                'vector_score': result['score'],
                **doc_info.get(doc_id, {})
            }
        
        # 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        
        # 添加 BM25 排名分数
        for rank, doc_id in enumerate(bm25_doc_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )
        
        # 添加向量搜索排名分数
        for rank, doc_id in enumerate(vector_doc_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                self.vector_weight / (self.rrf_k + rank)
            )
        
        # 构建结果
        fused_results = []
        for doc_id, score in rrf_scores.items():
            info = doc_info[doc_id]
            fused_results.append(HybridSearchResult(
                id=doc_id,
                memory=info.get('memory', info.get('text', '')),
                metadata=info['metadata'],
                score=score,
                bm25_score=info.get('bm25_score'),
                vector_score=info.get('vector_score'),
                created_at=info.get('created_at'),
                updated_at=info.get('updated_at')
            ))
        
        # 按 RRF 分数降序排序
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
    
    def search(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[HybridSearchResult]:
        """
        执行混合搜索
        
        Args:
            bm25_results: BM25 搜索结果列表
            vector_results: 向量搜索结果列表
            top_k: 返回结果数量
            
        Returns:
            混合后的 HybridSearchResult 列表
        """
        if not bm25_results and not vector_results:
            return []
        
        if self.strategy == HybridStrategy.LINEAR or self.strategy == HybridStrategy.MIN_MAX:
            results = self._linear_fusion(bm25_results, vector_results)
        elif self.strategy == HybridStrategy.RRF:
            results = self._rrf_fusion(bm25_results, vector_results)
        else:
            results = self._linear_fusion(bm25_results, vector_results)
        
        return results[:top_k]
