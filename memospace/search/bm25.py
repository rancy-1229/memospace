"""
BM25 关键词搜索模块

实现 BM25（Okapi BM25）排名算法
"""

import math
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from memospace.text_processing.lemmatizer import Lemmatizer, ProcessedText


@dataclass
class BM25Result:
    """BM25 搜索结果"""
    doc_id: str
    score: float
    doc_text: str
    metadata: Dict[str, Any] = None


class BM25Index:
    """BM25 索引"""
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        lemmatizer: Lemmatizer = None
    ):
        """
        初始化 BM25 索引
        
        Args:
            k1: BM25 参数 k1，控制词频饱和度
            b: BM25 参数 b，控制文档长度归一化的影响
            lemmatizer: 词形还原器，使用默认的如果不提供
        """
        self.k1 = k1
        self.b = b
        self.lemmatizer = lemmatizer or Lemmatizer(use_nltk=True)
        
        # 索引数据
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> {text, metadata}
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> 文档长度（词数）
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        
        # 倒排索引
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)  # term -> {doc_ids}
        self.term_freq: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id -> freq}
        self.term_doc_count: Dict[str, int] = defaultdict(int)  # term -> 出现的文档数
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        添加文档到索引
        
        Args:
            doc_id: 文档 ID
            text: 文档文本
            metadata: 附加元数据
        """
        # 处理文本
        processed = self.lemmatizer.process(text)
        tokens = processed.filtered_lemmas
        
        # 计算词频
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # 更新索引
        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata or {},
            'processed': processed
        }
        self.doc_lengths[doc_id] = len(tokens)
        
        # 更新倒排索引和词频
        for term, freq in tf.items():
            if doc_id not in self.inverted_index[term]:
                self.inverted_index[term].add(doc_id)
                self.term_doc_count[term] += 1
            self.term_freq[term][doc_id] = freq
        
        # 更新文档计数和平均长度
        self.doc_count = len(self.documents)
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0
    
    def add_documents(self, docs: List[Tuple[str, str, Dict[str, Any]]]):
        """
        批量添加文档
        
        Args:
            docs: 文档列表，每个元素是 (doc_id, text, metadata)
        """
        for doc_id, text, metadata in docs:
            self.add_document(doc_id, text, metadata)
    
    def remove_document(self, doc_id: str):
        """
        从索引中移除文档
        
        Args:
            doc_id: 文档 ID
        """
        if doc_id not in self.documents:
            return
        
        # 获取文档的词项
        processed = self.documents[doc_id]['processed']
        tokens = set(processed.filtered_lemmas)
        
        # 更新索引
        for term in tokens:
            if term in self.inverted_index:
                self.inverted_index[term].discard(doc_id)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
                    del self.term_doc_count[term]
            if term in self.term_freq:
                if doc_id in self.term_freq[term]:
                    del self.term_freq[term][doc_id]
                    if not self.term_freq[term]:
                        del self.term_freq[term]
        
        # 删除文档
        del self.documents[doc_id]
        del self.doc_lengths[doc_id]
        
        # 更新统计信息
        self.doc_count = len(self.documents)
        if self.doc_count > 0:
            total_len = sum(self.doc_lengths.values())
            self.avg_doc_len = total_len / self.doc_count
        else:
            self.avg_doc_len = 0.0
    
    def _score_document(self, query_tokens: List[str], doc_id: str) -> float:
        """
        计算单个文档的 BM25 分数
        
        Args:
            query_tokens: 查询词项列表
            doc_id: 文档 ID
            
        Returns:
            BM25 分数
        """
        score = 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)
        
        for term in query_tokens:
            if term not in self.term_freq or doc_id not in self.term_freq[term]:
                continue
            
            # 获取词频
            f = self.term_freq[term][doc_id]
            df = self.term_doc_count[term]
            
            # 计算 IDF (Inverse Document Frequency)
            if df == 0 or self.doc_count - df + 0.5 <= 0:
                continue
            idf = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1.0
            )
            
            # 计算 BM25 项
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[BM25Result]:
        """
        搜索文档
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            
        Returns:
            BM25Result 列表，按分数降序排列
        """
        if self.doc_count == 0:
            return []
        
        # 处理查询
        processed = self.lemmatizer.process(query)
        query_tokens = processed.filtered_lemmas
        
        if not query_tokens:
            return []
        
        # 收集所有可能相关的文档（包含至少一个查询词项）
        candidate_doc_ids = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_doc_ids.update(self.inverted_index[term])
        
        if not candidate_doc_ids:
            return []
        
        # 计算每个候选文档的分数
        results = []
        for doc_id in candidate_doc_ids:
            score = self._score_document(query_tokens, doc_id)
            if score > 0:
                doc = self.documents[doc_id]
                results.append(BM25Result(
                    doc_id=doc_id,
                    score=score,
                    doc_text=doc['text'],
                    metadata=doc['metadata']
                ))
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def clear(self):
        """清空索引"""
        self.documents.clear()
        self.doc_lengths.clear()
        self.avg_doc_len = 0.0
        self.doc_count = 0
        self.inverted_index.clear()
        self.term_freq.clear()
        self.term_doc_count.clear()
