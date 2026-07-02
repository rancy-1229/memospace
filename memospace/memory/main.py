import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from memospace.configs.base import MemoryConfig, MemoryItem, HistoryItem, EntityItem
from memospace.memory.prompts import EXTRACTION_PROMPT
from memospace.memory.storage import SQLiteManager
from memospace.memory.entity_store import EntityStore
from memospace.entity_extraction import extract_entities, extract_entities_simple
from memospace.utils.factory import LlmFactory, EmbedderFactory, VectorStoreFactory, RerankerFactory
from memospace.search.bm25 import BM25Index
from memospace.search.hybrid_search import HybridSearch, HybridSearchResult
from memospace.text_processing.lemmatizer import Lemmatizer


class Memory:
    """核心记忆类"""

    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        初始化 Memory 实例
        
        Args:
            config: 配置对象，如果为 None 则使用默认配置
            **kwargs: 可以直接传入配置参数，会覆盖 config 中的值
        """
        if config is None:
            config = MemoryConfig()
        
        # 处理传入的 kwargs
        if "llm_config" in kwargs:
            config.llm = config.llm.__class__(**kwargs["llm_config"])
        if "embedding_config" in kwargs:
            config.embedder = config.embedder.__class__(**kwargs["embedding_config"])
        if "vector_store_config" in kwargs:
            config.vector_store = config.vector_store.__class__(**kwargs["vector_store_config"])
        
        self.config = config
        self.llm = LlmFactory.create(config.llm)
        self.embedder = EmbedderFactory.create(config.embedder)
        self.vector_store = VectorStoreFactory.create(config.vector_store)
        self.history_db = SQLiteManager(config.history_db_path)
        
        # 初始化 Reranker（如果配置了）
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(config.reranker, self.llm)
        
        # 初始化 EntityStore（如果启用了实体功能）
        self.entity_store = None
        if config.enable_entities:
            self.entity_store = EntityStore(config.entity_db_path)
        
        # 初始化混合搜索（如果启用）
        self.bm25_index = None
        self.hybrid_search = None
        self.lemmatizer = None
        if config.enable_hybrid_search:
            self.lemmatizer = Lemmatizer(use_nltk=True)
            self.bm25_index = BM25Index(
                k1=config.bm25_k1,
                b=config.bm25_b,
                lemmatizer=self.lemmatizer
            )
            self.hybrid_search = HybridSearch(
                bm25_weight=config.bm25_weight,
                vector_weight=config.vector_weight,
                strategy=config.hybrid_strategy,
                rrf_k=config.rrf_k,
                normalize_scores=config.normalize_hybrid_scores
            )

    def _extract_facts(self, text: str) -> List[str]:
        """
        使用 LLM 从文本中提取事实
        
        Args:
            text: 输入文本
            
        Returns:
            提取的事实列表
        """
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text}
        ]
        
        response = self.llm.generate_response(messages)
        
        # 尝试解析 JSON
        try:
            # 清理响应，只保留 JSON 部分
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            facts = json.loads(response.strip())
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass
        
        # 如果解析失败，将整个响应作为一个事实
        return [response]

    def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        添加记忆
        
        Args:
            text: 输入文本
            user_id: 用户 ID（可选）
            agent_id: Agent ID（可选）
            metadata: 附加元数据（可选）
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            添加的记忆列表
        """
        # 提取事实
        facts = self._extract_facts(text)
        if not facts:
            return []
        
        # 提取实体（如果启用）
        entities = []
        if self.entity_store:
            entities = extract_entities(text)
        
        # 生成 embedding
        embeddings = self.embedder.embed_batch(facts)
        
        # 准备元数据
        now = datetime.utcnow().isoformat()
        payloads = []
        ids = []
        history_records = []
        
        for fact in facts:
            memory_id = str(uuid.uuid4())
            ids.append(memory_id)
            
            payload = {
                "memory": fact,
                "created_at": now,
                "updated_at": now
            }
            
            if user_id:
                payload["user_id"] = user_id
            if agent_id:
                payload["agent_id"] = agent_id
            if metadata:
                payload.update(metadata)
            
            payloads.append(payload)
            
            # 准备历史记录
            history_records.append({
                "memory_id": memory_id,
                "old_memory": None,
                "new_memory": fact,
                "event": "add",
                "actor_id": actor_id,
                "role": role
            })
            
            # 存储实体关联（如果启用）
            if self.entity_store and entities:
                self.entity_store.add_entities(entities, memory_id, user_id, agent_id)
            
            # 添加到 BM25 索引（如果启用）
            if self.bm25_index:
                bm25_metadata = payload.copy()
                self.bm25_index.add_document(
                    doc_id=memory_id,
                    text=fact,
                    metadata=bm25_metadata
                )
        
        # 存储到向量数据库
        self.vector_store.add(embeddings, payloads, ids)
        
        # 记录历史
        if history_records:
            self.history_db.batch_add_history(history_records)
        
        # 返回 MemoryItem 列表
        return [
            MemoryItem(
                id=id_,
                memory=payload["memory"],
                metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
                created_at=payload["created_at"],
                updated_at=payload["updated_at"]
            )
            for id_, payload in zip(ids, payloads)
        ]

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        enable_entity_boost: bool = True,
    ) -> List[MemoryItem]:
        """
        搜索相关记忆
        
        Args:
            query: 查询文本
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            top_k: 返回结果数量
            filters: 其他过滤条件（可选）
            enable_entity_boost: 是否启用实体增强
            
        Returns:
            相关的记忆列表
        """
        # 构建过滤条件
        search_filters = filters.copy() if filters else {}
        if user_id:
            search_filters["user_id"] = user_id
        if agent_id:
            search_filters["agent_id"] = agent_id
        
        # 提取查询中的实体（如果启用）
        query_entities = []
        entity_memory_ids = set()
        if self.entity_store and enable_entity_boost:
            query_entities = extract_entities(query)
            if query_entities:
                entity_texts = [ent.text for ent in query_entities]
                entity_memory_ids = set(self.entity_store.get_memories_for_entities(entity_texts))
        
        # 生成查询 embedding
        query_embedding = self.embedder.embed(query)
        
        # 搜索（可能获取更多结果用于重排序和实体增强）
        search_top_k = top_k * 4 if (self.reranker or entity_memory_ids) else top_k
        results = self.vector_store.search(query_embedding, search_top_k, search_filters)
        
        # 转换为 MemoryItem 列表并应用实体增强
        memory_items = []
        seen_ids = set()
        for result in results:
            memory_id = result["id"]
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            
            payload = result["payload"]
            score = result["score"]
            
            # 实体增强：如果该记忆与查询中的实体相关，提升分数
            if entity_memory_ids and memory_id in entity_memory_ids:
                score *= self.config.entity_boost_factor
            
            memory_items.append(MemoryItem(
                id=memory_id,
                memory=payload.get("memory", ""),
                metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
                score=score,
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at")
            ))
        
        # 按分数重新排序
        memory_items.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # 如果配置了 Reranker，进行重排序
        if self.reranker and memory_items:
            # 提取文档内容
            documents = [item.memory for item in memory_items]
            
            # 重排序
            rerank_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
            )
            
            # 根据重排序结果重新排序 MemoryItem
            reranked_items = []
            for rerank_result in rerank_results:
                idx = rerank_result["index"]
                item = memory_items[idx]
                # 更新分数为 reranker 的分数
                reranked_item = MemoryItem(
                    id=item.id,
                    memory=item.memory,
                    metadata=item.metadata,
                    score=rerank_result["relevance_score"],
                    created_at=item.created_at,
                    updated_at=item.updated_at
                )
                reranked_items.append(reranked_item)
            
            return reranked_items
        
        return memory_items[:top_k]

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        根据 ID 获取单个记忆
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            记忆对象，如果不存在返回 None
        """
        result = self.vector_store.get(memory_id)
        if not result:
            return None
        
        payload = result["payload"]
        return MemoryItem(
            id=result["id"],
            memory=payload.get("memory", ""),
            metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at")
        )

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        获取所有记忆（可过滤）
        
        Args:
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            limit: 返回数量限制
            filters: 其他过滤条件（可选）
            
        Returns:
            记忆列表
        """
        # 构建过滤条件
        search_filters = filters.copy() if filters else {}
        if user_id:
            search_filters["user_id"] = user_id
        if agent_id:
            search_filters["agent_id"] = agent_id
        
        # 获取所有结果
        results = self.vector_store.get_all(search_filters, limit)
        
        # 转换为 MemoryItem
        memory_items = []
        for result in results:
            payload = result["payload"]
            memory_items.append(MemoryItem(
                id=result["id"],
                memory=payload.get("memory", ""),
                metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at")
            ))
        
        return memory_items

    def update(
        self,
        memory_id: str,
        new_memory: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        更新记忆
        
        Args:
            memory_id: 记忆 ID
            new_memory: 新的记忆内容（可选）
            new_metadata: 新的元数据（可选）
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            是否成功
        """
        existing = self.vector_store.get(memory_id)
        if not existing:
            return False
        
        payload = existing["payload"].copy()
        old_memory = payload.get("memory")
        now = datetime.utcnow().isoformat()
        payload["updated_at"] = now
        
        # 检查是否有实际变化
        has_change = False
        if new_memory and new_memory != old_memory:
            has_change = True
        if new_metadata:
            for key, value in new_metadata.items():
                if payload.get(key) != value:
                    has_change = True
                    break
        
        if not has_change:
            return True
        
        success = False
        if new_memory:
            payload["memory"] = new_memory
            # 重新生成 embedding
            new_embedding = self.embedder.embed(new_memory)
            success = self.vector_store.update(memory_id, payload, new_embedding)
        elif new_metadata:
            payload.update(new_metadata)
            success = self.vector_store.update(memory_id, payload)
        
        # 记录历史
        if success:
            self.history_db.add_history(
                memory_id=memory_id,
                old_memory=old_memory,
                new_memory=payload.get("memory"),
                event="update",
                actor_id=actor_id,
                role=role
            )
        
        return success

    def delete(
        self,
        memory_id: str,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆 ID
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            是否成功
        """
        existing = self.vector_store.get(memory_id)
        if not existing:
            return False
        
        old_memory = existing["payload"].get("memory")
        
        # 先记录历史
        self.history_db.add_history(
            memory_id=memory_id,
            old_memory=old_memory,
            new_memory=None,
            event="delete",
            actor_id=actor_id,
            role=role
        )
        
        # 删除实体关联（如果启用）
        if self.entity_store:
            self.entity_store.delete_entities_for_memory(memory_id)
        
        # 从 BM25 索引中移除（如果启用）
        if self.bm25_index:
            self.bm25_index.remove_document(memory_id)
        
        # 然后删除
        return self.vector_store.delete(memory_id)

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        删除所有记忆（可过滤）
        
        Args:
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            filters: 其他过滤条件（可选）
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            是否成功
        """
        # 构建过滤条件
        delete_filters = filters.copy() if filters else {}
        if user_id:
            delete_filters["user_id"] = user_id
        if agent_id:
            delete_filters["agent_id"] = agent_id
        
        # 获取要删除的记忆并记录历史
        memories_to_delete = self.vector_store.get_all(delete_filters)
        
        if memories_to_delete:
            history_records = []
            for mem in memories_to_delete:
                payload = mem["payload"]
                history_records.append({
                    "memory_id": mem["id"],
                    "old_memory": payload.get("memory"),
                    "new_memory": None,
                    "event": "delete",
                    "actor_id": actor_id,
                    "role": role
                })
            
            self.history_db.batch_add_history(history_records)
            
            # 删除实体关联（如果启用）
            if self.entity_store:
                for mem in memories_to_delete:
                    self.entity_store.delete_entities_for_memory(mem["id"])
            
            # 从 BM25 索引中移除（如果启用）
            if self.bm25_index:
                for mem in memories_to_delete:
                    self.bm25_index.remove_document(mem["id"])
        
        # 删除
        return self.vector_store.delete_all(delete_filters if delete_filters else None)

    def history(self, memory_id: str) -> List[HistoryItem]:
        """
        获取记忆的历史记录
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            历史记录列表
        """
        history_records = self.history_db.get_history(memory_id)
        return [
            HistoryItem(
                id=record["id"],
                memory_id=record["memory_id"],
                old_memory=record["old_memory"],
                new_memory=record["new_memory"],
                event=record["event"],
                created_at=record["created_at"],
                updated_at=record["updated_at"],
                is_deleted=record["is_deleted"],
                actor_id=record["actor_id"],
                role=record["role"]
            )
            for record in history_records
        ]
    
    # 实体相关方法
    def get_entities(self, memory_id: str) -> List[EntityItem]:
        """
        获取指定记忆关联的实体
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            实体列表
        """
        if not self.entity_store:
            return []
        
        entities = self.entity_store.get_entities_for_memory(memory_id)
        return [
            EntityItem(
                id=ent["id"],
                text=ent["text"],
                entity_type=ent["entity_type"],
                ner_label=ent["ner_label"],
                canonical=ent["canonical"],
                confidence=ent.get("confidence"),
                created_at=ent["created_at"],
                updated_at=ent["updated_at"],
            )
            for ent in entities
        ]
    
    def search_entities(self, query: str, limit: int = 20) -> List[EntityItem]:
        """
        搜索实体
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
            
        Returns:
            实体列表
        """
        if not self.entity_store:
            return []
        
        entities = self.entity_store.search_entities(query, limit)
        return [
            EntityItem(
                id=ent["id"],
                text=ent["text"],
                entity_type=ent["entity_type"],
                ner_label=ent["ner_label"],
                canonical=ent["canonical"],
                memory_count=ent.get("memory_count"),
                created_at=ent["created_at"],
                updated_at=ent["updated_at"],
            )
            for ent in entities
        ]
    
    def get_all_entities(
        self,
        entity_type: Optional[str] = None,
        limit: Optional[int] = 100,
    ) -> List[EntityItem]:
        """
        获取所有实体
        
        Args:
            entity_type: 可选的实体类型过滤
            limit: 返回结果数量限制
            
        Returns:
            实体列表
        """
        if not self.entity_store:
            return []
        
        entities = self.entity_store.get_all_entities(entity_type, limit)
        return [
            EntityItem(
                id=ent["id"],
                text=ent["text"],
                entity_type=ent["entity_type"],
                ner_label=ent["ner_label"],
                canonical=ent["canonical"],
                memory_count=ent.get("memory_count"),
                created_at=ent["created_at"],
                updated_at=ent["updated_at"],
            )
            for ent in entities
        ]
    
    def get_memories_by_entity(self, entity_text: str) -> List[MemoryItem]:
        """
        获取与指定实体关联的所有记忆
        
        Args:
            entity_text: 实体文本
            
        Returns:
            记忆列表
        """
        if not self.entity_store:
            return []
        
        memory_ids = self.entity_store.get_memories_for_entity(entity_text)
        memory_items = []
        for memory_id in memory_ids:
            memory_item = self.get(memory_id)
            if memory_item:
                memory_items.append(memory_item)
        return memory_items
    
    def hybrid_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        enable_entity_boost: bool = True,
        use_hybrid: bool = True,
    ):
        """
        混合搜索：结合 BM25 关键词搜索和向量相似度搜索
        
        Args:
            query: 查询文本
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            top_k: 返回结果数量
            filters: 其他过滤条件（可选）
            enable_entity_boost: 是否启用实体提升
            use_hybrid: 是否使用混合搜索（false 时只用向量搜索）
        
        Returns:
            混合搜索结果列表
        """
        # 构建过滤条件
        search_filters = filters.copy() if filters else {}
        if user_id:
            search_filters["user_id"] = user_id
        if agent_id:
            search_filters["agent_id"] = agent_id
        
        # 如果不启用混合搜索或没有 BM25 索引，回退到普通搜索
        if not use_hybrid or not self.bm25_index or not self.hybrid_search:
            return self.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                top_k=top_k,
                filters=filters,
                enable_entity_boost=enable_entity_boost
            )
        
        # 1. 执行 BM25 搜索
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        bm25_dicts = [
            {
                'doc_id': result.doc_id,
                'text': result.doc_text,
                'metadata': result.metadata,
                'score': result.score
            }
            for result in bm25_results
        ]
        
        # 2. 执行向量搜索（同时应用实体提升）
        vector_results_list = self.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k * 2,
            filters=filters,
            enable_entity_boost=enable_entity_boost
        )
        vector_dicts = [
            {
                'id': item.id,
                'memory': item.memory,
                'metadata': item.metadata,
                'score': item.score,
                'created_at': item.created_at,
                'updated_at': item.updated_at
            }
            for item in vector_results_list
        ]
        
        # 3. 混合搜索结果
        hybrid_results = self.hybrid_search.search(
            bm25_results=bm25_dicts,
            vector_results=vector_dicts,
            top_k=top_k
        )
        
        # 转换为 MemoryItem 格式
        final_results = []
        for result in hybrid_results:
            memory_item = MemoryItem(
                id=result.id,
                memory=result.memory,
                metadata=result.metadata,
                score=result.score,
                created_at=result.created_at,
                updated_at=result.updated_at
            )
            final_results.append(memory_item)
        
        return final_results
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 10
    ):
        """
        仅使用 BM25 关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            BM25 搜索结果列表
        """
        if not self.bm25_index:
            return []
        
        bm25_results = self.bm25_index.search(query, top_k=top_k)
        return bm25_results


class AsyncMemory:
    """异步核心记忆类"""

    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        初始化 AsyncMemory 实例
        
        Args:            config: 配置对象，如果为 None 则使用默认配置
            **kwargs: 可以直接传入配置参数，会覆盖 config 中的值
        """
        if config is None:
            config = MemoryConfig()
        
        # 处理传入的 kwargs
        if "llm_config" in kwargs:
            config.llm = config.llm.__class__(**kwargs["llm_config"])
        if "embedding_config" in kwargs:
            config.embedder = config.embedder.__class__(**kwargs["embedding_config"])
        if "vector_store_config" in kwargs:
            config.vector_store = config.vector_store.__class__(**kwargs["vector_store_config"])
        
        self.config = config
        self.llm = LlmFactory.create(config.llm)
        self.embedder = EmbedderFactory.create(config.embedder)
        self.vector_store = VectorStoreFactory.create(config.vector_store)
        self.history_db = SQLiteManager(config.history_db_path)
        
        # 初始化 Reranker（如果配置了）
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(config.reranker, self.llm)

    async def _extract_facts(self, text: str) -> List[str]:
        """
        使用 LLM 从文本中提取事实（异步）
        
        Args:
            text: 输入文本
            
        Returns:
            提取的事实列表
        """
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text}
        ]
        
        response = await self.llm.agenerate_response(messages)
        
        # 尝试解析 JSON
        try:
            # 清理响应，只保留 JSON 部分
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            facts = json.loads(response.strip())
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass
        
        # 如果解析失败，将整个响应作为一个事实
        return [response]

    async def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        添加记忆（异步）
        
        Args:
            text: 输入文本
            user_id: 用户 ID（可选）
            agent_id: Agent ID（可选）
            metadata: 附加元数据（可选）
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            添加的记忆列表
        """
        # 提取事实
        facts = await self._extract_facts(text)
        if not facts:
            return []
        
        # 生成 embedding
        embeddings = await self.embedder.aembed_batch(facts)
        
        # 准备元数据
        now = datetime.utcnow().isoformat()
        payloads = []
        ids = []
        history_records = []
        
        for fact in facts:
            memory_id = str(uuid.uuid4())
            ids.append(memory_id)
            
            payload = {
                "memory": fact,
                "created_at": now,
                "updated_at": now
            }
            
            if user_id:
                payload["user_id"] = user_id
            if agent_id:
                payload["agent_id"] = agent_id
            if metadata:
                payload.update(metadata)
            
            payloads.append(payload)
            
            # 准备历史记录
            history_records.append({
                "memory_id": memory_id,
                "old_memory": None,
                "new_memory": fact,
                "event": "add",
                "actor_id": actor_id,
                "role": role
            })
        
        # 存储到向量数据库
        await self.vector_store.aadd(embeddings, payloads, ids)
        
        # 记录历史
        if history_records:
            self.history_db.batch_add_history(history_records)
        
        # 返回 MemoryItem 列表
        return [
            MemoryItem(
                id=id_,
                memory=payload["memory"],
                metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
                created_at=payload["created_at"],
                updated_at=payload["updated_at"]
            )
            for id_, payload in zip(ids, payloads)
        ]

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        搜索相关记忆（异步）
        
        Args:
            query: 查询文本
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            top_k: 返回结果数量
            filters: 其他过滤条件（可选）
            
        Returns:
            相关的记忆列表
        """
        # 构建过滤条件
        search_filters = filters.copy() if filters else {}
        if user_id:
            search_filters["user_id"] = user_id
        if agent_id:
            search_filters["agent_id"] = agent_id
        
        # 生成查询 embedding
        query_embedding = await self.embedder.aembed(query)
        
        # 搜索（可能获取更多结果用于重排序）
        search_top_k = top_k * 3 if self.reranker else top_k
        results = await self.vector_store.asearch(query_embedding, search_top_k, search_filters)
        
        # 转换为 MemoryItem 列表
        memory_items = []
        for result in results:
            payload = result["payload"]
            memory_items.append(MemoryItem(
                id=result["id"],
                memory=payload.get("memory", ""),
                metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
                score=result["score"],
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at")
            ))
        
        # 如果配置了 Reranker，进行重排序
        if self.reranker and memory_items:
            # 提取文档内容
            documents = [item.memory for item in memory_items]
            
            # 重排序
            rerank_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
            )
            
            # 根据重排序结果重新排序 MemoryItem
            reranked_items = []
            for rerank_result in rerank_results:
                idx = rerank_result["index"]
                item = memory_items[idx]
                # 更新分数为 reranker 的分数
                reranked_item = MemoryItem(
                    id=item.id,
                    memory=item.memory,
                    metadata=item.metadata,
                    score=rerank_result["relevance_score"],
                    created_at=item.created_at,
                    updated_at=item.updated_at
                )
                reranked_items.append(reranked_item)
            
            return reranked_items
        
        return memory_items[:top_k]

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        根据 ID 获取单个记忆（异步）
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            记忆对象，如果不存在返回 None
        """
        result = await self.vector_store.aget(memory_id)
        if not result:
            return None
        
        payload = result["payload"]
        return MemoryItem(
            id=result["id"],
            memory=payload.get("memory", ""),
            metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at")
        )

    async def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        获取所有记忆（可过滤）（异步）
        
        Args:
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            limit: 返回数量限制
            filters: 其他过滤条件（可选）
            
        Returns:
            记忆列表
        """
        # 构建过滤条件
        search_filters = filters.copy() if filters else {}
        if user_id:
            search_filters["user_id"] = user_id
        if agent_id:
            search_filters["agent_id"] = agent_id
        
        # 获取所有结果
        results = await self.vector_store.aget_all(search_filters, limit)
        
        # 转换为 MemoryItem
        memory_items = []
        for result in results:
            payload = result["payload"]
            memory_items.append(MemoryItem(
                id=result["id"],
                memory=payload.get("memory", ""),
                metadata={k: v for k, v in payload.items() if k not in ["memory", "created_at", "updated_at"]},
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at")
            ))
        
        return memory_items

    async def update(
        self,
        memory_id: str,
        new_memory: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        更新记忆（异步）
        
        Args:
            memory_id: 记忆 ID
            new_memory: 新的记忆内容（可选）
            new_metadata: 新的元数据（可选）
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            是否成功
        """
        existing = await self.vector_store.aget(memory_id)
        if not existing:
            return False
        
        payload = existing["payload"].copy()
        old_memory = payload.get("memory")
        now = datetime.utcnow().isoformat()
        payload["updated_at"] = now
        
        # 检查是否有实际变化
        has_change = False
        if new_memory and new_memory != old_memory:
            has_change = True
        if new_metadata:
            for key, value in new_metadata.items():
                if payload.get(key) != value:
                    has_change = True
                    break
        
        if not has_change:
            return True
        
        success = False
        if new_memory:
            payload["memory"] = new_memory
            # 重新生成 embedding
            new_embedding = await self.embedder.aembed(new_memory)
            success = await self.vector_store.aupdate(memory_id, payload, new_embedding)
        elif new_metadata:
            payload.update(new_metadata)
            success = await self.vector_store.aupdate(memory_id, payload)
        
        # 记录历史
        if success:
            self.history_db.add_history(
                memory_id=memory_id,
                old_memory=old_memory,
                new_memory=payload.get("memory"),
                event="update",
                actor_id=actor_id,
                role=role
            )
        
        return success

    async def delete(
        self,
        memory_id: str,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        删除记忆（异步）
        
        Args:
            memory_id: 记忆 ID
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            是否成功
        """
        existing = await self.vector_store.aget(memory_id)
        if not existing:
            return False
        
        old_memory = existing["payload"].get("memory")
        
        # 先记录历史
        self.history_db.add_history(
            memory_id=memory_id,
            old_memory=old_memory,
            new_memory=None,
            event="delete",
            actor_id=actor_id,
            role=role
        )
        
        # 然后删除
        return await self.vector_store.adelete(memory_id)

    async def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        删除所有记忆（可过滤）（异步）
        
        Args:
            user_id: 用户 ID 过滤（可选）
            agent_id: Agent ID 过滤（可选）
            filters: 其他过滤条件（可选）
            actor_id: 操作人 ID（可选，用于历史记录）
            role: 角色（可选，用于历史记录）
            
        Returns:
            是否成功
        """
        # 构建过滤条件
        delete_filters = filters.copy() if filters else {}
        if user_id:
            delete_filters["user_id"] = user_id
        if agent_id:
            delete_filters["agent_id"] = agent_id
        
        # 获取要删除的记忆并记录历史
        memories_to_delete = await self.vector_store.aget_all(delete_filters)
        
        if memories_to_delete:
            history_records = []
            for mem in memories_to_delete:
                payload = mem["payload"]
                history_records.append({
                    "memory_id": mem["id"],
                    "old_memory": payload.get("memory"),
                    "new_memory": None,
                    "event": "delete",
                    "actor_id": actor_id,
                    "role": role
                })
            
            self.history_db.batch_add_history(history_records)
        
        # 删除
        return await self.vector_store.adelete_all(delete_filters if delete_filters else None)

    async def history(self, memory_id: str) -> List[HistoryItem]:
        """
        获取记忆的历史记录（异步）
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            历史记录列表
        """
        history_records = self.history_db.get_history(memory_id)
        return [
            HistoryItem(
                id=record["id"],
                memory_id=record["memory_id"],
                old_memory=record["old_memory"],
                new_memory=record["new_memory"],
                event=record["event"],
                created_at=record["created_at"],
                updated_at=record["updated_at"],
                is_deleted=record["is_deleted"],
                actor_id=record["actor_id"],
                role=record["role"]
            )
            for record in history_records
        ]
