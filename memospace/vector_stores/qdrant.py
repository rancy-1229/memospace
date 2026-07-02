import uuid
from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from memospace.vector_stores.base import VectorStoreBase


class QdrantVectorStore(VectorStoreBase):
    """Qdrant 向量存储实现"""

    def __init__(self, config):
        super().__init__(config)
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key
        )
        self._async_client = None
        self.collection_name = config.collection_name
        self._ensure_collection()

    @property
    def async_client(self):
        """延迟初始化异步客户端"""
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key
            )
        return self._async_client

    def _ensure_collection(self):
        """确保集合存在（同步）"""
        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 1536, "distance": "Cosine"}
            )

    async def _aensure_collection(self):
        """确保集合存在（异步）"""
        collections = (await self.async_client.get_collections()).collections
        if not any(col.name == self.collection_name for col in collections):
            await self.async_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 1536, "distance": "Cosine"}
            )

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """构建 Qdrant 过滤条件"""
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions)

    def add(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        points = [
            PointStruct(id=id_, vector=vector, payload=payload)
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return ids

    async def aadd(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        points = [
            PointStruct(id=id_, vector=vector, payload=payload)
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]
        
        await self.async_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return ids

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter
        )
        return [
            {
                "id": str(result.id),
                "payload": result.payload,
                "score": result.score
            }
            for result in results
        ]

    async def asearch(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter
        )
        return [
            {
                "id": str(result.id),
                "payload": result.payload,
                "score": result.score
            }
            for result in results
        ]

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id]
        )
        if points:
            return {
                "id": str(points[0].id),
                "payload": points[0].payload
            }
        return None

    async def aget(self, id: str) -> Optional[Dict[str, Any]]:
        points = await self.async_client.retrieve(
            collection_name=self.collection_name,
            ids=[id]
        )
        if points:
            return {
                "id": str(points[0].id),
                "payload": points[0].payload
            }
        return None

    def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        # 使用 scroll 获取所有结果
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit or 1000
        )
        return [
            {
                "id": str(record.id),
                "payload": record.payload
            }
            for record in records
        ]

    async def aget_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        # 使用 scroll 获取所有结果
        records, _ = await self.async_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit or 1000
        )
        return [
            {
                "id": str(record.id),
                "payload": record.payload
            }
            for record in records
        ]

    def update(
        self,
        id: str,
        payload: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None
    ) -> bool:
        # 获取现有点
        existing = self.get(id)
        if not existing:
            return False
        
        if vector and payload:
            # 同时更新向量和 payload
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=id, vector=vector, payload=payload)]
            )
        elif vector:
            # 只更新向量
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=id, vector=vector, payload=existing["payload"])]
            )
        elif payload:
            # 只更新 payload
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[id]
            )
        return True

    async def aupdate(
        self,
        id: str,
        payload: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None
    ) -> bool:
        # 获取现有点
        existing = await self.aget(id)
        if not existing:
            return False
        
        if vector and payload:
            # 同时更新向量和 payload
            await self.async_client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=id, vector=vector, payload=payload)]
            )
        elif vector:
            # 只更新向量
            await self.async_client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=id, vector=vector, payload=existing["payload"])]
            )
        elif payload:
            # 只更新 payload
            await self.async_client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[id]
            )
        return True

    def delete(self, id: str) -> bool:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[id]
        )
        return True

    async def adelete(self, id: str) -> bool:
        await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=[id]
        )
        return True

    def delete_all(self, filters: Optional[Dict[str, Any]] = None) -> bool:
        qdrant_filter = self._build_filter(filters)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_filter if qdrant_filter else []
        )
        return True

    async def adelete_all(self, filters: Optional[Dict[str, Any]] = None) -> bool:
        qdrant_filter = self._build_filter(filters)
        await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_filter if qdrant_filter else []
        )
        return True
