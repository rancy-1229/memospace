from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class VectorStoreBase(ABC):
    """向量存储抽象基类"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def add(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        添加向量（同步）
        
        Args:
            vectors: 向量列表
            payloads: 对应的 payload 列表
            ids: 可选的 ID 列表
            
        Returns:
            添加的 ID 列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量（同步）
        
        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取单个向量（同步）
        
        Args:
            id: 向量 ID
            
        Returns:
            向量和 payload，如果不存在返回 None
        """
        pass

    @abstractmethod
    def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取所有向量（同步，可过滤）
        
        Args:
            filters: 过滤条件
            limit: 返回数量限制
            
        Returns:
            向量列表
        """
        pass

    @abstractmethod
    def update(
        self,
        id: str,
        payload: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None
    ) -> bool:
        """
        更新向量（同步）
        
        Args:
            id: 向量 ID
            payload: 新的 payload（可选）
            vector: 新的向量（可选）
            
        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        删除向量（同步）
        
        Args:
            id: 向量 ID
            
        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def delete_all(self, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        删除所有向量（同步，可过滤）
        
        Args:
            filters: 过滤条件
            
        Returns:
            是否成功
        """
        pass

    async def aadd(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        添加向量（异步）
        
        Args:
            vectors: 向量列表
            payloads: 对应的 payload 列表
            ids: 可选的 ID 列表
            
        Returns:
            添加的 ID 列表
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.add(vectors, payloads, ids)
        )

    async def asearch(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量（异步）
        
        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.search(query_vector, limit, filters)
        )

    async def aget(self, id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取单个向量（异步）
        
        Args:
            id: 向量 ID
            
        Returns:
            向量和 payload，如果不存在返回 None
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.get(id))

    async def aget_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取所有向量（异步，可过滤）
        
        Args:
            filters: 过滤条件
            limit: 返回数量限制
            
        Returns:
            向量列表
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.get_all(filters, limit)
        )

    async def aupdate(
        self,
        id: str,
        payload: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None
    ) -> bool:
        """
        更新向量（异步）
        
        Args:
            id: 向量 ID
            payload: 新的 payload（可选）
            vector: 新的向量（可选）
            
        Returns:
            是否成功
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.update(id, payload, vector)
        )

    async def adelete(self, id: str) -> bool:
        """
        删除向量（异步）
        
        Args:
            id: 向量 ID
            
        Returns:
            是否成功
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.delete(id))

    async def adelete_all(self, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        删除所有向量（异步，可过滤）
        
        Args:
            filters: 过滤条件
            
        Returns:
            是否成功
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.delete_all(filters)
        )
