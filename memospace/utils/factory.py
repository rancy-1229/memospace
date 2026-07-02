from typing import Dict, Type, Any

from memospace.llms.base import LLMBase
from memospace.llms.openai import OpenAILLM
from memospace.embeddings.base import EmbeddingBase
from memospace.embeddings.openai import OpenAIEmbedding
from memospace.vector_stores.base import VectorStoreBase
from memospace.vector_stores.qdrant import QdrantVectorStore
from memospace.rerankers.base import RerankerBase
from memospace.rerankers.cohere_reranker import CohereReranker
from memospace.rerankers.huggingface_reranker import HuggingFaceReranker
from memospace.rerankers.llm_reranker import LLMReranker


class LlmFactory:
    """LLM 工厂类"""
    
    _providers: Dict[str, Type[LLMBase]] = {
        "openai": OpenAILLM
    }
    
    @classmethod
    def create(cls, config) -> LLMBase:
        """
        创建 LLM 实例
        
        Args:
            config: LLM 配置对象
            
        Returns:
            LLM 实例
        """
        provider_name = config.provider.lower()
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        return cls._providers[provider_name](config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMBase]):
        """注册新的 LLM Provider"""
        cls._providers[name.lower()] = provider_class


class EmbedderFactory:
    """Embedding 工厂类"""
    
    _providers: Dict[str, Type[EmbeddingBase]] = {
        "openai": OpenAIEmbedding
    }
    
    @classmethod
    def create(cls, config) -> EmbeddingBase:
        """
        创建 Embedding 实例
        
        Args:
            config: Embedding 配置对象
            
        Returns:
            Embedding 实例
        """
        provider_name = config.provider.lower()
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
        return cls._providers[provider_name](config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[EmbeddingBase]):
        """注册新的 Embedding Provider"""
        cls._providers[name.lower()] = provider_class


class VectorStoreFactory:
    """Vector Store 工厂类"""
    
    _providers: Dict[str, Type[VectorStoreBase]] = {
        "qdrant": QdrantVectorStore
    }
    
    @classmethod
    def create(cls, config) -> VectorStoreBase:
        """
        创建 Vector Store 实例
        
        Args:
            config: Vector Store 配置对象
            
        Returns:
            Vector Store 实例
        """
        provider_name = config.provider.lower()
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported vector store provider: {provider_name}")
        return cls._providers[provider_name](config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[VectorStoreBase]):
        """注册新的 Vector Store Provider"""
        cls._providers[name.lower()] = provider_class


class RerankerFactory:
    """Reranker 工厂类"""
    
    _providers: Dict[str, Type[RerankerBase]] = {
        "cohere": CohereReranker,
        "huggingface": HuggingFaceReranker,
    }
    
    @classmethod
    def create(cls, config, llm=None) -> RerankerBase:
        """
        创建 Reranker 实例
        
        Args:
            config: Reranker 配置对象
            llm: LLM 实例（仅用于 LLMReranker）
            
        Returns:
            Reranker 实例
        """
        provider_name = config.provider.lower()
        
        if provider_name == "llm":
            if llm is None:
                raise ValueError("LLM is required for LLMReranker")
            return LLMReranker(config, llm)
        
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported reranker provider: {provider_name}")
        
        return cls._providers[provider_name](config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[RerankerBase]):
        """注册新的 Reranker Provider"""
        cls._providers[name.lower()] = provider_class
