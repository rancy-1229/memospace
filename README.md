# Memospace

一个简单但功能完整的 AI Agent 记忆模块 SDK。

## 功能特性

- **多 Provider 支持**：目前支持 OpenAI (LLM 和 Embeddings) 和 Qdrant (向量存储)
- **灵活的配置**：通过配置对象轻松自定义各个组件
- **智能记忆提取**：使用 LLM 自动从文本中提取关键事实
- **语义搜索**：基于向量相似度快速检索相关记忆
- **元数据过滤**：支持按 user_id、agent_id 等维度过滤记忆
- **完整的 CRUD**：添加、搜索、获取、更新、删除记忆
- **异步支持**：提供 AsyncMemory 类用于异步操作
- **历史记录管理**：记录记忆变更历史，支持时间回溯
- **Reranker 支持**：支持 Cohere、HuggingFace、LLM-based reranker 提升搜索质量
- **实体提取和链接**：自动提取实体，关联记忆，搜索时实体增强
- **混合搜索**：结合 BM25 关键词搜索和向量相似度搜索，提供更优结果

## 安装

```bash
# 克隆项目
cd memospace

# 使用 pip 安装（开发模式）
pip install -e .

# 可选依赖（增强功能）
pip install -e ".[nltk]"  # 用于更好的词形还原
```

## 前置要求

1. **OpenAI API Key**：需要设置 `OPENAI_API_KEY` 环境变量
2. **Qdrant 实例**：需要运行 Qdrant（可以使用 Docker）

### 运行 Qdrant（Docker）

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## 快速开始

### 基础使用

```python
from memospace import Memory

# 初始化（使用默认配置）
memory = Memory()

# 添加记忆
user_id = "user_123"
memory.add("用户说：我叫张三，喜欢编程和阅读", user_id=user_id)

# 搜索记忆
results = memory.search("用户喜欢什么？", user_id=user_id, top_k=3)
for r in results:
    print(f"{r.score:.4f} - {r.memory}")

# 获取所有记忆
all_memories = memory.get_all(user_id=user_id)
```

### 混合搜索

```python
from memospace import Memory

memory = Memory()

# 使用 hybrid_search 获得更好的搜索结果
# 结合 BM25 关键词匹配和向量相似度
results = memory.hybrid_search("人工智能", top_k=5)
```

### 异步使用

```python
from memospace import AsyncMemory
import asyncio

async def main():
    memory = AsyncMemory()
    await memory.add("我喜欢 Python")
    results = await memory.search("喜欢什么")
    print(results)

asyncio.run(main())
```

## 自定义配置

```python
from memospace import Memory
from memospace.configs import (
    MemoryConfig, LlmConfig, EmbedderConfig, VectorStoreConfig
)
from memospace.search.hybrid_search import HybridStrategy

# 创建自定义配置
config = MemoryConfig(
    llm=LlmConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.5
    ),
    embedder=EmbedderConfig(
        provider="openai",
        model="text-embedding-3-small"
    ),
    vector_store=VectorStoreConfig(
        provider="qdrant",
        host="localhost",
        port=6333,
        collection_name="my_memories"
    ),
    # 启用混合搜索
    enable_hybrid_search=True,
    hybrid_strategy=HybridStrategy.LINEAR,
    bm25_weight=0.4,
    vector_weight=0.6,
    # 启用实体功能
    enable_entities=True,
    entity_boost_factor=1.5
)

# 或者直接通过 kwargs 配置
memory = Memory(
    llm_config={"model": "gpt-4o"},
    embedding_config={"model": "text-embedding-3-large"},
    vector_store_config={"collection_name": "my_memories"}
)
```

## 高级功能

### 实体提取和链接

```python
from memospace import Memory

memory = Memory()

# 添加记忆时会自动提取实体
memory.add("Apple 发布了新的 iPhone", user_id="user_1")

# 通过实体查找记忆
related_memories = memory.get_memories_by_entity("Apple")

# 搜索时自动应用实体提升
results = memory.hybrid_search("智能手机", enable_entity_boost=True)
```

### 历史记录管理

```python
from memospace import Memory

memory = Memory()

# 添加和更新记忆
memory.add("我喜欢红色", user_id="user_1")
memory_items = memory.get_all(user_id="user_1")

if memory_items:
    memory.update(memory_items[0].id, new_memory="我喜欢蓝色")
    
    # 查看历史记录
    history = memory.history(memory_items[0].id)
    for record in history:
        print(f"{record.event}: {record.old_memory} -> {record.new_memory}")
```

## API 文档

### Memory 类

主要方法：

- `add(text, user_id=None, agent_id=None, metadata=None)`：添加记忆
- `search(query, user_id=None, agent_id=None, top_k=10, filters=None)`：搜索记忆
- `hybrid_search(query, user_id=None, agent_id=None, top_k=10, filters=None)`：混合搜索
- `bm25_search(query, top_k=10)`：仅 BM25 搜索
- `get(memory_id)`：获取单个记忆
- `get_all(user_id=None, agent_id=None, limit=None, filters=None)`：获取所有记忆
- `update(memory_id, new_memory=None, new_metadata=None)`：更新记忆
- `delete(memory_id)`：删除记忆
- `delete_all(user_id=None, agent_id=None, filters=None)`：删除所有记忆
- `history(memory_id)`：获取历史记录
- `get_entities(memory_id)`：获取关联实体
- `search_entities(query)`：搜索实体
- `get_all_entities()`：获取所有实体
- `get_memories_by_entity(entity_text)`：通过实体查找记忆

### AsyncMemory 类

提供与 Memory 类相同的方法，但都是异步版本。

### MemoryItem 类

记忆数据模型，包含：

- `id`：记忆 ID
- `memory`：记忆内容
- `metadata`：元数据
- `score`：相关性得分（搜索时返回）
- `created_at` / `updated_at`：时间戳

## 使用示例

项目提供了多个使用示例，位于 `examples/` 目录下：

- `basic_usage.py`：基础使用
- `async_usage.py`：异步使用
- `history_usage.py`：历史记录管理
- `reranker_usage.py`：Reranker 使用
- `entity_usage.py`：实体提取和链接
- `hybrid_search_usage.py`：混合搜索

## 项目结构

```
memospace/
├── memospace/
│   ├── __init__.py
│   ├── configs/          # 配置类
│   ├── llms/            # LLM 实现
│   ├── embeddings/      # Embedding 实现
│   ├── vector_stores/   # 向量存储实现
│   ├── rerankers/       # Reranker 实现
│   ├── search/          # 搜索模块（BM25 + 混合搜索）
│   ├── text_processing/ # 文本处理（词形还原）
│   ├── memory/          # 核心记忆模块
│   └── utils/           # 工具类
├── examples/            # 使用示例
├── tests/               # 测试
├── pyproject.toml
├── FEATURES.md          # 功能路线图
└── README.md
```

## 功能路线图

详见 [FEATURES.md](./FEATURES.md) 查看已完成和计划中的功能。

## 扩展新的 Provider

### 添加新的 LLM Provider

1. 继承 `LLMBase` 实现你的 Provider 类
2. 使用 `LlmFactory.register_provider()` 注册

```python
from memospace.llms.base import LLMBase
from memospace.utils.factory import LlmFactory

class MyLLM(LLMBase):
    def generate_response(self, messages, **kwargs):
        # 实现你的逻辑
        pass

# 注册
LlmFactory.register_provider("my_llm", MyLLM)
```

Embedding 和 Vector Store 类似。

## 许可证

MIT
