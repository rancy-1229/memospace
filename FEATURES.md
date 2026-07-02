# Memospace 功能清单与路线图

本文档记录 Mem0 中包含但 Memospace 目前缺少的功能，以及实现优先级。

## 缺少的核心功能

### 1. 异步支持 (AsyncMemory) ✅
- [x] `AsyncMemory` 类
- [x] 异步 Provider 接口
- [x] 异步 OpenAI LLM 实现
- [x] 异步 OpenAI Embedding 实现
- [x] 异步 Qdrant Vector Store 实现
- [x] 异步使用示例

### 2. 历史记录管理 (History) ✅
- [x] SQLiteManager 实现
- [x] 记忆变更历史追踪
- [x] `history(memory_id)` 方法
- [x] 事件类型、时间戳、操作人记录
- [x] 历史表迁移逻辑

### 3. Reranker 支持 ✅
- [x] Reranker 抽象基类
- [x] Reranker 配置类
- [x] Cohere Reranker 实现
- [x] HuggingFace Reranker 实现
- [x] LLM-based Reranker 实现
- [x] 集成到 Memory 搜索流程

### 4. 实体提取和链接 (Entity Extraction & Linking) ✅
- [x] 实体提取模块 (spaCy 集成 + 正则回退)
- [x] Entity Store 单独存储 (SQLite)
- [x] 实体与记忆关联
- [x] 搜索时实体增强
- [x] 实体相关 API (get_entities, search_entities, get_all_entities, get_memories_by_entity)
- [x] 使用示例 (examples/entity_usage.py)

### 5. 混合搜索 (Hybrid Search) ✅
- [x] BM25 关键词搜索
- [x] Lemmatization 模块（带可选 NLTK 依赖）
- [x] 混合评分和排序（支持 LINEAR 和 RRF 策略）
- [x] hybrid_search() 方法集成到 Memory 类
- [x] bm25_search() 方法单独调用 BM25
- [x] 配置项：enable_hybrid_search, hybrid_strategy, weights 等
- [x] 使用示例 (examples/hybrid_search_usage.py)

### 6. 时间感知和衰减 (Temporal Reasoning & Decay)
- [ ] 记忆过期 (expiration_date)
- [ ] 时间衰减逻辑
- [ ] 时间感知搜索排序

### 7. 云端客户端 (Cloud/Platform Client)
- [ ] MemoryClient 类
- [ ] AsyncMemoryClient 类
- [ ] API 调用封装

### 8. 更多 Provider 支持
**LLMs (20+):**
- [ ] Anthropic
- [ ] Gemini
- [ ] Groq
- [ ] Ollama
- [ ] Together
- [ ] vLLM
- [ ] LM Studio

**Vector Stores (30+):**
- [ ] Pinecone
- [ ] Chroma
- [ ] Weaviate
- [ ] Milvus
- [ ] MongoDB
- [ ] Redis
- [ ] pgvector
- [ ] FAISS

**Embeddings (15+):**
- [ ] Azure OpenAI
- [ ] Gemini
- [ ] FastEmbed
- [ ] Together
- [ ] Vertex AI

### 9. 高级配置和验证
- [ ] 完善的输入验证
- [ ] Entity ID 格式验证
- [ ] 搜索参数验证
- [ ] 敏感信息脱敏
- [ ] custom_instructions 支持

### 10. Telemetry 和 Notices
- [ ] 使用统计
- [ ] 性能监控
- [ ] 友好提示和升级建议

### 11. 异常处理体系
- [ ] 自定义异常类
- [ ] 详细错误信息
- [ ] 修复建议

### 12. 项目管理 (Project)
- [ ] 项目配置
- [ ] 自定义分类
- [ ] 检索标准

## 实现优先级

### 第一优先级 (核心功能增强)
1. 异步支持 (AsyncMemory)
2. 历史记录管理
3. 输入验证和异常处理

### 第二优先级 (质量提升)
1. Reranker
2. 混合搜索
3. 更多 Provider (Ollama 等)

### 第三优先级 (高级特性)
1. 实体提取
2. 时间感知和衰减
3. 云端客户端

## 当前进度
- [x] 项目初始化
- [x] 基础配置类
- [x] Memory 核心类
- [x] OpenAI LLM (同步)
- [x] OpenAI Embedding (同步)
- [x] Qdrant Vector Store (同步)
- [x] 异步支持
- [x] 历史记录管理
