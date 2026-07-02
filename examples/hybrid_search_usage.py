"""
混合搜索功能使用示例
演示如何使用 BM25 关键词搜索 + 向量相似度搜索的混合搜索
"""

from memospace import Memory
from memospace.configs.base import MemoryConfig
from memospace.search.hybrid_search import HybridStrategy


def main():
    print("=" * 70)
    print("混合搜索功能使用示例")
    print("=" * 70)
    
    # 1. 初始化 Memory，启用混合搜索
    print("\n1. 初始化 Memory，启用混合搜索...")
    config = MemoryConfig(
        enable_hybrid_search=True,
        hybrid_strategy=HybridStrategy.LINEAR,
        bm25_weight=0.4,
        vector_weight=0.6
    )
    memory = Memory(config=config)
    
    # 2. 添加一些示例记忆
    print("\n2. 添加示例记忆...")
    sample_memories = [
        "Apple announces new iPhone with advanced camera features",
        "Microsoft unveils latest Windows operating system update",
        "Google introduces new AI-powered search capabilities",
        "Tesla releases autonomous driving software version 12.0",
        "Amazon launches new AWS cloud computing services",
        "SpaceX successfully lands Starship prototype",
        "Meta unveils new virtual reality headset for metaverse",
        "NVIDIA announces groundbreaking GPU for AI training",
        "Samsung reveals foldable smartphone with improved display",
        "Intel launches next-generation processor for data centers"
    ]
    
    for text in sample_memories:
        memory.add(text, user_id="demo_user")
    print(f"   已添加 {len(sample_memories)} 条示例记忆")
    
    # 3. 搜索示例
    print("\n3. 演示搜索功能...")
    test_queries = [
        "AI technology companies",
        "new smartphone release",
        "software update announcement",
        "electric vehicle technology"
    ]
    
    for query in test_queries:
        print(f"\n   查询: \"{query}\"")
        
        # 普通向量搜索
        vector_results = memory.search(query, top_k=3, use_hybrid=False)
        print(f"   向量搜索结果:")
        for i, item in enumerate(vector_results, 1):
            print(f"     {i}. {item.memory}")
        
        # 混合搜索
        hybrid_results = memory.hybrid_search(query, top_k=3, use_hybrid=True)
        print(f"   混合搜索结果:")
        for i, item in enumerate(hybrid_results, 1):
            print(f"     {i}. {item.memory}")
    
    # 4. BM25 直接搜索
    print("\n4. BM25 直接搜索（关键词匹配）...")
    bm25_query = "Apple iPhone camera"
    bm25_results = memory.bm25_search(bm25_query, top_k=3)
    print(f"   查询: \"{bm25_query}\"")
    for i, result in enumerate(bm25_results, 1):
        print(f"   {i}. [BM25 分数: {result.score:.4f}] {result.doc_text}")
    
    # 5. 不同混合策略对比
    print("\n5. 不同混合策略对比...")
    strategies = [
        ("线性加权", HybridStrategy.LINEAR),
        ("RRF 排序融合", HybridStrategy.RRF)
    ]
    
    test_query = "new product launches"
    
    for strategy_name, strategy in strategies:
        print(f"\n   策略: {strategy_name}")
        
        # 更新配置
        config_temp = MemoryConfig(
            enable_hybrid_search=True,
            hybrid_strategy=strategy,
            bm25_weight=0.5,
            vector_weight=0.5
        )
        memory_temp = Memory(config=config_temp)
        
        # 先添加相同的记忆
        for text in sample_memories:
            memory_temp.add(text, user_id="temp_user")
        
        # 搜索
        results = memory_temp.hybrid_search(test_query, top_k=3)
        print(f"   结果:")
        for i, item in enumerate(results, 1):
            print(f"     {i}. {item.memory}")
    
    print("\n" + "=" * 70)
    print("混合搜索功能演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
