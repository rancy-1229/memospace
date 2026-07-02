"""
Memospace Reranker 使用示例
"""

from memospace import Memory
from memospace.configs import RerankerConfig


def main():
    # 示例 1：使用默认配置（无 Reranker）
    print("=" * 60)
    print("示例 1：不使用 Reranker")
    print("=" * 60)
    
    memory = Memory()
    user_id = "reranker_demo_user"
    
    # 添加一些示例记忆
    memory.add("用户喜欢阅读科幻小说，特别是太空主题", user_id=user_id)
    memory.add("用户对咖啡有浓厚兴趣，喜欢尝试不同的咖啡豆", user_id=user_id)
    memory.add("用户热爱编程，经常用 Python 写代码", user_id=user_id)
    memory.add("用户喜欢爬山，周末经常去郊外徒步", user_id=user_id)
    memory.add("用户最近在学习机器学习和深度学习", user_id=user_id)
    
    # 搜索（无 Reranker）
    query = "用户喜欢什么技术相关的东西？"
    print(f"\n搜索查询: {query}")
    results = memory.search(query, user_id=user_id, top_k=3)
    
    print("\n搜索结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [Score: {result.score:.4f}] {result.memory}")
    
    # 清空记忆以便后面使用
    memory.delete_all(user_id=user_id)
    
    # 示例 2：使用 Cohere Reranker
    print("\n" + "=" * 60)
    print("示例 2：使用 Cohere Reranker")
    print("=" * 60)
    
    # 配置 Cohere Reranker
    cohere_config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        top_n=3
    )
    
    # 初始化带有 Reranker 的 Memory
    memory_with_cohere = Memory(
        reranker_config=cohere_config
    )
    
    # 重新添加记忆
    memory_with_cohere.add("用户喜欢阅读科幻小说，特别是太空主题", user_id=user_id)
    memory_with_cohere.add("用户对咖啡有浓厚兴趣，喜欢尝试不同的咖啡豆", user_id=user_id)
    memory_with_cohere.add("用户热爱编程，经常用 Python 写代码", user_id=user_id)
    memory_with_cohere.add("用户喜欢爬山，周末经常去郊外徒步", user_id=user_id)
    memory_with_cohere.add("用户最近在学习机器学习和深度学习", user_id=user_id)
    
    # 搜索（使用 Cohere Reranker）
    print(f"\n搜索查询: {query}")
    results_with_cohere = memory_with_cohere.search(query, user_id=user_id, top_k=3)
    
    print("\n重排序后的搜索结果:")
    for i, result in enumerate(results_with_cohere, 1):
        print(f"  {i}. [Score: {result.score:.4f}] {result.memory}")
    
    # 清空记忆
    memory_with_cohere.delete_all(user_id=user_id)
    
    # 示例 3：使用 LLM-based Reranker
    print("\n" + "=" * 60)
    print("示例 3：使用 LLM-based Reranker")
    print("=" * 60)
    
    llm_config = RerankerConfig(
        provider="llm",
        top_n=3
    )
    
    memory_with_llm_rerank = Memory(
        reranker_config=llm_config
    )
    
    # 重新添加记忆
    memory_with_llm_rerank.add("用户喜欢阅读科幻小说，特别是太空主题", user_id=user_id)
    memory_with_llm_rerank.add("用户对咖啡有浓厚兴趣，喜欢尝试不同的咖啡豆", user_id=user_id)
    memory_with_llm_rerank.add("用户热爱编程，经常用 Python 写代码", user_id=user_id)
    memory_with_llm_rerank.add("用户喜欢爬山，周末经常去郊外徒步", user_id=user_id)
    memory_with_llm_rerank.add("用户最近在学习机器学习和深度学习", user_id=user_id)
    
    # 搜索
    print(f"\n搜索查询: {query}")
    results_with_llm = memory_with_llm_rerank.search(query, user_id=user_id, top_k=3)
    
    print("\nLLM 重排序后的搜索结果:")
    for i, result in enumerate(results_with_llm, 1):
        print(f"  {i}. [Score: {result.score:.4f}] {result.memory}")
    
    # 清空记忆
    memory_with_llm_rerank.delete_all(user_id=user_id)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
