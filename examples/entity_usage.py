"""
实体提取和关联功能使用示例

演示如何使用 memospace 的实体提取、存储和搜索增强功能
"""

from memospace import Memory
from memospace.configs.base import MemoryConfig


def main():
    print("=" * 60)
    print("实体提取和关联功能示例")
    print("=" * 60)
    
    # 初始化 Memory（默认启用实体功能）
    config = MemoryConfig(
        enable_entities=True,  # 启用实体功能
        entity_boost_factor=1.5  # 实体匹配时的分数提升因子
    )
    memory = Memory(config=config)
    
    # 1. 添加包含实体的记忆
    print("\n1. 添加包含实体的记忆...")
    
    memories_to_add = [
        "John Smith works at Google in Mountain View. He loves Python programming.",
        "Sarah Johnson is a data scientist at Amazon. She uses machine learning daily.",
        "The Eiffel Tower is in Paris, France. It was built for the 1889 World's Fair.",
        "Tokyo is the capital of Japan. It has a population of over 13 million people.",
        "Steve Jobs co-founded Apple in 1976. The iPhone was released in 2007.",
    ]
    
    for idx, text in enumerate(memories_to_add, 1):
        print(f"   添加记忆 {idx}: {text[:60]}...")
        memory.add(text, user_id="user_001")
    
    print("   ✓ 记忆添加完成")
    
    # 2. 获取所有实体
    print("\n2. 获取所有提取的实体...")
    all_entities = memory.get_all_entities()
    print(f"   共找到 {len(all_entities)} 个实体:")
    for ent in all_entities:
        print(f"   - [{ent.entity_type}] {ent.text} (关联 {ent.memory_count} 个记忆)")
    
    # 3. 搜索实体
    print("\n3. 搜索特定实体...")
    search_terms = ["Google", "Paris", "Apple"]
    for term in search_terms:
        entities = memory.search_entities(term)
        if entities:
            print(f"   搜索 '{term}':")
            for ent in entities:
                print(f"   - {ent.text} ({ent.entity_type})")
    
    # 4. 通过实体查找记忆
    print("\n4. 通过实体查找关联记忆...")
    target_entities = ["John Smith", "Eiffel Tower", "Apple"]
    for entity_text in target_entities:
        related_memories = memory.get_memories_by_entity(entity_text)
        print(f"   与 '{entity_text}' 相关的记忆:")
        for idx, mem in enumerate(related_memories, 1):
            print(f"   {idx}. {mem.memory[:80]}...")
    
    # 5. 实体增强搜索
    print("\n5. 实体增强搜索演示...")
    
    # 普通搜索
    print("   普通搜索 'Google':")
    normal_results = memory.search("Google", enable_entity_boost=False)
    for idx, result in enumerate(normal_results, 1):
        print(f"   {idx}. [分数: {result.score:.4f}] {result.memory[:60]}...")
    
    # 实体增强搜索
    print("\n   实体增强搜索 'Google':")
    boosted_results = memory.search("Google", enable_entity_boost=True)
    for idx, result in enumerate(boosted_results, 1):
        print(f"   {idx}. [分数: {result.score:.4f}] {result.memory[:60]}...")
    
    # 6. 获取特定记忆的实体
    print("\n6. 获取特定记忆关联的实体...")
    # 先获取一个记忆
    all_memories = memory.get_all(user_id="user_001", limit=1)
    if all_memories:
        sample_memory = all_memories[0]
        print(f"   记忆内容: {sample_memory.memory[:80]}...")
        entities = memory.get_entities(sample_memory.id)
        print(f"   关联实体:")
        for ent in entities:
            print(f"   - [{ent.entity_type}] {ent.text}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
