"""
Memospace 异步使用示例
"""

import os
import asyncio
from memospace import AsyncMemory


async def main():
    # 确保设置了 OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置 OPENAI_API_KEY 环境变量")
        return

    # 1. 初始化 AsyncMemory 实例（使用默认配置）
    print("初始化 AsyncMemory...")
    memory = AsyncMemory()

    user_id = "user_001"

    # 2. 添加记忆
    print("\n添加记忆...")
    memories = await memory.add(
        "用户说：我叫李明，今年 30 岁，是一名软件工程师，喜欢编程和阅读科幻小说。",
        user_id=user_id
    )
    print(f"添加了 {len(memories)} 条记忆：")
    for m in memories:
        print(f"  - {m.memory}")

    # 添加更多记忆
    print("\n添加更多记忆...")
    memories = await memory.add(
        "用户说：我昨天去了咖啡馆，喝了一杯拿铁，感觉很不错。我每周都会去那里两三次。",
        user_id=user_id
    )
    print(f"添加了 {len(memories)} 条记忆")

    # 3. 搜索记忆
    print("\n搜索相关记忆...")
    query = "用户喜欢做什么？"
    results = await memory.search(query, user_id=user_id, top_k=3)
    print(f"查询: '{query}'")
    print(f"找到 {len(results)} 条相关记忆：")
    for r in results:
        print(f"  - [{r.score:.4f}] {r.memory}")

    # 4. 获取所有记忆
    print("\n获取用户的所有记忆...")
    all_memories = await memory.get_all(user_id=user_id)
    print(f"共有 {len(all_memories)} 条记忆：")
    for m in all_memories:
        print(f"  - {m.memory} (ID: {m.id})")

    if all_memories:
        # 5. 获取单个记忆
        print("\n获取单个记忆...")
        single_memory = await memory.get(all_memories[0].id)
        print(f"记忆内容: {single_memory.memory}")

        # 6. 更新记忆
        print("\n更新记忆...")
        success = await memory.update(
            single_memory.id,
            new_memory="李明是一名资深软件工程师，喜欢编程和阅读科幻小说"
        )
        print(f"更新成功: {success}")

        # 验证更新
        updated = await memory.get(single_memory.id)
        print(f"更新后的内容: {updated.memory}")

        # 7. 删除记忆
        print("\n删除记忆...")
        delete_success = await memory.delete(single_memory.id)
        print(f"删除成功: {delete_success}")

    # 8. 清空所有记忆（可选，演示用）
    # print("\n清空所有记忆...")
    # await memory.delete_all(user_id=user_id)
    # print("已清空")


if __name__ == "__main__":
    asyncio.run(main())
