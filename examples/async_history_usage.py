"""
Memospace 异步历史记录使用示例
"""

import asyncio
from memospace import AsyncMemory


async def main():
    # 初始化 AsyncMemory 实例
    print("初始化 AsyncMemory 实例...")
    memory = AsyncMemory()

    user_id = "user_async_history_demo"

    # 1. 添加初始记忆
    print("\n--- 1. 添加初始记忆 ---")
    memories = await memory.add(
        "用户说：我热爱人工智能和机器学习，最近在学习深度学习",
        user_id=user_id,
        actor_id="user",
        role="user"
    )
    print(f"添加了 {len(memories)} 条记忆：")
    memory_id = None
    for m in memories:
        if memory_id is None:
            memory_id = m.id  # 保存第一个记忆的 ID
        print(f"  - ID: {m.id}, 内容: {m.memory}")

    # 2. 更新记忆
    print("\n--- 2. 更新记忆 ---")
    if memory_id:
        success = await memory.update(
            memory_id,
            new_memory="用户热爱人工智能和机器学习，最近在学习深度学习和大语言模型",
            actor_id="user",
            role="user"
        )
        print(f"更新成功: {success}")

    # 3. 再次更新
    print("\n--- 3. 再次更新记忆 ---")
    if memory_id:
        success = await memory.update(
            memory_id,
            new_metadata={"category": "AI/ML learning", "priority": "high"},
            actor_id="system",
            role="system"
        )
        print(f"更新成功: {success}")

    # 4. 查看历史记录
    print("\n--- 4. 查看记忆历史记录 ---")
    if memory_id:
        history = await memory.history(memory_id)
        print(f"历史记录数: {len(history)}")
        for i, record in enumerate(history, 1):
            print(f"\n记录 {i}:")
            print(f"  - ID: {record.id}")
            print(f"  - 事件类型: {record.event}")
            print(f"  - 旧内容: {record.old_memory}")
            print(f"  - 新内容: {record.new_memory}")
            print(f"  - 操作人: {record.actor_id}")
            print(f"  - 角色: {record.role}")
            print(f"  - 创建时间: {record.created_at}")
            print(f"  - 更新时间: {record.updated_at}")
            print(f"  - 是否已删除: {record.is_deleted}")

    # 5. 删除记忆
    print("\n--- 5. 删除记忆 ---")
    if memory_id:
        success = await memory.delete(memory_id, actor_id="user", role="user")
        print(f"删除成功: {success}")

    # 6. 查看删除后的历史记录
    print("\n--- 6. 查看删除后的历史记录 ---")
    if memory_id:
        history = await memory.history(memory_id)
        print(f"历史记录数: {len(history)}")
        for i, record in enumerate(history, 1):
            print(f"\n记录 {i}:")
            print(f"  - 事件类型: {record.event}")
            print(f"  - 旧内容: {record.old_memory}")
            print(f"  - 新内容: {record.new_memory}")
            print(f"  - 是否已删除: {record.is_deleted}")

    print("\n完成！")


if __name__ == "__main__":
    asyncio.run(main())
