"""
Memospace 历史记录使用示例
"""

from memospace import Memory
from memospace.configs import HistoryItem


def main():
    # 初始化 Memory 实例
    print("初始化 Memory 实例...")
    memory = Memory()

    user_id = "user_history_demo"

    # 1. 添加初始记忆
    print("\n--- 1. 添加初始记忆 ---")
    memories = memory.add(
        "用户说：我喜欢编程和阅读科幻小说，我最喜欢的作家是阿西莫夫",
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
        success = memory.update(
            memory_id,
            new_memory="用户喜欢编程和阅读科幻小说，最喜欢的作家是阿西莫夫和刘慈欣",
            actor_id="user",
            role="user"
        )
        print(f"更新成功: {success}")

    # 3. 再次更新
    print("\n--- 3. 再次更新记忆 ---")
    if memory_id:
        success = memory.update(
            memory_id,
            new_metadata={"topic": "reading preferences"},
            actor_id="system",
            role="system"
        )
        print(f"更新成功: {success}")

    # 4. 查看历史记录
    print("\n--- 4. 查看记忆历史记录 ---")
    if memory_id:
        history = memory.history(memory_id)
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
        success = memory.delete(memory_id, actor_id="user", role="user")
        print(f"删除成功: {success}")

    # 6. 查看删除后的历史记录
    print("\n--- 6. 查看删除后的历史记录 ---")
    if memory_id:
        history = memory.history(memory_id)
        print(f"历史记录数: {len(history)}")
        for i, record in enumerate(history, 1):
            print(f"\n记录 {i}:")
            print(f"  - 事件类型: {record.event}")
            print(f"  - 旧内容: {record.old_memory}")
            print(f"  - 新内容: {record.new_memory}")
            print(f"  - 是否已删除: {record.is_deleted}")

    print("\n完成！")


if __name__ == "__main__":
    main()
