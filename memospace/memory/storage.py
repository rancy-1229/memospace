"""
SQLite 历史记录存储管理器
"""

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    """SQLite 历史记录管理器"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 SQLiteManager
        
        Args:
            db_path: 数据库文件路径，如果为 None 则使用默认路径
        """
        if db_path is None:
            import os
            home_dir = os.path.expanduser("~")
            db_path = os.path.join(home_dir, ".memospace", "history.db")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()

    def _migrate_history_table(self) -> None:
        """
        迁移历史表结构（如果需要）
        """
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # 没有表需要迁移

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema.")

                # 清理可能存在的旧表
                cur.execute("DROP TABLE IF EXISTS history_old")

                # 重命名当前表
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # 创建新表
                self._create_history_table_internal(cur)

                # 复制数据
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # 删除旧表
                cur.execute("DROP TABLE history_old")

                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _create_history_table(self) -> None:
        """创建历史表"""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self._create_history_table_internal(self.connection.cursor())
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create history table: {e}")
                raise

    def _create_history_table_internal(self, cur: sqlite3.Cursor) -> None:
        """内部方法：创建历史表"""
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT,
                old_memory   TEXT,
                new_memory   TEXT,
                event        TEXT,
                created_at   DATETIME,
                updated_at   DATETIME,
                is_deleted   INTEGER,
                actor_id     TEXT,
                role         TEXT
            )
            """
        )
        
        # 创建索引以提高查询性能
        cur.execute("CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at)")

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        """
        添加历史记录
        
        Args:
            memory_id: 记忆 ID
            old_memory: 旧记忆内容
            new_memory: 新记忆内容
            event: 事件类型（'add', 'update', 'delete'）
            created_at: 创建时间（可选，默认当前时间）
            updated_at: 更新时间（可选，默认当前时间）
            is_deleted: 是否删除（0 或 1）
            actor_id: 操作人 ID（可选）
            role: 角色（可选）
        """
        now = datetime.now(timezone.utc).isoformat()
        if created_at is None:
            created_at = now
        if updated_at is None:
            updated_at = now

        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def batch_add_history(self, records: List[Dict[str, Any]]) -> None:
        """
        批量添加历史记录
        
        Args:
            records: 历史记录列表，每个记录是一个字典
        """
        now = datetime.now(timezone.utc).isoformat()
        
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                
                # 为每个记录填充默认时间
                processed_records = []
                for record in records:
                    processed_record = record.copy()
                    if processed_record.get("created_at") is None:
                        processed_record["created_at"] = now
                    if processed_record.get("updated_at") is None:
                        processed_record["updated_at"] = now
                    if "is_deleted" not in processed_record:
                        processed_record["is_deleted"] = 0
                    processed_records.append(processed_record)
                
                self.connection.executemany(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(uuid.uuid4()),
                            record.get("memory_id"),
                            record.get("old_memory"),
                            record.get("new_memory"),
                            record.get("event"),
                            record.get("created_at"),
                            record.get("updated_at"),
                            record.get("is_deleted", 0),
                            record.get("actor_id"),
                            record.get("role"),
                        )
                        for record in processed_records
                    ],
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to batch add history records: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        获取指定记忆的历史记录
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            历史记录列表，按创建时间排序
        """
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at ASC
                """,
                (memory_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "is_deleted": bool(r[7]),
                "actor_id": r[8],
                "role": r[9],
            }
            for r in rows
        ]

    def delete_history(self, memory_id: str) -> None:
        """
        删除指定记忆的所有历史记录
        
        Args:
            memory_id: 记忆 ID
        """
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    "DELETE FROM history WHERE memory_id = ?",
                    (memory_id,),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete history records: {e}")
                raise
