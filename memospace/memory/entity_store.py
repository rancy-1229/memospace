"""
Entity Store - SQLite-based storage for entities and their associations with memories.
"""

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memospace.entity_extraction import Entity

logger = logging.getLogger(__name__)


class EntityStore:
    """SQLite-based entity storage manager"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize EntityStore
        
        Args:
            db_path: Database file path, uses default if None
        """
        if db_path is None:
            import os
            home_dir = os.path.expanduser("~")
            db_path = os.path.join(home_dir, ".memospace", "entities.db")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        """Create entity tables"""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                
                # Entities table - stores unique entities
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entities (
                        id           TEXT PRIMARY KEY,
                        text         TEXT NOT NULL,
                        entity_type  TEXT NOT NULL,
                        ner_label    TEXT,
                        canonical    TEXT NOT NULL,  -- canonical form (lowercase)
                        created_at   DATETIME NOT NULL,
                        updated_at   DATETIME NOT NULL,
                        UNIQUE(canonical, entity_type)
                    )
                    """
                )
                
                # Memory-Entity association table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_entities (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT NOT NULL,
                        entity_id    TEXT NOT NULL,
                        confidence   REAL NOT NULL,
                        created_at   DATETIME NOT NULL,
                        FOREIGN KEY(entity_id) REFERENCES entities(id)
                    )
                    """
                )
                
                # Create indexes for performance
                cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_entities_memory_id ON memory_entities(memory_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_id ON memory_entities(entity_id)")
                
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create entity tables: {e}")
                raise

    def add_entity(
        self,
        entity: Entity,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        Add an entity and optionally associate it with a memory
        
        Args:
            entity: Entity object to add
            memory_id: Optional memory ID to associate with
            user_id: Optional user ID for scoping
            agent_id: Optional agent ID for scoping
            
        Returns:
            Entity ID
        """
        now = datetime.now(timezone.utc).isoformat()
        canonical = entity.text.lower()
        
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                
                # Check if entity already exists
                cur.execute(
                    "SELECT id FROM entities WHERE canonical = ? AND entity_type = ?",
                    (canonical, entity.entity_type)
                )
                row = cur.fetchone()
                
                if row:
                    entity_id = row[0]
                    # Update updated_at
                    cur.execute(
                        "UPDATE entities SET updated_at = ? WHERE id = ?",
                        (now, entity_id)
                    )
                else:
                    # Create new entity
                    entity_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO entities (
                            id, text, entity_type, ner_label, canonical, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entity_id,
                            entity.text,
                            entity.entity_type,
                            entity.ner_label,
                            canonical,
                            now,
                            now,
                        ),
                    )
                
                # Associate with memory if provided
                if memory_id:
                    # Check if association already exists
                    cur.execute(
                        "SELECT id FROM memory_entities WHERE memory_id = ? AND entity_id = ?",
                        (memory_id, entity_id)
                    )
                    if not cur.fetchone():
                        assoc_id = str(uuid.uuid4())
                        cur.execute(
                            """
                            INSERT INTO memory_entities (
                                id, memory_id, entity_id, confidence, created_at
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (assoc_id, memory_id, entity_id, entity.confidence, now),
                        )
                
                self.connection.execute("COMMIT")
                return entity_id
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add entity: {e}")
                raise

    def add_entities(
        self,
        entities: List[Entity],
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Add multiple entities and optionally associate them with a memory
        
        Args:
            entities: List of Entity objects
            memory_id: Optional memory ID to associate with
            user_id: Optional user ID for scoping
            agent_id: Optional agent ID for scoping
            
        Returns:
            List of entity IDs
        """
        entity_ids = []
        for entity in entities:
            entity_id = self.add_entity(entity, memory_id, user_id, agent_id)
            entity_ids.append(entity_id)
        return entity_ids

    def get_entities_for_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get all entities associated with a memory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            List of entity dictionaries
        """
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT e.id, e.text, e.entity_type, e.ner_label, e.canonical,
                       e.created_at, e.updated_at, me.confidence
                FROM entities e
                JOIN memory_entities me ON e.id = me.entity_id
                WHERE me.memory_id = ?
                ORDER BY me.confidence DESC
                """,
                (memory_id,),
            )
            rows = cur.fetchall()
            
            return [
                {
                    "id": r[0],
                    "text": r[1],
                    "entity_type": r[2],
                    "ner_label": r[3],
                    "canonical": r[4],
                    "created_at": r[5],
                    "updated_at": r[6],
                    "confidence": r[7],
                }
                for r in rows
            ]

    def get_memories_for_entity(self, entity_text: str) -> List[str]:
        """
        Get all memory IDs associated with an entity (by text)
        
        Args:
            entity_text: Entity text to search for
            
        Returns:
            List of memory IDs
        """
        canonical = entity_text.lower()
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT DISTINCT me.memory_id
                FROM memory_entities me
                JOIN entities e ON me.entity_id = e.id
                WHERE e.canonical = ?
                ORDER BY me.created_at DESC
                """,
                (canonical,),
            )
            rows = cur.fetchall()
            return [r[0] for r in rows]

    def get_memories_for_entities(self, entity_texts: List[str]) -> List[str]:
        """
        Get all memory IDs associated with any of the given entities
        
        Args:
            entity_texts: List of entity texts
            
        Returns:
            List of memory IDs
        """
        if not entity_texts:
            return []
        
        canonicals = [text.lower() for text in entity_texts]
        placeholders = ", ".join("?" for _ in canonicals)
        
        with self._lock:
            cur = self.connection.execute(
                f"""
                SELECT DISTINCT me.memory_id
                FROM memory_entities me
                JOIN entities e ON me.entity_id = e.id
                WHERE e.canonical IN ({placeholders})
                ORDER BY me.created_at DESC
                """,
                canonicals,
            )
            rows = cur.fetchall()
            return [r[0] for r in rows]

    def search_entities(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for entities matching a query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        query_lower = query.lower()
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT e.id, e.text, e.entity_type, e.ner_label, e.canonical,
                       e.created_at, e.updated_at, COUNT(me.id) as memory_count
                FROM entities e
                LEFT JOIN memory_entities me ON e.id = me.entity_id
                WHERE e.canonical LIKE ? OR e.text LIKE ?
                GROUP BY e.id
                ORDER BY memory_count DESC, e.updated_at DESC
                LIMIT ?
                """,
                (f"%{query_lower}%", f"%{query}%", limit),
            )
            rows = cur.fetchall()
            
            return [
                {
                    "id": r[0],
                    "text": r[1],
                    "entity_type": r[2],
                    "ner_label": r[3],
                    "canonical": r[4],
                    "created_at": r[5],
                    "updated_at": r[6],
                    "memory_count": r[7],
                }
                for r in rows
            ]

    def get_all_entities(
        self,
        entity_type: Optional[str] = None,
        limit: Optional[int] = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get all entities, optionally filtered by type
        
        Args:
            entity_type: Optional entity type filter
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        with self._lock:
            query = """
                SELECT e.id, e.text, e.entity_type, e.ner_label, e.canonical,
                       e.created_at, e.updated_at, COUNT(me.id) as memory_count
                FROM entities e
                LEFT JOIN memory_entities me ON e.id = me.entity_id
            """
            params = []
            
            if entity_type:
                query += " WHERE e.entity_type = ?"
                params.append(entity_type)
            
            query += " GROUP BY e.id ORDER BY memory_count DESC, e.updated_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cur = self.connection.execute(query, params)
            rows = cur.fetchall()
            
            return [
                {
                    "id": r[0],
                    "text": r[1],
                    "entity_type": r[2],
                    "ner_label": r[3],
                    "canonical": r[4],
                    "created_at": r[5],
                    "updated_at": r[6],
                    "memory_count": r[7],
                }
                for r in rows
            ]

    def remove_entity_from_memory(self, memory_id: str, entity_id: str) -> bool:
        """
        Remove an entity association from a memory
        
        Args:
            memory_id: Memory ID
            entity_id: Entity ID
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                
                cur.execute(
                    "DELETE FROM memory_entities WHERE memory_id = ? AND entity_id = ?",
                    (memory_id, entity_id),
                )
                
                self.connection.execute("COMMIT")
                return True
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to remove entity from memory: {e}")
                raise

    def delete_entities_for_memory(self, memory_id: str) -> bool:
        """
        Delete all entity associations for a memory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                
                cur.execute(
                    "DELETE FROM memory_entities WHERE memory_id = ?",
                    (memory_id,),
                )
                
                self.connection.execute("COMMIT")
                return True
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to delete entities for memory: {e}")
                raise

    def cleanup_unused_entities(self) -> int:
        """
        Delete entities that are no longer associated with any memories
        
        Returns:
            Number of entities deleted
        """
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                
                cur.execute(
                    """
                    DELETE FROM entities
                    WHERE id NOT IN (SELECT DISTINCT entity_id FROM memory_entities)
                    """
                )
                deleted_count = cur.rowcount
                
                self.connection.execute("COMMIT")
                return deleted_count
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to cleanup unused entities: {e}")
                raise
