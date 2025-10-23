"""
Procedural Memory Manager для AIRI Memory System
Уровень 6: Процедурная память - навыки, алгоритмы, процедуры, TTL = 365 дней
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json
import os

logger = logging.getLogger(__name__)

from chromadb import PersistentClient
from chromadb.config import Settings
try:
    from ..storage.goals_sqlite import GoalsSQLiteStorage
except Exception:  # pragma: no cover
    from storage.goals_sqlite import GoalsSQLiteStorage

logger = logging.getLogger(__name__)
try:
    from ..analyzers.behavior_analyzer import analyze_patterns, build_recommendations
except Exception:  # pragma: no cover
    from analyzers.behavior_analyzer import analyze_patterns, build_recommendations

@dataclass
class ProceduralMemoryItem:
    """Элемент процедурной памяти"""
    id: str
    name: str
    description: str
    user_id: str
    timestamp: datetime
    skill_type: str = "general"
    difficulty: float = 0.5
    proficiency: float = 0.0
    steps: List[str] = None
    prerequisites: List[str] = None
    related_skills: List[str] = None
    success_rate: float = 0.0
    practice_count: int = 0
    last_practiced: Optional[datetime] = None
    importance: float = 0.5

class ProceduralMemoryManager:
    """Менеджер процедурной памяти"""
    
    def __init__(self, chromadb_path: str = "./data/chroma_db", db_path: Optional[str] = None):
        self.chromadb_path = chromadb_path
        self.client = PersistentClient(
            path=chromadb_path,
            settings=Settings(anonymized_telemetry=False)
        )
        # Используем SentenceTransformerEmbeddingFunction для правильных distances
        try:
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer, falling back to DefaultEmbeddingFunction: {e}")
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="procedural_memory",
            metadata={"description": "Procedural memory for skills, algorithms, and procedures"},
            embedding_function=self.embedding_function
        )
        # Ensure goals table exists via global initializer 
        # Resolve SQLite path from env if not provided
        if db_path is None:
            data_root = os.getenv("AIRI_DATA_DIR", "./data")
            db_path = os.getenv("SQLITE_DB", os.path.join(data_root, "memory_system.db"))
        self._goals = GoalsSQLiteStorage(db_path=db_path)
        # Коллекция целей/задач (goals)
        # goals persist in SQLite storage
        self.max_items = 1000  # Максимум элементов в процедурной памяти
        self.ttl_days = 365      # Время жизни элемента (365 дней)
        
    async def add_skill(self, name: str, description: str, user_id: str,
                       skill_type: str = "general", difficulty: float = 0.5,
                       steps: List[str] = None, prerequisites: List[str] = None,
                       related_skills: List[str] = None, importance: float = 0.5,
                       proficiency: float = 0.0) -> str:
        """
        Добавить навык в процедурную память
        
        Args:
            name: Название навыка
            description: Описание навыка
            user_id: ID пользователя
            skill_type: Тип навыка (cognitive, motor, social, etc.)
            difficulty: Сложность навыка (0.0-1.0)
            steps: Шаги выполнения навыка
            prerequisites: Предварительные требования
            related_skills: Связанные навыки
            importance: Важность навыка (0.0-1.0)
            
        Returns:
            ID созданного навыка
        """
        try:
            skill_id = f"pm_{uuid.uuid4()}"
            timestamp = datetime.now()
            
            # Создаем метаданные
            metadata = {
                "user_id": user_id,
                "name": name,
                "skill_type": skill_type,
                "difficulty": difficulty,
                "proficiency": max(0.0, min(1.0, proficiency)),
                "steps": json.dumps(steps) if steps else "[]",
                "prerequisites": json.dumps(prerequisites) if prerequisites else "[]",
                "related_skills": json.dumps(related_skills) if related_skills else "[]",
                "success_rate": 0.0,
                "practice_count": 0,
                "last_practiced": timestamp.isoformat(),
                "importance": importance,
                "timestamp": timestamp.isoformat(),
                "memory_type": "procedural"
            }
            
            # Создаем документ для поиска
            doc = f"{name}: {description}"
            
            # Добавляем в ChromaDB
            self.collection.add(
                documents=[doc],
                metadatas=[metadata],
                ids=[skill_id]
            )
            
            # Очищаем старые элементы
            await self._cleanup_old_memories(user_id)
            
            logger.info(f"Added procedural skill: {skill_id}")
            return skill_id
            
        except Exception as e:
            logger.error(f"Error adding procedural skill: {e}")
            raise
    
    async def practice_skill(self, skill_id: str, success: bool) -> bool:
        """
        Практиковать навык
        
        Args:
            skill_id: ID навыка
            success: Успешность практики
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[skill_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            current_practice_count = metadata.get("practice_count", 0)
            current_success_rate = metadata.get("success_rate", 0.0)
            current_proficiency = metadata.get("proficiency", 0.0)
            
            # Обновляем статистику
            new_practice_count = current_practice_count + 1
            new_success_rate = ((current_success_rate * current_practice_count) + (1.0 if success else 0.0)) / new_practice_count
            
            # Обновляем мастерство на основе успешности
            proficiency_increase = 0.1 if success else -0.05
            new_proficiency = max(0.0, min(1.0, current_proficiency + proficiency_increase))
            
            metadata["practice_count"] = new_practice_count
            metadata["success_rate"] = new_success_rate
            metadata["proficiency"] = new_proficiency
            metadata["last_practiced"] = datetime.now().isoformat()
            
            # Обновляем метаданные
            self.collection.update(
                ids=[skill_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated practice stats for {skill_id}: success={success}")
            return True
            
        except Exception as e:
            logger.error(f"Error practicing skill: {e}")
            return False
    
    async def get_skill(self, skill_id: str) -> Optional[ProceduralMemoryItem]:
        """
        Получить навык по ID
        
        Args:
            skill_id: ID навыка
            
        Returns:
            Навык или None
        """
        try:
            results = self.collection.get(ids=[skill_id], include=["metadatas"])
            if not results["metadatas"]:
                return None
            
            metadata = results["metadatas"][0]
            return ProceduralMemoryItem(
                id=skill_id,
                name=metadata["name"],
                description=results["documents"][0].split(": ", 1)[1] if ": " in results["documents"][0] else "",
                user_id=metadata["user_id"],
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                skill_type=metadata.get("skill_type", "general"),
                difficulty=metadata.get("difficulty", 0.5),
                proficiency=metadata.get("proficiency", 0.0),
                steps=json.loads(metadata.get("steps", "[]")),
                prerequisites=json.loads(metadata.get("prerequisites", "[]")),
                related_skills=json.loads(metadata.get("related_skills", "[]")),
                success_rate=metadata.get("success_rate", 0.0),
                practice_count=metadata.get("practice_count", 0),
                last_practiced=datetime.fromisoformat(metadata.get("last_practiced", metadata["timestamp"])),
                importance=metadata.get("importance", 0.5)
            )
            
        except Exception as e:
            logger.error(f"Error getting skill: {e}")
            return None
    
    async def search_skills(self, user_id: str, query: str, 
                           skill_type: Optional[str] = None,
                           min_proficiency: float = 0.0, limit: int = 10) -> List[ProceduralMemoryItem]:
        """
        Поиск навыков
        
        Args:
            user_id: ID пользователя
            query: Поисковый запрос
            skill_type: Тип навыка для фильтрации
            min_proficiency: Минимальное мастерство
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных навыков
        """
        try:
            # Строим фильтр
            where_filter = {"user_id": user_id}
            if skill_type:
                where_filter = {"$and": [{"user_id": user_id}, {"skill_type": skill_type}]}
            
            results = self.collection.query(
                query_texts=[query],
                where=where_filter,
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            skills = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                proficiency = metadata.get("proficiency", 0.0)
                
                # Фильтруем по релевантности и мастерству
                if distance < 0.8 and proficiency >= min_proficiency:
                    skill = ProceduralMemoryItem(
                        id=results["ids"][0][i],
                        name=metadata["name"],
                        description=doc.split(": ", 1)[1] if ": " in doc else "",
                        user_id=metadata["user_id"],
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        skill_type=metadata.get("skill_type", "general"),
                        difficulty=metadata.get("difficulty", 0.5),
                        proficiency=proficiency,
                        steps=json.loads(metadata.get("steps", "[]")),
                        prerequisites=json.loads(metadata.get("prerequisites", "[]")),
                        related_skills=json.loads(metadata.get("related_skills", "[]")),
                        success_rate=metadata.get("success_rate", 0.0),
                        practice_count=metadata.get("practice_count", 0),
                        last_practiced=datetime.fromisoformat(metadata.get("last_practiced", metadata["timestamp"])),
                        importance=metadata.get("importance", 0.5)
                    )
                    skills.append(skill)
            
            return skills
            
        except Exception as e:
            logger.error(f"Error searching skills: {e}")
            return []
    
    async def get_skills_by_type(self, user_id: str, skill_type: str, 
                                limit: int = 50) -> List[ProceduralMemoryItem]:
        """
        Получить навыки по типу
        
        Args:
            user_id: ID пользователя
            skill_type: Тип навыка
            limit: Максимальное количество элементов
            
        Returns:
            Список навыков указанного типа
        """
        try:
            results = self.collection.get(
                where={"$and": [{"user_id": user_id}, {"skill_type": skill_type}]},
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            skills = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                skill = ProceduralMemoryItem(
                    id=results["ids"][i],
                    name=metadata["name"],
                    description=doc.split(": ", 1)[1] if ": " in doc else "",
                    user_id=metadata["user_id"],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    skill_type=metadata.get("skill_type", "general"),
                    difficulty=metadata.get("difficulty", 0.5),
                    proficiency=metadata.get("proficiency", 0.0),
                    steps=json.loads(metadata.get("steps", "[]")),
                    prerequisites=json.loads(metadata.get("prerequisites", "[]")),
                    related_skills=json.loads(metadata.get("related_skills", "[]")),
                    success_rate=metadata.get("success_rate", 0.0),
                    practice_count=metadata.get("practice_count", 0),
                    last_practiced=datetime.fromisoformat(metadata.get("last_practiced", metadata["timestamp"])),
                    importance=metadata.get("importance", 0.5)
                )
                skills.append(skill)
            
            # Сортируем по мастерству
            skills.sort(key=lambda x: x.proficiency, reverse=True)
            
            return skills
            
        except Exception as e:
            logger.error(f"Error getting skills by type: {e}")
            return []
    
    async def get_learning_path(self, user_id: str, target_skill: str) -> List[ProceduralMemoryItem]:
        """
        Получить путь обучения для навыка
        
        Args:
            user_id: ID пользователя
            target_skill: Целевой навык
            
        Returns:
            Список навыков в порядке обучения
        """
        try:
            # Находим целевой навык
            target_skills = await self.search_skills(user_id, target_skill)
            if not target_skills:
                return []
            
            target = target_skills[0]
            learning_path = []
            visited = set()
            
            # Рекурсивно строим путь через предварительные требования
            async def build_path(skill_id: str):
                if skill_id in visited:
                    return
                
                visited.add(skill_id)
                skill = await self.get_skill(skill_id)
                if not skill:
                    return
                
                # Добавляем предварительные требования
                for prereq in skill.prerequisites:
                    if prereq not in visited:
                        prereq_skills = await self.search_skills(user_id, prereq)
                        if prereq_skills:
                            await build_path(prereq_skills[0].id)
                
                learning_path.append(skill)
            
            await build_path(target.id)
            return learning_path
            
        except Exception as e:
            logger.error(f"Error getting learning path: {e}")
            return []
    
    async def get_related_skills(self, skill_id: str) -> List[ProceduralMemoryItem]:
        """
        Получить связанные навыки
        
        Args:
            skill_id: ID навыка
            
        Returns:
            Список связанных навыков
        """
        try:
            skill = await self.get_skill(skill_id)
            if not skill or not skill.related_skills:
                return []
            
            related = []
            for related_name in skill.related_skills:
                related_skills = await self.search_skills(skill.user_id, related_name)
                if related_skills:
                    related.extend(related_skills)
            
            return related
            
        except Exception as e:
            logger.error(f"Error getting related skills: {e}")
            return []
    
    async def update_proficiency(self, skill_id: str, proficiency: float) -> bool:
        """
        Обновить мастерство навыка
        
        Args:
            skill_id: ID навыка
            proficiency: Новое мастерство (0.0-1.0)
            
        Returns:
            True если успешно
        """
        try:
            # Получаем текущие метаданные
            results = self.collection.get(ids=[skill_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            metadata = results["metadatas"][0]
            metadata["proficiency"] = max(0.0, min(1.0, proficiency))
            
            # Обновляем метаданные
            self.collection.update(
                ids=[skill_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated proficiency for {skill_id}: {proficiency}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating proficiency: {e}")
            return False

    async def update_skill_progress(self, skill_id: str, proficiency_increase: float = 0.0, success: bool = True) -> bool:
        """Совместимый метод: обновляет прогресс навыка (мастерство и статистику практики)."""
        try:
            # Обновляем статистику практики
            _ = await self.practice_skill(skill_id, success=success)
            # Корректируем мастерство на заданную прибавку
            res = self.collection.get(ids=[skill_id], include=["metadatas"])
            if not res["metadatas"]:
                return False
            meta = res["metadatas"][0]
            current = meta.get("proficiency", 0.0)
            new_value = max(0.0, min(1.0, current + proficiency_increase))
            return await self.update_proficiency(skill_id, new_value)
        except Exception:
            return False

    async def add_goal(self, user_id: str, name: str, description: str, next_run: Optional[datetime] = None) -> str:
        """Добавить цель/задачу в процедурную память."""
        goal_id = f"pg_{uuid.uuid4()}"
        ts = datetime.now()
        metadata = {
            "user_id": user_id,
            "name": name,
            "description": description,
            "status": "active",
            "progress": 0.0,
            "timestamp": ts.isoformat(),
            "next_run": next_run.isoformat() if next_run else ts.isoformat(),
            "memory_type": "procedural_goal"
        }
        self._goals.add_goal(goal_id, user_id, name, description, next_run)
        return goal_id

    async def get_active_goals(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Вернуть активные цели пользователя."""
        return self._goals.get_active_goals(user_id, limit=limit)

    async def update_goal_progress(self, goal_id: str, progress: float, status: Optional[str] = None) -> bool:
        """Обновить прогресс/статус цели."""
        self._goals.update_goal(goal_id, progress, status)
        return True
    
    async def remove_skill(self, skill_id: str) -> bool:
        """
        Удалить навык из процедурной памяти
        
        Args:
            skill_id: ID навыка
            
        Returns:
            True если успешно
        """
        try:
            self.collection.delete(ids=[skill_id])
            logger.info(f"Removed procedural skill: {skill_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing skill: {e}")
            return False
    
    async def _cleanup_old_memories(self, user_id: str):
        """Очистка старых элементов процедурной памяти"""
        try:
            # Получаем все элементы пользователя
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            current_time = datetime.now()
            to_remove = []
            
            for i, metadata in enumerate(results["metadatas"]):
                timestamp = datetime.fromisoformat(metadata["timestamp"])
                age_days = (current_time - timestamp).total_seconds() / (24 * 3600)
                importance = metadata.get("importance", 0.5)
                proficiency = metadata.get("proficiency", 0.0)
                practice_count = metadata.get("practice_count", 0)
                
                # Удаляем старые навыки с низкой важностью, мастерством и количеством практик
                if (age_days > self.ttl_days and 
                    importance < 0.3 and 
                    proficiency < 0.2 and 
                    practice_count < 3):
                    to_remove.append(results["ids"][i])
            
            # Удаляем старые элементы
            if to_remove:
                self.collection.delete(ids=to_remove)
                logger.info(f"Cleaned up {len(to_remove)} old procedural memories")
            # Удаляем завершённые цели (SQLite)
            removed_goals = 0
            try:
                removed_goals = self._goals.delete_completed_goals(user_id)
                if removed_goals:
                    logger.info(f"Removed {removed_goals} completed procedural goals")
            except Exception:
                pass
            
            # Ограничиваем количество элементов
            all_results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if len(all_results["ids"]) > self.max_items:
                # Сортируем по важности, мастерству и количеству практик
                items_with_score = []
                for i, metadata in enumerate(all_results["metadatas"]):
                    importance = metadata.get("importance", 0.5)
                    proficiency = metadata.get("proficiency", 0.0)
                    practice_count = metadata.get("practice_count", 0)
                    score = importance * 0.4 + proficiency * 0.4 + min(practice_count / 10, 1.0) * 0.2
                    items_with_score.append((all_results["ids"][i], score))
                
                items_with_score.sort(key=lambda x: x[1])
                excess_count = len(items_with_score) - self.max_items
                to_remove = [item[0] for item in items_with_score[:excess_count]]
                
                if to_remove:
                    self.collection.delete(ids=to_remove)
                    logger.info(f"Removed {len(to_remove)} excess procedural memories")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")

    async def cleanup_old_memories(self, user_id: str) -> int:
        """Публичный метод для тестов: выполнить очистку и вернуть число удалённых объектов (примерно)."""
        before_nodes = self.collection.count()
        before_goals = self._goals.count()
        await self._cleanup_old_memories(user_id)
        after_nodes = self.collection.count()
        after_goals = self._goals.count()
        removed = max(0, (before_nodes - after_nodes)) + max(0, (before_goals - after_goals))
        return removed
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Получить статистику процедурной памяти
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                return {
                    "total_skills": 0,
                    "avg_proficiency": 0.0,
                    "avg_success_rate": 0.0,
                    "skill_types": {},
                    "most_practiced": 0
                }
            
            proficiencies = [m.get("proficiency", 0.0) for m in results["metadatas"]]
            success_rates = [m.get("success_rate", 0.0) for m in results["metadatas"]]
            practice_counts = [m.get("practice_count", 0) for m in results["metadatas"]]
            skill_types = {}
            
            for metadata in results["metadatas"]:
                skill_type = metadata.get("skill_type", "general")
                skill_types[skill_type] = skill_types.get(skill_type, 0) + 1
            
            base = {
                "total_skills": len(results["metadatas"]),
                "avg_proficiency": sum(proficiencies) / len(proficiencies),
                "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
                "skill_types": skill_types,
                "most_practiced": max(practice_counts) if practice_counts else 0
            }
            # Поведенческий анализ и рекомендации
            patterns = analyze_patterns(results["metadatas"]) if results.get("metadatas") else {}
            recs = build_recommendations(patterns) if patterns else []
            base["behavior_patterns"] = patterns
            base["recommendations"] = recs
            return base
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[ProceduralMemoryItem]:
        """
        Получить элемент процедурной памяти по ID
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            
        Returns:
            ProceduralMemoryItem или None если не найден
        """
        try:
            results = self.collection.get(
                ids=[memory_id],
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
            
            if not results["documents"] or not results["metadatas"]:
                return None
            
            doc = results["documents"][0]
            metadata = results["metadatas"][0]
            
            # Парсим timestamp
            ts = datetime.fromisoformat(metadata["timestamp"])
            
            # Парсим steps, prerequisites, related_skills
            steps = json.loads(metadata.get("steps", "[]")) if metadata.get("steps") else []
            prerequisites = json.loads(metadata.get("prerequisites", "[]")) if metadata.get("prerequisites") else []
            related_skills = json.loads(metadata.get("related_skills", "[]")) if metadata.get("related_skills") else []
            
            # Парсим last_practiced если есть
            last_practiced = None
            if metadata.get("last_practiced"):
                last_practiced = datetime.fromisoformat(metadata["last_practiced"])
            
            return ProceduralMemoryItem(
                id=memory_id,
                name=metadata.get("name", ""),
                description=doc,
                user_id=metadata["user_id"],
                timestamp=ts,
                skill_type=metadata.get("skill_type", "general"),
                difficulty=metadata.get("difficulty", 0.5),
                proficiency=metadata.get("proficiency", 0.0),
                steps=steps,
                prerequisites=prerequisites,
                related_skills=related_skills,
                success_rate=metadata.get("success_rate", 0.0),
                practice_count=metadata.get("practice_count", 0),
                last_practiced=last_practiced,
                importance=metadata.get("importance", 0.5)
            )
            
        except Exception as e:
            logger.error(f"Error getting procedural memory by ID {memory_id}: {e}")
            return None
    
    async def update_metadata(self, memory_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновить метаданные элемента процедурной памяти
        
        Args:
            memory_id: ID элемента памяти
            user_id: ID пользователя
            metadata: Новые метаданные
            
        Returns:
            True если успешно, False если ошибка
        """
        try:
            # Получаем текущий элемент
            current_item = await self.get_memory_by_id(memory_id, user_id)
            if not current_item:
                logger.warning(f"Procedural memory item {memory_id} not found for user {user_id}")
                return False
            
            # Обновляем метаданные
            updated_metadata = {
                "user_id": user_id,
                "timestamp": current_item.timestamp.isoformat(),
                "name": current_item.name,
                "skill_type": current_item.skill_type,
                "difficulty": current_item.difficulty,
                "proficiency": current_item.proficiency,
                "steps": json.dumps(current_item.steps) if current_item.steps else None,
                "prerequisites": json.dumps(current_item.prerequisites) if current_item.prerequisites else None,
                "related_skills": json.dumps(current_item.related_skills) if current_item.related_skills else None,
                "success_rate": current_item.success_rate,
                "practice_count": current_item.practice_count,
                "last_practiced": current_item.last_practiced.isoformat() if current_item.last_practiced else None,
                "importance": current_item.importance
            }
            
            # Добавляем новые метаданные
            updated_metadata.update(metadata)
            
            # Обновляем в ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            
            logger.debug(f"Updated metadata for procedural memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating procedural memory metadata {memory_id}: {e}")
            return False
