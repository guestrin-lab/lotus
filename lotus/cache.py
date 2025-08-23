import copy
import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from functools import wraps
from typing import Any, Callable

import pandas as pd

import lotus


def require_cache_enabled(func: Callable) -> Callable:
    """Decorator to check if caching is enabled before calling the function."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not lotus.settings.enable_cache:
            return None
        return func(self, *args, **kwargs)

    return wrapper


def operator_cache(func: Callable) -> Callable:
    """Decorator to add operator level caching."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        model = lotus.settings.lm
        use_operator_cache = lotus.settings.enable_cache

        if use_operator_cache and model.cache:

            def serialize(value: Any) -> Any:
                """
                Serialize a value into a JSON-serializable format.
                Supports basic types, pandas DataFrames, Enums, and objects with a `dict` or `__dict__` method.
                """
                if value is None or isinstance(value, (str, int, float, bool)):
                    return value
                elif isinstance(value, pd.DataFrame):
                    return value.to_json(orient="split")
                elif isinstance(value, (list, tuple)):
                    return [serialize(item) for item in value]
                elif isinstance(value, dict):
                    return {key: serialize(val) for key, val in value.items()}
                elif isinstance(value, Enum):
                    return {"__enum__": value.__class__.__name__, "value": value.name}
                elif hasattr(value, "dict"):
                    return value.dict()
                elif hasattr(value, "__dict__"):
                    return {key: serialize(val) for key, val in vars(value).items() if not key.startswith("_")}
                else:
                    # For unsupported types, convert to string (last resort)
                    lotus.logger.warning(f"Unsupported type {type(value)} for serialization. Converting to string.")
                    return str(value)

            serialize_self = serialize(self._obj)
            serialized_kwargs = {key: serialize(value) for key, value in kwargs.items()}
            serialized_args = [serialize(arg) for arg in args]
            cache_key = hashlib.sha256(
                json.dumps(
                    {"self": serialize_self, "args": serialized_args, "kwargs": serialized_kwargs}, sort_keys=True
                ).encode()
            ).hexdigest()
            virtual_usage_cache_key = cache_key + "_usage"

            cached_result = model.cache.get(cache_key)
            if cached_result is not None:
                lotus.logger.debug(f"Cache hit for {cache_key}")
                model.stats.operator_cache_hits += 1

                cached_virtual_usage = model.cache.get(virtual_usage_cache_key)
                if cached_virtual_usage is not None:
                    model.stats.virtual_usage += cached_virtual_usage

                return cached_result
            lotus.logger.debug(f"Cache miss for {cache_key}")

            virtual_usage_before = copy.deepcopy(lotus.settings.lm.stats.virtual_usage)
            result = func(self, *args, **kwargs)
            virtual_usage = lotus.settings.lm.stats.virtual_usage - virtual_usage_before
            model.cache.insert(virtual_usage_cache_key, virtual_usage)
            model.cache.insert(cache_key, result)
            return result

        return func(self, *args, **kwargs)

    return wrapper


class CacheType(Enum):
    IN_MEMORY = "in_memory"
    SQLITE = "sqlite"


class CacheConfig:
    def __init__(self, cache_type: CacheType, max_size: int, **kwargs):
        self.cache_type = cache_type
        self.max_size = max_size
        self.kwargs = kwargs


class Cache(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size

    @abstractmethod
    def get(self, key: str) -> Any | None:
        pass

    @abstractmethod
    def insert(self, key: str, value: Any):
        pass

    @abstractmethod
    def reset(self, max_size: int | None = None):
        pass


class CacheFactory:
    @staticmethod
    def create_cache(config: CacheConfig) -> Cache:
        if config.cache_type == CacheType.IN_MEMORY:
            return InMemoryCache(max_size=config.max_size)
        elif config.cache_type == CacheType.SQLITE:
            cache_dir = config.kwargs.get("cache_dir", os.path.expanduser("~/.lotus/cache"))
            if not isinstance(cache_dir, str):
                raise ValueError("cache_dir must be a string")
            return SQLiteCache(max_size=config.max_size, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported cache type: {config.cache_type}")

    @staticmethod
    def create_default_cache(max_size: int = 1024) -> Cache:
        return CacheFactory.create_cache(CacheConfig(CacheType.IN_MEMORY, max_size))


class ThreadLocalConnection:
    """Wrapper that automatically closes connection when thread dies"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def __del__(self):
        if self._conn is not None:
            self._conn.close()


class SQLiteCache(Cache):
    def __init__(self, max_size: int, cache_dir=os.path.expanduser("~/.lotus/cache")):
        super().__init__(max_size)
        self.db_path = os.path.join(cache_dir, "lotus_cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._local = threading.local()
        self._create_table()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn_wrapper"):
            self._local.conn_wrapper = ThreadLocalConnection(self.db_path)
        return self._local.conn_wrapper.connection

    def _create_table(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    last_accessed INTEGER
                )
            """)

    def _get_time(self):
        return int(time.time())

    def get(self, key: str) -> Any | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                lotus.logger.debug(f"Cache hit for {key}")
                value = pickle.loads(result[0])
                conn.execute(
                    "UPDATE cache SET last_accessed = ? WHERE key = ?",
                    (
                        self._get_time(),
                        key,
                    ),
                )
                return value
            cursor.close()
        return None

    def insert(self, key: str, value: Any):
        pickled_value = pickle.dumps(value)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, last_accessed) 
                VALUES (?, ?, ?)
            """,
                (key, pickled_value, self._get_time()),
            )
            self._enforce_size_limit()

    def _enforce_size_limit(self):
        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count > self.max_size:
                num_to_delete = count - self.max_size
                conn.execute(
                    """
                    DELETE FROM cache WHERE key IN (
                        SELECT key FROM cache
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                """,
                    (num_to_delete,),
                )

    def reset(self, max_size: int | None = None):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM cache")
        if max_size is not None:
            self.max_size = max_size


class InMemoryCache(Cache):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key in self.cache:
            lotus.logger.debug(f"Cache hit for {key}")

        return self.cache.get(key)

    def insert(self, key: str, value: Any):
        self.cache[key] = value

        # LRU eviction
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def reset(self, max_size: int | None = None):
        self.cache.clear()
        if max_size is not None:
            self.max_size = max_size
