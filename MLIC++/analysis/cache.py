import os
import pickle
from functools import lru_cache, wraps
from pathlib import Path

class SimpleCache:
    def __init__(self, cache_dir="simple_cache", memory_cache_size=100):
        """
        初始化简单缓存系统。
        :param cache_dir: 磁盘缓存目录。
        :param memory_cache_size: 内存缓存大小（LRU 缓存的最大条目数）。
        """
        self.cache_dir = Path(cache_dir)
        self.memory_cache_size = memory_cache_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_functions = []  # 用于存储所有被缓存的函数

    def cached(self, func):
        """
        缓存装饰器，自动处理内存和磁盘缓存。
        :param func: 需要缓存的函数。
        :return: 装饰后的函数。
        """
        # 使用 lru_cache 作为内存缓存
        func = lru_cache(maxsize=self.memory_cache_size)(func)
        self.cached_functions.append(func)  # 将被缓存的函数添加到列表中

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键（基于函数名和参数）
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            # 检查磁盘缓存
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    try:
                        result = pickle.load(f)
                        if isinstance(result, (dict, list)):  # 确保 result 是预期的类型
                            return result
                        else:
                            print(f"Warning: Unexpected type in cache file: {type(result)}")
                    except (pickle.PickleError, TypeError) as e:
                        print(f"Error loading cache file: {e}")

            # 调用函数进行计算（lru_cache 会自动处理内存缓存）
            result = func(*args, **kwargs)

            # 将结果存入磁盘缓存
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    def clear_cache(self, memory_only=False):
        """
        清除内存和磁盘缓存。
        :param memory_only: 如果为 True，只清除内存缓存；否则清除内存和磁盘缓存。
        """
        # # 清除内存缓存
        # for func in self.cached_functions:
        #     func.cache_clear()

        # 清除磁盘缓存
        if not memory_only:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()

    @staticmethod
    def _generate_cache_key(func_name, args, kwargs):
        """
        生成缓存键（基于函数名和参数）。
        :param func_name: 函数名。
        :param args: 函数的位置参数。
        :param kwargs: 函数的关键字参数。
        :return: 缓存键（字符串）。
        """
        # 将参数转换为字符串
        args_str = ",".join(map(str, args))
        kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{func_name}({args_str},{kwargs_str})"