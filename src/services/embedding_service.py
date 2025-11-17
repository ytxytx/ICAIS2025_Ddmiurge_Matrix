import asyncio
import logging
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from cachetools import TTLCache
import numpy as np

from ..config import config
from openai import AsyncOpenAI


class EmbeddingService:
    """嵌入和相似度服务 - 处理文本嵌入计算和相似度分析"""
    
    def __init__(self):
        self.logger = logging.getLogger("service.embedding")
        
        # 初始化嵌入客户端
        self.client = AsyncOpenAI(
            base_url=config.SCI_EMBEDDING_BASE_URL,
            api_key=config.SCI_EMBEDDING_API_KEY
        )
        
        # 缓存配置
        self.cache = TTLCache(
            maxsize=1000,
            ttl=config.EMBEDDING_CACHE_TTL
        )
        
        # 性能配置
        self.batch_size = 32
        self.max_retries = 3
        self.retry_delay = 1.0

    async def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        cache_key = self._get_cache_key(text)
        
        # 检查缓存
        if cache_key in self.cache:
            self.logger.debug(f"缓存命中: {text[:50]}...")
            return self.cache[cache_key]
        
        try:
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=config.SCI_EMBEDDING_MODEL,
                        input=text
                    )
                    
                    embedding = response.data[0].embedding
                    
                    # 验证嵌入向量
                    if self._validate_embedding(embedding):
                        # 缓存结果
                        self.cache[cache_key] = embedding
                        self.logger.debug(f"嵌入计算成功: {text[:50]}... (维度: {len(embedding)})")
                        return embedding
                    else:
                        self.logger.warning(f"嵌入向量验证失败: {text[:50]}...")
                        break
                        
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"嵌入计算失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
            
        except Exception as e:
            self.logger.error(f"嵌入计算最终失败: {str(e)}")
        
        return []

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量"""
        if not texts:
            return []
        
        self.logger.info(f"批量计算嵌入向量，文本数量: {len(texts)}")
        
        # 分批处理
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = []
        
        for batch in batches:
            try:
                # 检查缓存
                cached_embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                for i, text in enumerate(batch):
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.cache:
                        cached_embeddings.append(self.cache[cache_key])
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                # 为未缓存的文本计算嵌入
                if uncached_texts:
                    response = await self.client.embeddings.create(
                        model=config.SCI_EMBEDDING_MODEL,
                        input=uncached_texts
                    )
                    
                    # 处理响应并缓存结果
                    for i, embedding_data in enumerate(response.data):
                        embedding = embedding_data.embedding
                        if self._validate_embedding(embedding):
                            text = uncached_texts[i]
                            cache_key = self._get_cache_key(text)
                            self.cache[cache_key] = embedding
                            cached_embeddings.insert(uncached_indices[i], embedding)
                        else:
                            # 插入空向量作为占位符
                            cached_embeddings.insert(uncached_indices[i], [])
                
                all_embeddings.extend(cached_embeddings)
                
            except Exception as e:
                self.logger.error(f"批量嵌入计算失败: {str(e)}")
                # 为失败的批次添加空向量
                all_embeddings.extend([[] for _ in range(len(batch))])
        
        self.logger.info(f"批量嵌入计算完成，成功: {len([e for e in all_embeddings if e])}/{len(all_embeddings)}")
        return all_embeddings

    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            
            # 确保向量维度一致
            if len(a) != len(b):
                min_len = min(len(a), len(b))
                a = a[:min_len]
                b = b[:min_len]
            
            # 计算余弦相似度
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"相似度计算失败: {str(e)}")
            return 0.0

    def find_most_similar(self, query_embedding: List[float], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找与查询向量最相似的候选项"""
        if not query_embedding or not candidates:
            return []
        
        similarities = []
        
        for candidate in candidates:
            candidate_embedding = candidate.get('embedding')
            if not candidate_embedding:
                continue
                
            similarity = self.calculate_similarity(query_embedding, candidate_embedding)
            similarities.append({
                'item': candidate,
                'similarity': similarity
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回排序后的结果
        return [{
            'item': sim['item'],
            'similarity': sim['similarity'],
            'rank': i + 1
        } for i, sim in enumerate(similarities)]

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()

    def _validate_embedding(self, embedding: List[float]) -> bool:
        """验证嵌入向量的有效性"""
        if not embedding:
            return False
        
        # 检查向量维度
        if len(embedding) == 0:
            return False
        
        # 检查向量是否包含NaN或无穷大值
        if any(np.isnan(val) or np.isinf(val) for val in embedding):
            return False
        
        # 检查向量范数（避免零向量）
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return False
        
        return True

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """归一化嵌入向量"""
        if not embedding:
            return []
        
        try:
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            
            if norm == 0:
                return embedding
            
            normalized = vec / norm
            return normalized.tolist()
            
        except Exception as e:
            self.logger.error(f"向量归一化失败: {str(e)}")
            return embedding

    async def semantic_search(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """语义搜索 - 基于查询在文档集合中查找最相关的文档"""
        if not query or not documents:
            return []
        
        self.logger.info(f"执行语义搜索，查询: {query[:50]}..., 文档数量: {len(documents)}")
        
        try:
            # 获取查询嵌入
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                return []
            
            # 为文档计算嵌入（如果还没有）
            documents_with_embeddings = []
            for doc in documents:
                if 'embedding' not in doc and 'text' in doc:
                    doc_embedding = await self.get_embedding(doc['text'])
                    doc['embedding'] = doc_embedding
                documents_with_embeddings.append(doc)
            
            # 查找最相似的文档
            similar_docs = self.find_most_similar(query_embedding, documents_with_embeddings)
            
            # 返回前top_k个结果
            results = similar_docs[:top_k]
            self.logger.info(f"语义搜索完成，返回 {len(results)} 个结果")
            
            return results
            
        except Exception as e:
            self.logger.error(f"语义搜索失败: {str(e)}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.cache),
            'cache_max_size': self.cache.maxsize,
            'cache_ttl': self.cache.ttl,
            'hit_rate': getattr(self.cache, 'hit_count', 0) / max(getattr(self.cache, 'total_count', 1), 1)
        }


class SimilarityResult:
    """相似度结果类"""
    
    def __init__(self, item: Any, similarity: float, rank: int):
        self.item = item
        self.similarity = similarity
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item': self.item,
            'similarity': self.similarity,
            'rank': self.rank
        }


class EmbeddingCache:
    """嵌入缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.hit_count = 0
        self.total_count = 0
    
    async def get_embedding(self, text: str, compute_func) -> List[float]:
        """获取嵌入向量，使用缓存"""
        self.total_count += 1
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.cache:
            self.hit_count += 1
            return self.cache[cache_key]
        
        embedding = await compute_func(text)
        if embedding:
            self.cache[cache_key] = embedding
        
        return embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self.cache),
            'max_size': self.cache.maxsize,
            'hit_count': self.hit_count,
            'total_count': self.total_count,
            'hit_rate': self.hit_count / max(self.total_count, 1)
        }
