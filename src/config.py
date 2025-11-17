import os
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """应用配置管理类"""
    
    # 模型配置
    SCI_MODEL_BASE_URL: str = Field(
        default="https://api.deepseek.com",
        description="模型API基础URL"
    )
    SCI_EMBEDDING_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="嵌入服务API端点"
    )
    SCI_LLM_MODEL: str = Field(
        default="deepseek-chat",
        description="标准LLM模型"
    )
    SCI_LLM_REASONING_MODEL: str = Field(
        default="deepseek-reasoner", 
        description="推理LLM模型"
    )
    SCI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-v4",
        description="嵌入模型"
    )
    
    # API密钥配置
    SCI_MODEL_API_KEY: str = Field(
        default="",
        description="模型API密钥"
    )
    SCI_EMBEDDING_API_KEY: str = Field(
        default="",
        description="嵌入API密钥"
    )
    
    # 服务配置
    PORT: int = Field(
        default=3000,
        description="服务端口"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="日志级别"
    )
    REQUEST_TIMEOUT: int = Field(
        default=900,  # 15分钟
        description="请求超时时间（秒）"
    )
    
    # 性能配置
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=5,
        description="最大并发请求数"
    )
    EMBEDDING_CACHE_TTL: int = Field(
        default=3600,
        description="嵌入缓存TTL（秒）"
    )
    API_RATE_LIMIT: int = Field(
        default=10,
        description="API速率限制（请求/秒）"
    )
    
    # 学术API配置
    OPENALEX_BASE_URL: str = Field(
        default="https://api.openalex.org",
        description="OpenAlex API基础URL"
    )
    SEMANTIC_SCHOLAR_BASE_URL: str = Field(
        default="https://api.semanticscholar.org/graph/v1",
        description="Semantic Scholar API基础URL"
    )
    ARXIV_BASE_URL: str = Field(
        default="http://export.arxiv.org/api/query",
        description="arXiv API基础URL"
    )
    
    # 文档处理配置
    PDF_CHUNK_SIZE: int = Field(
        default=2000,
        description="PDF分块大小"
    )
    PDF_CHUNK_OVERLAP: int = Field(
        default=200,
        description="PDF分块重叠区域"
    )
    MAX_PDF_PAGES: int = Field(
        default=50,
        description="最大PDF页数限制"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 全局配置实例
config = Config()


class ErrorCategories:
    """错误分类和处理策略"""
    
    API_TIMEOUT = {
        "level": "warning",
        "strategy": "retry_then_fallback",
        "max_retries": 2
    }
    
    RATE_LIMIT = {
        "level": "warning", 
        "strategy": "wait_and_retry",
        "backoff_factor": 2.0
    }
    
    PARSING_ERROR = {
        "level": "error",
        "strategy": "fallback_parsing",
        "degrade_gracefully": True
    }
    
    MODEL_ERROR = {
        "level": "error",
        "strategy": "switch_model",
        "fallback_model": "deepseek-chat"
    }


class TimeoutConfig:
    """超时配置管理"""
    
    LITERATURE_REVIEW = 900  # 15分钟
    PAPER_QA = 900  # 15分钟
    IDEATION = 600  # 10分钟
    PAPER_REVIEW = 1200  # 20分钟
    
    @classmethod
    def get_timeout(cls, endpoint: str) -> int:
        """获取端点超时时间"""
        return getattr(cls, endpoint.upper(), cls.LITERATURE_REVIEW)
