from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class Paper(BaseModel):
    """论文基础信息模型"""
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    embedding: Optional[List[float]] = None


class PaperDetails(BaseModel):
    """论文详细信息模型"""
    paper: Paper
    full_text: Optional[str] = None
    references: List[Paper] = []
    citations: List[Paper] = []
    keywords: List[str] = []
    topics: List[str] = []


class DocumentChunk(BaseModel):
    """文档分块模型"""
    content: str
    page_number: int
    chunk_index: int
    section_type: Optional[str] = None  # abstract, introduction, methodology, etc.


class DocumentStructure(BaseModel):
    """文档结构模型"""
    title: Optional[str] = None
    abstract: Optional[str] = None
    introduction: Optional[str] = None
    methodology: Optional[str] = None
    experiments: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None
    references: List[str] = []


class PaperElements(BaseModel):
    """论文关键元素提取模型"""
    contributions: List[str] = []
    methods: List[str] = []
    datasets: List[str] = []
    metrics: List[str] = []
    findings: List[str] = []
    limitations: List[str] = []


class QueryAnalysis(BaseModel):
    """查询分析结果模型"""
    domain: str
    keywords: List[str]
    intent: str
    embedding: List[float]
    complexity: str = "medium"  # low, medium, high


class KnowledgeBase(BaseModel):
    """知识库模型"""
    papers: List[Paper]
    trends: List[str] = []
    gaps: List[str] = []
    related_works: List[Paper] = []


class ResearchIdea(BaseModel):
    """研究想法模型"""
    title: str
    description: str
    methodology: str
    expected_impact: str
    feasibility: str  # low, medium, high
    novelty_score: float  # 0-10


class RatedIdea(BaseModel):
    """评分后的研究想法模型"""
    idea: ResearchIdea
    scores: Dict[str, float]  # novelty, feasibility, impact, clarity
    overall_score: float
    explanations: Dict[str, str]


class ThoughtStep(BaseModel):
    """思考步骤模型"""
    step_type: str  # analysis, retrieval, generation, evaluation
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class Comparison(BaseModel):
    """论文对比分析模型"""
    paper: Paper
    aspects: Dict[str, str]  # novelty, methodology, results
    similarity_score: float


class ReviewScores(BaseModel):
    """评审评分模型"""
    overall: float  # 0-10
    novelty: float  # 0-10
    technical_quality: float  # 0-10
    clarity: float  # 0-10
    confidence: float  # 0-5


class PaperAnalysis(BaseModel):
    """论文分析结果模型"""
    structure: DocumentStructure
    contributions: List[str] = []
    methodology: List[str] = []
    experiments: List[str] = []
    results: List[str] = []
    limitations: List[str] = []


class StructuredReview(BaseModel):
    """结构化评审模型"""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    questions: List[str]
    scores: ReviewScores


class Score(BaseModel):
    """评分模型"""
    value: float
    explanation: str
    confidence: float = 1.0


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    message: str
    context: Optional[str] = None
    suggestion: Optional[str] = None


class HealthCheck(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: datetime
    checks: Dict[str, bool]
    version: str = "1.0.0"


class StreamChunk(BaseModel):
    """流式响应数据块模型"""
    object: str = "chat.completion.chunk"
    choices: List[Dict[str, Any]]


class AgentConfig(BaseModel):
    """智能体配置模型"""
    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 600
    enable_streaming: bool = True
