import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..config import config
from ..models import Paper, PaperDetails


class AcademicDataService:
    """学术数据检索服务 - 集成多个学术API"""
    
    def __init__(self):
        self.logger = logging.getLogger("service.academic_data")
        self.cache = {}  # 简单缓存实现
        self.rate_limiter = TokenBucket(capacity=10, refill_rate=1)  # 10请求/秒
        
        # API客户端配置
        self.api_clients = {
            "openalex": {
                "base_url": config.OPENALEX_BASE_URL,
                "headers": {"User-Agent": "AI-Scientist-Challenge/1.0"}
            },
            "semantic_scholar": {
                "base_url": config.SEMANTIC_SCHOLAR_BASE_URL,
                "headers": {"User-Agent": "AI-Scientist-Challenge/1.0"}
            },
            "arxiv": {
                "base_url": config.ARXIV_BASE_URL,
                "headers": {"User-Agent": "AI-Scientist-Challenge/1.0"}
            }
        }

    async def search_related_work(self, query: str, domain: str, limit: int = 20) -> List[Paper]:
        """搜索相关研究工作"""
        self.logger.info(f"搜索相关研究: {query} (领域: {domain})")
        
        try:
            # 构建搜索查询
            search_queries = self._build_search_queries(query, domain)
            
            # 并行搜索多个API
            search_tasks = [
                self._search_openalex(search_queries, limit),
                self._search_semantic_scholar(search_queries, limit),
                self._search_arxiv(search_queries, limit)
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # 整合结果
            all_papers = []
            for result in results:
                if isinstance(result, list):
                    all_papers.extend(result)
            
            # 去重和排序
            deduplicated_papers = self._deduplicate_papers(all_papers)
            ranked_papers = self._rank_by_relevance(deduplicated_papers, query)
            
            self.logger.info(f"找到 {len(ranked_papers)} 篇相关论文")
            return ranked_papers[:limit]
            
        except Exception as e:
            self.logger.error(f"搜索相关研究失败: {str(e)}")
            return []

    async def get_paper_details(self, paper_id: str, source: str = "semantic_scholar") -> Optional[PaperDetails]:
        """获取论文详细信息"""
        cache_key = f"{source}_{paper_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if source == "semantic_scholar":
                paper_details = await self._get_semantic_scholar_details(paper_id)
            elif source == "openalex":
                paper_details = await self._get_openalex_details(paper_id)
            else:
                self.logger.warning(f"不支持的论文源: {source}")
                return None
            
            if paper_details:
                self.cache[cache_key] = paper_details
                
            return paper_details
            
        except Exception as e:
            self.logger.error(f"获取论文详情失败: {str(e)}")
            return None

    async def find_similar_papers(self, embedding: List[float], limit: int = 10) -> List[Paper]:
        """基于嵌入向量查找相似论文"""
        # 这里可以集成向量数据库或语义搜索
        # 目前返回空列表，后续可以扩展
        self.logger.info(f"基于嵌入向量查找相似论文 (维度: {len(embedding)})")
        return []

    def _build_search_queries(self, query: str, domain: str) -> List[str]:
        """构建搜索查询"""
        queries = []
        
        # 基础查询
        queries.append(query)
        
        # 领域特定查询
        if domain.lower() in ["ai", "machine learning", "artificial intelligence"]:
            queries.append(f"{query} machine learning")
            queries.append(f"{query} artificial intelligence")
        elif domain.lower() in ["nlp", "natural language processing"]:
            queries.append(f"{query} natural language processing")
            queries.append(f"{query} transformer")
        elif domain.lower() in ["cv", "computer vision"]:
            queries.append(f"{query} computer vision")
            queries.append(f"{query} deep learning")
        
        # 添加时间限制（最近3年）
        current_year = datetime.now().year
        for i in range(len(queries)):
            queries[i] = f"{queries[i]} {current_year-3}-{current_year}"
        
        return queries

    async def _search_openalex(self, queries: List[str], limit: int) -> List[Paper]:
        """使用OpenAlex搜索论文"""
        papers = []
        
        try:
            async with httpx.AsyncClient() as client:
                for query in queries[:2]:  # 限制查询数量
                    await self.rate_limiter.acquire()
                    
                    params = {
                        "search": query,
                        "per-page": min(limit, 25),
                        "sort": "cited_by_count:desc"
                    }
                    
                    response = await client.get(
                        f"{self.api_clients['openalex']['base_url']}/works",
                        params=params,
                        headers=self.api_clients['openalex']['headers'],
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for work in data.get("results", []):
                            paper = self._parse_openalex_work(work)
                            if paper:
                                papers.append(paper)
                    
                    await asyncio.sleep(0.1)  # 避免请求过快
                    
        except Exception as e:
            self.logger.warning(f"OpenAlex搜索失败: {str(e)}")
        
        return papers

    async def _search_semantic_scholar(self, queries: List[str], limit: int) -> List[Paper]:
        """使用Semantic Scholar搜索论文"""
        papers = []
        
        try:
            async with httpx.AsyncClient() as client:
                for query in queries[:2]:
                    await self.rate_limiter.acquire()
                    
                    params = {
                        "query": query,
                        "limit": min(limit, 100),
                        "fields": "paperId,title,abstract,authors,year,venue,citationCount,url,doi"
                    }
                    
                    response = await client.get(
                        f"{self.api_clients['semantic_scholar']['base_url']}/paper/search",
                        params=params,
                        headers=self.api_clients['semantic_scholar']['headers'],
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for paper_data in data.get("data", []):
                            paper = self._parse_semantic_scholar_paper(paper_data)
                            if paper:
                                papers.append(paper)
                    
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            self.logger.warning(f"Semantic Scholar搜索失败: {str(e)}")
        
        return papers

    async def _search_arxiv(self, queries: List[str], limit: int) -> List[Paper]:
        """使用arXiv搜索论文"""
        papers = []
        
        try:
            async with httpx.AsyncClient() as client:
                for query in queries[:1]:  # arXiv限制较严格
                    await self.rate_limiter.acquire()
                    
                    params = {
                        "search_query": f"all:{query}",
                        "max_results": min(limit, 50),
                        "sortBy": "relevance",
                        "sortOrder": "descending"
                    }
                    
                    response = await client.get(
                        self.api_clients['arxiv']['base_url'],
                        params=params,
                        headers=self.api_clients['arxiv']['headers'],
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        # 解析arXiv的Atom格式响应
                        papers.extend(self._parse_arxiv_response(response.text))
                    
                    await asyncio.sleep(1.0)  # arXiv有严格的速率限制
                    
        except Exception as e:
            self.logger.warning(f"arXiv搜索失败: {str(e)}")
        
        return papers

    def _parse_openalex_work(self, work: Dict[str, Any]) -> Optional[Paper]:
        """解析OpenAlex工作数据"""
        try:
            return Paper(
                id=work.get("id", ""),
                title=work.get("title", ""),
                abstract=work.get("abstract", ""),
                authors=[author.get("author", {}).get("display_name", "") 
                        for author in work.get("authorships", [])],
                publication_date=work.get("publication_date", ""),
                venue=work.get("host_venue", {}).get("display_name", ""),
                citation_count=work.get("cited_by_count", 0),
                url=work.get("doi", work.get("id", "")),
                doi=work.get("doi", "")
            )
        except Exception as e:
            self.logger.warning(f"解析OpenAlex数据失败: {str(e)}")
            return None

    def _parse_semantic_scholar_paper(self, paper_data: Dict[str, Any]) -> Optional[Paper]:
        """解析Semantic Scholar论文数据"""
        try:
            return Paper(
                id=paper_data.get("paperId", ""),
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                authors=[author.get("name", "") for author in paper_data.get("authors", [])],
                publication_date=str(paper_data.get("year", "")),
                venue=paper_data.get("venue", ""),
                citation_count=paper_data.get("citationCount", 0),
                url=paper_data.get("url", ""),
                doi=paper_data.get("doi", "")
            )
        except Exception as e:
            self.logger.warning(f"解析Semantic Scholar数据失败: {str(e)}")
            return None

    def _parse_arxiv_response(self, response_text: str) -> List[Paper]:
        """解析arXiv响应（简化实现）"""
        # 这里应该实现完整的Atom格式解析
        # 目前返回空列表，避免解析错误
        return []

    async def _get_semantic_scholar_details(self, paper_id: str) -> Optional[PaperDetails]:
        """获取Semantic Scholar论文详情"""
        try:
            async with httpx.AsyncClient() as client:
                await self.rate_limiter.acquire()
                
                response = await client.get(
                    f"{self.api_clients['semantic_scholar']['base_url']}/paper/{paper_id}",
                    params={
                        "fields": "paperId,title,abstract,authors,year,venue,citationCount,url,doi,references,citations,topics"
                    },
                    headers=self.api_clients['semantic_scholar']['headers'],
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    paper = self._parse_semantic_scholar_paper(data)
                    if paper:
                        return PaperDetails(
                            paper=paper,
                            references=[],
                            citations=[],
                            keywords=[],
                            topics=data.get("topics", [])
                        )
                        
        except Exception as e:
            self.logger.warning(f"获取Semantic Scholar详情失败: {str(e)}")
        
        return None

    async def _get_openalex_details(self, work_id: str) -> Optional[PaperDetails]:
        """获取OpenAlex工作详情"""
        try:
            async with httpx.AsyncClient() as client:
                await self.rate_limiter.acquire()
                
                response = await client.get(
                    f"{self.api_clients['openalex']['base_url']}/works/{work_id}",
                    headers=self.api_clients['openalex']['headers'],
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    paper = self._parse_openalex_work(data)
                    if paper:
                        return PaperDetails(
                            paper=paper,
                            references=[],
                            citations=[],
                            keywords=[],
                            topics=[]
                        )
                        
        except Exception as e:
            self.logger.warning(f"获取OpenAlex详情失败: {str(e)}")
        
        return None

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """论文去重"""
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            if paper.id and paper.id not in seen_ids:
                seen_ids.add(paper.id)
                unique_papers.append(paper)
            elif paper.doi and paper.doi not in seen_ids:
                seen_ids.add(paper.doi)
                unique_papers.append(paper)
            elif paper.title and paper.title not in seen_ids:
                seen_ids.add(paper.title)
                unique_papers.append(paper)
        
        return unique_papers

    def _rank_by_relevance(self, papers: List[Paper], query: str) -> List[Paper]:
        """按相关性排序论文"""
        # 简单的基于引用数和标题匹配的排序
        def score_paper(paper: Paper) -> float:
            score = 0.0
            
            # 引用数权重
            if paper.citation_count:
                score += min(paper.citation_count / 100, 10.0)
            
            # 标题匹配权重
            if paper.title and query.lower() in paper.title.lower():
                score += 5.0
            
            # 摘要匹配权重
            if paper.abstract and query.lower() in paper.abstract.lower():
                score += 3.0
            
            return score
        
        return sorted(papers, key=score_paper, reverse=True)


class TokenBucket:
    """令牌桶限流器"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # 令牌/秒
        self.last_refill = datetime.now()

    async def acquire(self, tokens: int = 1):
        """获取令牌"""
        while self.tokens < tokens:
            await self._refill()
            await asyncio.sleep(0.1)
        
        self.tokens -= tokens

    async def _refill(self):
        """补充令牌"""
        now = datetime.now()
        time_passed = (now - self.last_refill).total_seconds()
        new_tokens = time_passed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
