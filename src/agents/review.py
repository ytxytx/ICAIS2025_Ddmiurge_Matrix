import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any
from datetime import datetime

from .base import AgentBase
from ..models import (
    AgentConfig, PaperAnalysis, Comparison, StructuredReview, 
    ReviewScores, ThoughtStep
)
from ..services.document_processor import DocumentProcessor
from ..services.academic_data import AcademicDataService
from ..services.embedding_service import EmbeddingService


class ReviewAgent(AgentBase):
    """论文评审智能体 - 对论文进行结构化评审"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.document_processor = DocumentProcessor()
        self.academic_service = AcademicDataService()
        self.embedding_service = EmbeddingService()
        
        # 初始化组件
        self.paper_analyzer = PaperAnalyzer(self.document_processor)
        self.comparator = RelatedWorkComparator(self.academic_service, self.embedding_service)
        self.review_generator = StructuredReviewGenerator(self.client)
        self.validator = ReviewValidator()

    async def execute(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """执行论文评审任务"""
        pdf_content = kwargs.get('pdf_content', '')
        
        if not self._validate_input(query) or not pdf_content:
            yield self._format_thought("❌ 查询内容或PDF文件无效")
            return

        try:
            # 思考循环
            async for thought in self._thinking_loop():
                yield thought

            # 执行评审流程
            async for chunk in self._review_process(query, pdf_content):
                yield chunk
                    
        except asyncio.TimeoutError:
            self.logger.warning("论文评审任务超时")
            async for chunk in self._handle_timeout():
                yield chunk
        except Exception as e:
            self.logger.error(f"论文评审任务失败: {str(e)}")
            async for chunk in self._fallback_strategy(query, pdf_content):
                yield chunk

    async def _thinking_loop(self) -> AsyncGenerator[str, None]:
        """论文评审思考循环"""
        steps = []
        
        # 文档解析阶段
        analysis_step = ThoughtStep(
            step_type="analysis",
            content="解析论文内容和结构...",
            timestamp=datetime.now()
        )
        steps.append(analysis_step)
        yield self._format_thought("📄 解析论文内容和结构...")
        await asyncio.sleep(0.5)
        
        # 相关研究对比阶段
        comparison_step = ThoughtStep(
            step_type="comparison",
            content="查找相关研究进行对比分析...",
            timestamp=datetime.now()
        )
        steps.append(comparison_step)
        yield self._format_thought("🔬 查找相关研究进行对比分析...")
        await asyncio.sleep(0.5)
        
        # 评审生成阶段
        review_step = ThoughtStep(
            step_type="review",
            content="生成结构化评审意见...",
            timestamp=datetime.now()
        )
        steps.append(review_step)
        yield self._format_thought("📋 生成结构化评审意见...")
        await asyncio.sleep(0.5)
        
        # 验证阶段
        validation_step = ThoughtStep(
            step_type="validation",
            content="验证评审完整性和质量...",
            timestamp=datetime.now()
        )
        steps.append(validation_step)
        yield self._format_thought("✅ 验证评审完整性和质量...")
        
        self.thought_steps = steps

    async def _review_process(self, query: str, pdf_content: str) -> AsyncGenerator[str, None]:
        """论文评审流程"""
        try:
            # 1. 论文分析
            yield self._format_thought("📄 深度解析论文内容...")
            paper_analysis = await self.paper_analyzer.analyze(pdf_content)
            
            # 2. 相关研究对比
            yield self._format_thought("🔬 检索和对比相关研究工作...")
            comparisons = await self.comparator.compare(paper_analysis)
            
            # 3. 生成结构化评审
            yield self._format_thought("📋 生成详细的结构化评审...")
            review = await self.review_generator.generate(paper_analysis, comparisons, query)
            
            # 4. 验证评审质量
            yield self._format_thought("✅ 验证评审内容的完整性...")
            if await self.validator.validate(review):
                await self._stream_structured_review(review)
            else:
                yield self._format_thought("⚠️ 评审内容不完整，使用备选方案...")
                await self._fallback_review(paper_analysis)
            
        except Exception as e:
            self.logger.error(f"论文评审流程失败: {str(e)}")
            yield self._format_thought(f"❌ 论文评审过程中出现错误: {str(e)}")
            async for chunk in self._fallback_basic_review(pdf_content):
                yield chunk

    async def _stream_structured_review(self, review: StructuredReview):
        """流式输出结构化评审"""
        # 输出摘要
        summary_output = f"## 论文摘要\n\n{review.summary}\n\n"
        async for chunk in self._stream_llm(summary_output):
            yield chunk
        
        # 输出优点
        strengths_output = f"## 论文优点\n\n" + "\n".join([f"- {strength}" for strength in review.strengths]) + "\n\n"
        async for chunk in self._stream_llm(strengths_output):
            yield chunk
        
        # 输出缺点
        weaknesses_output = f"## 论文不足\n\n" + "\n".join([f"- {weakness}" for weakness in review.weaknesses]) + "\n\n"
        async for chunk in self._stream_llm(weaknesses_output):
            yield chunk
        
        # 输出问题
        questions_output = f"## 作者问题\n\n" + "\n".join([f"- {question}" for question in review.questions]) + "\n\n"
        async for chunk in self._stream_llm(questions_output):
            yield chunk
        
        # 输出评分
        scores_output = f"## 评分结果\n\n"
        scores_output += f"- **总体评分**: {review.scores.overall:.1f}/10\n"
        scores_output += f"- **创新性**: {review.scores.novelty:.1f}/10\n"
        scores_output += f"- **技术质量**: {review.scores.technical_quality:.1f}/10\n"
        scores_output += f"- **清晰度**: {review.scores.clarity:.1f}/10\n"
        scores_output += f"- **评审信心**: {review.scores.confidence:.1f}/5\n\n"
        
        async for chunk in self._stream_llm(scores_output):
            yield chunk

    async def _fallback_review(self, paper_analysis: PaperAnalysis):
        """评审降级策略"""
        yield self._format_thought("⚠️ 使用简化评审模式...")
        
        prompt = f"""
        请基于以下论文分析结果提供简要评审：

        论文标题：{paper_analysis.structure.title or '未识别'}
        论文摘要：{paper_analysis.structure.abstract or '未提取'}
        主要贡献：{', '.join(paper_analysis.contributions) if paper_analysis.contributions else '待分析'}

        要求：
        1. 简要总结论文内容
        2. 指出主要优点和不足
        3. 提出2-3个关键问题
        4. 给出总体评价
        """
        
        async for chunk in self._stream_llm(prompt):
            yield chunk

    async def _fallback_basic_review(self, pdf_content: str):
        """基础评审降级策略"""
        yield self._format_thought("⚠️ 使用基础评审模式...")
        
        # 提取文本
        text = self.document_processor.extract_text(pdf_content)
        if not text:
            yield self._format_content("无法从PDF中提取文本内容")
            return
        
        prompt = f"""
        请对以下论文内容进行简要评审：

        论文内容：
        {text[:4000]}  # 限制文本长度

        要求：
        1. 总结论文核心内容
        2. 评价论文质量
        3. 提出改进建议
        """
        
        async for chunk in self._stream_llm(prompt):
            yield chunk


class PaperAnalyzer:
    """论文分析器"""
    
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor

    async def analyze(self, pdf_content: str) -> PaperAnalysis:
        """分析论文内容"""
        # 提取文本
        text = self.document_processor.extract_text(pdf_content)
        
        # 分块处理
        chunks = self.document_processor.chunk_document(text)
        
        # 识别结构
        structure = self.document_processor.identify_structure(chunks)
        
        # 提取关键元素
        elements = self.document_processor.extract_key_elements(structure)
        
        return PaperAnalysis(
            structure=structure,
            contributions=elements.contributions,
            methodology=elements.methods,
            experiments=elements.datasets + elements.metrics,
            results=elements.findings,
            limitations=elements.limitations
        )


class RelatedWorkComparator:
    """相关工作比较器"""
    
    def __init__(self, academic_service: AcademicDataService, embedding_service: EmbeddingService):
        self.academic_service = academic_service
        self.embedding_service = embedding_service

    async def compare(self, paper_analysis: PaperAnalysis) -> List[Comparison]:
        """比较相关工作"""
        if not paper_analysis.contributions:
            return []
        
        # 基于论文贡献检索相关论文
        query = " ".join(paper_analysis.contributions[:3])
        similar_papers = await self.academic_service.search_related_work(
            query, 
            "AI",  # 默认领域
            limit=8
        )
        
        comparisons = []
        for paper in similar_papers[:5]:  # 比较前5篇相关论文
            comparison = await self._compare_single_paper(paper_analysis, paper)
            comparisons.append(comparison)
        
        return sorted(comparisons, key=lambda x: x.similarity_score, reverse=True)

    async def _compare_single_paper(self, target: PaperAnalysis, other) -> Comparison:
        """比较单篇论文"""
        aspects = {}
        
        # 创新性比较
        aspects["novelty"] = await self._compare_novelty(target, other)
        
        # 方法比较
        aspects["methodology"] = await self._compare_methodology(target, other)
        
        # 结果比较
        aspects["results"] = await self._compare_results(target, other)
        
        # 计算相似度
        similarity_score = await self._calculate_similarity(target, other)
        
        return Comparison(
            paper=other,
            aspects=aspects,
            similarity_score=similarity_score
        )

    async def _compare_novelty(self, target: PaperAnalysis, other) -> str:
        """比较创新性"""
        # 简单的创新性比较
        if target.contributions and other.abstract:
            target_text = " ".join(target.contributions)
            other_text = other.abstract or other.title or ""
            
            # 使用嵌入计算相似度
            target_embedding = await self.embedding_service.get_embedding(target_text)
            other_embedding = await self.embedding_service.get_embedding(other_text)
            
            similarity = self.embedding_service.calculate_similarity(target_embedding, other_embedding)
            
            if similarity > 0.8:
                return "创新性较低，与现有工作高度相似"
            elif similarity > 0.5:
                return "有一定创新性，但核心思路相近"
            else:
                return "创新性较高，提出了新的思路"
        
        return "创新性待评估"

    async def _compare_methodology(self, target: PaperAnalysis, other) -> str:
        """比较方法"""
        if target.methodology and other.abstract:
            return "方法具有一定独特性"
        return "方法比较待深入分析"

    async def _compare_results(self, target: PaperAnalysis, other) -> str:
        """比较结果"""
        if target.results:
            return "实验结果需要更多对比分析"
        return "结果对比信息不足"

    async def _calculate_similarity(self, target: PaperAnalysis, other) -> float:
        """计算相似度"""
        if target.contributions and other.abstract:
            target_text = " ".join(target.contributions)
            other_text = other.abstract or other.title or ""
            
            target_embedding = await self.embedding_service.get_embedding(target_text)
            other_embedding = await self.embedding_service.get_embedding(other_text)
            
            return self.embedding_service.calculate_similarity(target_embedding, other_embedding)
        
        return 0.0


class StructuredReviewGenerator:
    """结构化评审生成器"""
    
    def __init__(self, client):
        self.client = client

    async def generate(self, paper_analysis: PaperAnalysis, comparisons: List[Comparison], query: str) -> StructuredReview:
        """生成结构化评审"""
        # 并行生成各评审部分
        section_tasks = {
            "summary": self._generate_summary(paper_analysis, query),
            "strengths": self._generate_strengths(paper_analysis, comparisons),
            "weaknesses": self._generate_weaknesses(paper_analysis, comparisons),
            "questions": self._generate_questions(paper_analysis, comparisons)
        }
        
        section_results = {}
        for section, task in section_tasks.items():
            section_results[section] = await task
        
        # 计算评分
        scores = await self._calculate_scores(paper_analysis, comparisons)
        
        return StructuredReview(
            **section_results,
            scores=scores
        )

    async def _generate_summary(self, paper_analysis: PaperAnalysis, query: str) -> str:
        """生成摘要"""
        prompt = f"""
        基于以下论文信息，生成评审摘要：

        论文标题：{paper_analysis.structure.title or '未识别'}
        论文摘要：{paper_analysis.structure.abstract or '未提取'}
        主要贡献：{', '.join(paper_analysis.contributions) if paper_analysis.contributions else '待分析'}
        评审要求：{query}

        要求：
        1. 简要总结论文核心内容
        2. 突出论文的主要贡献
        3. 保持客观中立的语气
        4. 控制在200字以内
        """
        
        response = await self._call_llm(prompt)
        return response or "论文摘要生成失败"

    async def _generate_strengths(self, paper_analysis: PaperAnalysis, comparisons: List[Comparison]) -> List[str]:
        """生成优点列表"""
        prompt = f"""
        基于以下论文信息，指出论文的优点：

        论文贡献：{', '.join(paper_analysis.contributions) if paper_analysis.contributions else '待分析'}
        研究方法：{', '.join(paper_analysis.methodology) if paper_analysis.methodology else '待分析'}
        实验结果：{', '.join(paper_analysis.results) if paper_analysis.results else '待分析'}

        相关比较：{len(comparisons)} 篇相关论文

        要求：
        1. 列出3-5个主要优点
        2. 基于论文具体内容
        3. 考虑创新性、技术质量、实验设计等方面
        4. 每个优点要具体明确
        """
        
        response = await self._call_llm(prompt)
        return self._parse_list_from_response(response)

    async def _generate_weaknesses(self, paper_analysis: PaperAnalysis, comparisons: List[Comparison]) -> List[str]:
        """生成缺点列表"""
        prompt = f"""
        基于以下论文信息，指出论文的不足和改进空间：

        论文局限性：{', '.join(paper_analysis.limitations) if paper_analysis.limitations else '待分析'}
        实验设计：{', '.join(paper_analysis.experiments) if paper_analysis.experiments else '待分析'}
        相关比较：{len(comparisons)} 篇相关论文

        要求：
        1. 列出3-5个主要不足
        2. 基于论文具体内容
        3. 考虑方法局限性、实验不足、分析深度等方面
        4. 每个不足要具体明确，并提供改进建议
        """
        
        response = await self._call_llm(prompt)
        return self._parse_list_from_response(response)

    async def _generate_questions(self, paper_analysis: PaperAnalysis, comparisons: List[Comparison]) -> List[str]:
        """生成问题列表"""
        prompt = f"""
        基于以下论文信息，提出需要作者澄清的问题：

        论文内容：{paper_analysis.structure.abstract or '未提取'}
        方法细节：{', '.join(paper_analysis.methodology) if paper_analysis.methodology else '待分析'}
        实验结果：{', '.join(paper_analysis.results) if paper_analysis.results else '待分析'}

        要求：
        1. 提出3-5个关键问题
        2. 问题要具体且有深度
        3. 关注方法细节、实验设置、结果解释等方面
        4. 帮助改进论文质量
        """
        
        response = await self._call_llm(prompt)
        return self._parse_list_from_response(response)

    async def _calculate_scores(self, paper_analysis: PaperAnalysis, comparisons: List[Comparison]) -> ReviewScores:
        """计算评分"""
        # 基于论文质量和比较结果的简单评分
        novelty = await self._score_novelty(paper_analysis, comparisons)
        technical_quality = await self._score_technical_quality(paper_analysis)
        clarity = await self._score_clarity(paper_analysis)
        
        # 计算总体评分
        overall = (novelty + technical_quality + clarity) / 3
        confidence = 4.0  # 默认置信度
        
        return ReviewScores(
            overall=overall,
            novelty=novelty,
            technical_quality=technical_quality,
            clarity=clarity,
            confidence=confidence
        )

    async def _score_novelty(self, paper_analysis: PaperAnalysis, comparisons: List[Comparison]) -> float:
        """评分创新性"""
        if not comparisons:
            return 7.0
        
        # 基于相似度评分创新性
        avg_similarity = sum(comp.similarity_score for comp in comparisons) / len(comparisons)
        
        if avg_similarity > 0.8:
            return 5.0  # 创新性较低
        elif avg_similarity > 0.6:
            return 6.5  # 有一定创新性
        elif avg_similarity > 0.4:
            return 7.5  # 创新性较好
        else:
            return 8.5  # 创新性很高

    async def _score_technical_quality(self, paper_analysis: PaperAnalysis) -> float:
        """评分技术质量"""
        score = 7.0  # 基础分
        
        # 基于方法完整性加分
        if paper_analysis.methodology and len(paper_analysis.methodology) >= 2:
            score += 0.5
        
        # 基于实验设计加分
        if paper_analysis.experiments and len(paper_analysis.experiments) >= 2:
            score += 0.5
        
        # 基于结果分析加分
        if paper_analysis.results and len(paper_analysis.results) >= 2:
            score += 0.5
        
        return min(score, 9.5)

    async def _score_clarity(self, paper_analysis: PaperAnalysis) -> float:
        """评分清晰度"""
        score = 7.0  # 基础分
        
        # 基于结构完整性加分
        if (paper_analysis.structure.abstract and 
            paper_analysis.structure.methodology and 
            paper_analysis.structure.results):
            score += 1.0
        
        # 基于贡献明确性加分
        if paper_analysis.contributions and len(paper_analysis.contributions) >= 2:
            score += 0.5
        
        return min(score, 9.0)

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM调用失败: {str(e)}")
            return ""

    def _parse_list_from_response(self, response: str) -> List[str]:
        """从响应中解析列表"""
        items = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 匹配列表项格式
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                items.append(line[2:].strip())
            elif line and len(line) > 10:  # 非空且有一定长度
                items.append(line)
        
        return items[:5]  # 限制数量


class ReviewValidator:
    """评审验证器"""
    
    async def validate(self, review: StructuredReview) -> bool:
        """验证评审完整性"""
        if not review.summary or len(review.summary.strip()) < 50:
            return False
        
        if not review.strengths or len(review.strengths) < 2:
            return False
        
        if not review.weaknesses or len(review.weaknesses) < 2:
            return False
        
        if not review.questions or len(review.questions) < 2:
            return False
        
        # 验证评分范围
        scores = review.scores
        if not (0 <= scores.overall <= 10 and
                0 <= scores.novelty <= 10 and
                0 <= scores.technical_quality <= 10 and
                0 <= scores.clarity <= 10 and
                0 <= scores.confidence <= 5):
            return False
        
        return True
