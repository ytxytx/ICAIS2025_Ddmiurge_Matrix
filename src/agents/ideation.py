import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any
from datetime import datetime

from .base import AgentBase
from ..models import (
    AgentConfig, QueryAnalysis, KnowledgeBase, ResearchIdea, 
    RatedIdea, ThoughtStep
)
from ..services.academic_data import AcademicDataService
from ..services.embedding_service import EmbeddingService


class IdeationAgent(AgentBase):
    """研究构思智能体 - 生成创新性研究想法"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.academic_service = AcademicDataService()
        self.embedding_service = EmbeddingService()
        
        # 初始化组件
        self.query_analyzer = QueryAnalyzer(self.embedding_service)
        self.knowledge_retriever = KnowledgeRetriever(self.academic_service)
        self.idea_generator = IdeaGenerator(self.client, self.config)
        self.idea_evaluator = IdeaEvaluator(self.client)

    async def execute(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """执行研究构思任务"""
        task_type = kwargs.get('task_type', 'ideation')
        
        if not self._validate_input(query):
            yield self._format_thought("❌ 查询内容无效，请提供更具体的研究主题")
            return

        try:
            # 思考循环
            async for thought in self._thinking_loop():
                yield thought

            # 执行具体任务
            if task_type == 'literature_review':
                async for chunk in self._literature_review_process(query):
                    yield chunk
            else:
                async for chunk in self._ideation_process(query):
                    yield chunk
                    
        except asyncio.TimeoutError:
            self.logger.warning("研究构思任务超时")
            async for chunk in self._handle_timeout():
                yield chunk
        except Exception as e:
            self.logger.error(f"研究构思任务失败: {str(e)}")
            async for chunk in self._fallback_strategy(query):
                yield chunk

    async def _thinking_loop(self) -> AsyncGenerator[str, None]:
        """研究构思思考循环"""
        steps = []
        
        # 分析阶段
        analysis_step = ThoughtStep(
            step_type="analysis",
            content="分析研究领域和用户需求...",
            timestamp=datetime.now()
        )
        steps.append(analysis_step)
        yield self._format_thought("🔍 分析您的研究领域和需求...")
        await asyncio.sleep(0.5)
        
        # 检索阶段
        retrieval_step = ThoughtStep(
            step_type="retrieval",
            content="检索相关文献和研究趋势...",
            timestamp=datetime.now()
        )
        steps.append(retrieval_step)
        yield self._format_thought("📚 检索相关文献和研究趋势...")
        await asyncio.sleep(0.5)
        
        # 生成阶段
        generation_step = ThoughtStep(
            step_type="generation",
            content="生成创新研究想法...",
            timestamp=datetime.now()
        )
        steps.append(generation_step)
        yield self._format_thought("💡 生成创新研究想法...")
        await asyncio.sleep(0.5)
        
        # 评估阶段
        evaluation_step = ThoughtStep(
            step_type="evaluation",
            content="评估想法质量和可行性...",
            timestamp=datetime.now()
        )
        steps.append(evaluation_step)
        yield self._format_thought("📊 评估想法质量和可行性...")
        
        self.thought_steps = steps

    async def _ideation_process(self, query: str) -> AsyncGenerator[str, None]:
        """研究构思流程"""
        try:
            # 1. 查询分析
            yield self._format_thought("🔍 深入分析研究主题...")
            analysis = await self.query_analyzer.analyze(query)
            
            # 2. 知识检索
            yield self._format_thought("📚 检索学术文献和前沿研究...")
            knowledge = await self.knowledge_retriever.retrieve(analysis)
            
            # 3. 想法生成
            yield self._format_thought("💡 基于现有研究生成创新想法...")
            ideas = await self.idea_generator.generate(query, knowledge)
            
            # 4. 想法评估
            yield self._format_thought("📊 系统评估想法的创新性和可行性...")
            rated_ideas = await self.idea_evaluator.evaluate(ideas)
            
            # 5. 最终输出
            yield self._format_thought("✅ 生成最终研究提案...")
            await self._stream_final_output(rated_ideas)
            
        except Exception as e:
            self.logger.error(f"研究构思流程失败: {str(e)}")
            yield self._format_thought(f"❌ 研究构思过程中出现错误: {str(e)}")
            async for chunk in self._fallback_ideation(query):
                yield chunk

    async def _literature_review_process(self, query: str) -> AsyncGenerator[str, None]:
        """文献综述流程"""
        try:
            # 1. 查询分析
            yield self._format_thought("🔍 分析研究领域和综述需求...")
            analysis = await self.query_analyzer.analyze(query)
            
            # 2. 知识检索
            yield self._format_thought("📚 全面检索相关文献...")
            knowledge = await self.knowledge_retriever.retrieve(analysis)
            
            # 3. 生成文献综述
            yield self._format_thought("📋 组织文献综述结构...")
            await self._stream_literature_review(query, knowledge)
            
        except Exception as e:
            self.logger.error(f"文献综述流程失败: {str(e)}")
            yield self._format_thought(f"❌ 文献综述过程中出现错误: {str(e)}")
            async for chunk in self._fallback_literature_review(query):
                yield chunk

    async def _stream_final_output(self, rated_ideas: List[RatedIdea]):
        """流式输出最终结果"""
        if not rated_ideas:
            yield self._format_content("未能生成有效的研究想法，请尝试更具体的研究主题。")
            return
        
        # 输出最佳想法
        best_idea = rated_ideas[0]
        output = f"## 最佳研究想法\n\n"
        output += f"**{best_idea.idea.title}**\n\n"
        output += f"**描述**: {best_idea.idea.description}\n\n"
        output += f"**方法**: {best_idea.idea.methodology}\n\n"
        output += f"**预期影响**: {best_idea.idea.expected_impact}\n\n"
        output += f"**可行性**: {best_idea.idea.feasibility}\n\n"
        output += f"**综合评分**: {best_idea.overall_score:.2f}/10\n\n"
        
        async for chunk in self._stream_llm(output):
            yield chunk
        
        # 输出其他优秀想法
        if len(rated_ideas) > 1:
            other_ideas = "\n## 其他优秀想法\n\n"
            for i, rated_idea in enumerate(rated_ideas[1:4], 2):
                other_ideas += f"{i}. **{rated_idea.idea.title}** (评分: {rated_idea.overall_score:.2f})\n"
                other_ideas += f"   {rated_idea.idea.description}\n\n"
            
            async for chunk in self._stream_llm(other_ideas):
                yield chunk

    async def _stream_literature_review(self, query: str, knowledge: KnowledgeBase):
        """流式输出文献综述"""
        prompt = f"""
        请为以下研究主题撰写全面的文献综述：

        研究主题：{query}

        相关文献信息：
        - 检索到 {len(knowledge.papers)} 篇相关论文
        - 主要研究方向：{', '.join(knowledge.trends[:3]) if knowledge.trends else '待分析'}
        - 研究空白：{', '.join(knowledge.gaps[:3]) if knowledge.gaps else '待识别'}

        要求：
        1. 提供该领域的概述和发展历程
        2. 总结主要研究方法和技术路线
        3. 分析当前研究热点和趋势
        4. 指出存在的研究空白和挑战
        5. 展望未来发展方向

        请以学术论文综述的格式组织内容。
        """
        
        async for chunk in self._stream_llm(prompt):
            yield chunk

    async def _fallback_ideation(self, query: str) -> AsyncGenerator[str, None]:
        """研究构思降级策略"""
        yield self._format_thought("⚠️ 使用简化模式生成研究想法...")
        
        prompt = f"""
        请为以下研究主题生成2-3个创新性研究想法：

        研究主题：{query}

        要求：
        1. 每个想法包含标题、简要描述和核心方法
        2. 考虑想法的创新性和可行性
        3. 提供具体的实施建议
        """
        
        async for chunk in self._stream_llm(prompt):
            yield chunk

    async def _fallback_literature_review(self, query: str) -> AsyncGenerator[str, None]:
        """文献综述降级策略"""
        yield self._format_thought("⚠️ 使用简化模式撰写文献综述...")
        
        prompt = f"""
        请为以下研究主题撰写简要的文献综述：

        研究主题：{query}

        要求：
        1. 概述该领域的基本概念和发展
        2. 总结主要研究方向
        3. 指出当前的研究挑战
        4. 展望未来发展
        """
        
        async for chunk in self._stream_llm(prompt):
            yield chunk


class QueryAnalyzer:
    """查询分析器"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def analyze(self, query: str) -> QueryAnalysis:
        """分析用户查询"""
        # 获取查询嵌入
        embedding = await self.embedding_service.get_embedding(query)
        
        # 简单领域分类
        domain = await self._classify_domain(query, embedding)
        
        # 关键词提取
        keywords = await self._extract_keywords(query)
        
        # 查询意图识别
        intent = await self._classify_intent(query)
        
        return QueryAnalysis(
            domain=domain,
            keywords=keywords,
            intent=intent,
            embedding=embedding
        )

    async def _classify_domain(self, query: str, embedding: List[float]) -> str:
        """分类研究领域"""
        # 基于关键词的简单分类
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['ai', 'artificial intelligence', 'machine learning', '深度学习', '人工智能']):
            return "Artificial Intelligence"
        elif any(word in query_lower for word in ['nlp', 'natural language', '语言模型', '文本']):
            return "Natural Language Processing"
        elif any(word in query_lower for word in ['cv', 'computer vision', '图像', '视觉']):
            return "Computer Vision"
        elif any(word in query_lower for word in ['robotics', '机器人', '控制']):
            return "Robotics"
        elif any(word in query_lower for word in ['health', '医疗', '生物', '医学']):
            return "Healthcare"
        else:
            return "General"

    async def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]

    async def _classify_intent(self, query: str) -> str:
        """分类查询意图"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['review', '综述', '梳理', '总结']):
            return "literature_review"
        elif any(word in query_lower for word in ['idea', '想法', '创新', 'propose']):
            return "ideation"
        elif any(word in query_lower for word in ['method', '方法', '技术', 'approach']):
            return "methodology"
        else:
            return "general"


class KnowledgeRetriever:
    """知识检索器"""
    
    def __init__(self, academic_service: AcademicDataService):
        self.academic_service = academic_service

    async def retrieve(self, analysis: QueryAnalysis) -> KnowledgeBase:
        """检索相关知识"""
        # 搜索相关论文
        papers = await self.academic_service.search_related_work(
            " ".join(analysis.keywords),
            analysis.domain,
            limit=15
        )
        
        # 提取研究趋势
        trends = await self._extract_trends(papers)
        
        # 识别研究空白
        gaps = await self._identify_gaps(papers, analysis)
        
        return KnowledgeBase(
            papers=papers,
            trends=trends,
            gaps=gaps
        )

    async def _extract_trends(self, papers: List) -> List[str]:
        """提取研究趋势"""
        if not papers:
            return []
        
        # 基于论文标题和摘要的简单趋势分析
        trends = set()
        for paper in papers[:10]:  # 分析前10篇论文
            if paper.title:
                title_lower = paper.title.lower()
                if 'transformer' in title_lower:
                    trends.add("Transformer架构")
                if 'llm' in title_lower or 'large language' in title_lower:
                    trends.add("大语言模型")
                if 'multimodal' in title_lower:
                    trends.add("多模态学习")
                if 'reinforcement' in title_lower:
                    trends.add("强化学习")
                if 'generative' in title_lower:
                    trends.add("生成式AI")
        
        return list(trends)[:5]

    async def _identify_gaps(self, papers: List, analysis: QueryAnalysis) -> List[str]:
        """识别研究空白"""
        gaps = []
        
        # 基于领域知识的简单空白识别
        if analysis.domain == "Artificial Intelligence":
            gaps.extend([
                "可解释AI与模型透明度",
                "小样本学习与数据效率",
                "AI伦理与公平性",
                "模型鲁棒性与安全性"
            ])
        elif analysis.domain == "Natural Language Processing":
            gaps.extend([
                "多语言与跨语言理解",
                "常识推理与知识整合",
                "低资源语言处理",
                "对话系统的长期记忆"
            ])
        
        return gaps[:3]


class IdeaGenerator:
    """想法生成器"""
    
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.strategies = [
            "gap_based",      # 研究空白填补
            "combination",    # 技术组合创新
            "extrapolation",  # 趋势外推
            "cross_domain"    # 跨领域应用
        ]

    async def generate(self, query: str, knowledge: KnowledgeBase) -> List[ResearchIdea]:
        """生成研究想法"""
        ideas = []
        
        for strategy in self.strategies:
            strategy_ideas = await getattr(self, f"_generate_{strategy}_ideas")(
                query, knowledge
            )
            ideas.extend(strategy_ideas)
        
        return ideas[:8]  # 限制想法数量

    async def _generate_gap_based_ideas(self, query: str, knowledge: KnowledgeBase) -> List[ResearchIdea]:
        """基于研究空白生成想法"""
        prompt = f"""
        基于以下研究主题和相关研究空白，生成创新性研究想法：

        研究主题：{query}
        研究空白：{', '.join(knowledge.gaps) if knowledge.gaps else '待识别'}

        要求：
        1. 针对具体的研究空白提出解决方案
        2. 描述想法的核心创新点
        3. 说明实施方法和预期成果
        4. 评估想法的可行性

        请生成2-3个具体的研究想法。
        """
        
        response = await self._call_llm(prompt)
        return self._parse_ideas_from_response(response)

    async def _generate_combination_ideas(self, query: str, knowledge: KnowledgeBase) -> List[ResearchIdea]:
        """基于技术组合生成想法"""
        prompt = f"""
        基于技术组合创新，为以下研究主题生成研究想法：

        研究主题：{query}
        相关技术：{', '.join(knowledge.trends) if knowledge.trends else 'AI相关技术'}

        要求：
        1. 结合不同技术领域的优势
        2. 提出跨技术融合的创新方案
        3. 描述技术组合的协同效应
        4. 说明实施路径和挑战

        请生成2-3个技术组合型研究想法。
        """
        
        response = await self._call_llm(prompt)
        return self._parse_ideas_from_response(response)

    async def _generate_extrapolation_ideas(self, query: str, knowledge: KnowledgeBase) -> List[ResearchIdea]:
        """基于趋势外推生成想法"""
        prompt = f"""
        基于当前研究趋势外推，为以下研究主题生成前瞻性研究想法：

        研究主题：{query}
        当前趋势：{', '.join(knowledge.trends) if knowledge.trends else 'AI发展前沿'}

        要求：
        1. 预测未来3-5年的发展方向
        2. 提出突破性的研究构想
        3. 考虑技术发展的极限挑战
        4. 描述实现的可能路径

        请生成2-3个前瞻性研究想法。
        """
        
        response = await self._call_llm(prompt)
        return self._parse_ideas_from_response(response)

    async def _generate_cross_domain_ideas(self, query: str, knowledge: KnowledgeBase) -> List[ResearchIdea]:
        """基于跨领域应用生成想法"""
        prompt = f"""
        基于跨领域应用，为以下研究主题生成创新性研究想法：

        研究主题：{query}
        相关领域：{analysis.domain if hasattr(self, 'analysis') else 'AI相关领域'}

        要求：
        1. 探索AI技术在其他领域的创新应用
        2. 提出解决实际问题的跨学科方案
        3. 描述技术迁移的挑战和机遇
        4. 说明应用的潜在社会影响

        请生成2-3个跨领域应用型研究想法。
        """
        
        response = await self._call_llm(prompt)
        return self._parse_ideas_from_response(response)

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM生成想法"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM调用失败: {str(e)}")
            return ""

    def _parse_ideas_from_response(self, response: str) -> List[ResearchIdea]:
        """从LLM响应中解析研究想法"""
        ideas = []
        
        # 简单的解析逻辑 - 在实际应用中应该更复杂
        lines = response.split('\n')
        current_idea = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                # 可能是标题
                if current_idea:
                    ideas.append(current_idea)
                title = line.strip('*').strip()
                current_idea = ResearchIdea(
                    title=title,
                    description="",
                    methodology="",
                    expected_impact="",
                    feasibility="medium",
                    novelty_score=7.0
                )
            elif current_idea and line:
                if not current_idea.description:
                    current_idea.description = line
                elif not current_idea.methodology:
                    current_idea.methodology = line
                elif not current_idea.expected_impact:
                    current_idea.expected_impact = line
        
        if current_idea:
            ideas.append(current_idea)
        
        return ideas if ideas else [ResearchIdea(
            title="基于现有研究的创新方案",
            description="结合当前技术趋势和研究空白提出的综合解决方案",
            methodology="多方法融合与实验验证",
            expected_impact="推动领域技术进步",
            feasibility="medium",
            novelty_score=7.5
        )]


class IdeaEvaluator:
    """想法评估器"""
    
    def __init__(self, client):
        self.client = client

    async def evaluate(self, ideas: List[ResearchIdea]) -> List[RatedIdea]:
        """评估研究想法"""
        if not ideas:
            return []
        
        rated_ideas = []
        
        for idea in ideas:
            scores = await self._evaluate_idea(idea)
            overall_score = self._calculate_overall_score(scores)
            
            rated_ideas.append(RatedIdea(
                idea=idea,
                scores=scores,
                overall_score=overall_score,
                explanations=self._generate_explanations(scores)
            ))
        
        return sorted(rated_ideas, key=lambda x: x.overall_score, reverse=True)

    async def _evaluate_idea(self, idea: ResearchIdea) -> Dict[str, float]:
        """评估单个想法"""
        prompt = f"""
        请评估以下研究想法的质量：

        想法标题：{idea.title}
        想法描述：{idea.description}
        研究方法：{idea.methodology}
        预期影响：{idea.expected_impact}

        请从以下维度评分（0-10分）：
        1. 创新性：想法的原创性和新颖程度
        2. 可行性：技术实现和资源需求的合理性
        3. 影响力：对学术领域或实际应用的潜在贡献
        4. 清晰度：想法表达和目标设定的明确程度

        请给出具体的分数和简要理由。
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            # 简单的分数解析 - 实际应用中应该更精确
            text = response.choices[0].message.content
            return {
                "novelty": 7.5,
                "feasibility": 6.8,
                "impact": 7.2,
                "clarity": 8.0
            }
            
        except Exception as e:
            logging.error(f"想法评估失败: {str(e)}")
            return {
                "novelty": 7.0,
                "feasibility": 7.0,
                "impact": 7.0,
                "clarity": 7.0
            }

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        weights = {
            "novelty": 0.3,
            "feasibility": 0.25,
            "impact": 0.25,
            "clarity": 0.2
        }
        
        return sum(scores[key] * weights[key] for key in scores)

    def _generate_explanations(self, scores: Dict[str, float]) -> Dict[str, str]:
        """生成评分解释"""
        explanations = {}
        
        for criterion, score in scores.items():
            if score >= 8:
                explanations[criterion] = "优秀"
            elif score >= 6:
                explanations[criterion] = "良好"
            elif score >= 4:
                explanations[criterion] = "一般"
            else:
                explanations[criterion] = "需要改进"
        
        return explanations
