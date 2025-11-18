import asyncio
import logging
from typing import AsyncGenerator
from datetime import datetime

from .base import AgentBase
from ..models import AgentConfig, ThoughtStep
from ..services.document_processor import DocumentProcessor


class ReviewAgent(AgentBase):
    """论文评审智能体 - 简化版本，保留PDF阅读功能"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.document_processor = DocumentProcessor()

    async def execute(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """执行论文评审任务 - 简化版本"""
        pdf_content = kwargs.get('pdf_content', '')
        
        if not self._validate_input(query) or not pdf_content:
            yield self._format_thought("❌ 查询内容或PDF文件无效")
            return

        try:
            # 简化的思考循环
            yield self._format_thought("📄 解析论文内容...")
            await asyncio.sleep(0.5)
            
            yield self._format_thought("🔍 分析论文质量...")
            await asyncio.sleep(0.5)
            
            yield self._format_thought("📋 生成评审意见...")
            await asyncio.sleep(0.5)

            # 执行简化的评审流程
            async for chunk in self._simple_review_process(query, pdf_content):
                yield chunk
                    
        except asyncio.TimeoutError:
            self.logger.warning("论文评审任务超时")
            async for chunk in self._handle_timeout():
                yield chunk
        except Exception as e:
            self.logger.error(f"论文评审任务失败: {str(e)}")
            async for chunk in self._fallback_strategy(query, pdf_content):
                yield chunk

    async def _simple_review_process(self, query: str, pdf_content: str) -> AsyncGenerator[str, None]:
        """简化的论文评审流程 - 只进行一次LLM调用"""
        # 提取PDF文本内容
        text = self.document_processor.extract_text(pdf_content)
        if not text:
            yield self._format_content("无法从PDF中提取文本内容")
            return
        
        # 限制文本长度，避免token超限
        text_preview = text[:3000]  # 限制为3000字符
        
        prompt = f"""
        请对以下论文进行结构化评审：

        评审要求：{query}

        论文内容：
        {text_preview}

        请按照以下结构提供评审意见：

        ## 论文摘要
        简要总结论文的核心内容和主要贡献

        ## 主要优点
        - 列出3-4个论文的主要优点
        - 关注创新性、技术质量、实验设计等方面

        ## 主要不足
        - 列出3-4个论文的主要不足和改进空间
        - 关注方法局限性、实验不足、分析深度等方面

        ## 关键问题
        - 提出3-4个需要作者澄清的关键问题
        - 关注方法细节、实验设置、结果解释等方面

        ## 总体评价
        - 给出总体评分（0-10分）
        - 简要说明评分理由
        - 提供改进建议

        请直接输出评审内容，不要添加其他说明。
        """
        
        # 使用流式LLM调用
        async for chunk in self._stream_llm(prompt):
            yield chunk
