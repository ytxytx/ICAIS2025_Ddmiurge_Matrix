import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI

from ..config import config, TimeoutConfig
from ..models import (
    ThoughtStep, AgentConfig, ErrorResponse, StreamChunk
)


class AgentBase(ABC):
    """智能体基类 - 提供统一的智能体接口和核心功能"""
    
    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.logger = logging.getLogger(f"agent.{agent_config.name}")
        
        # 初始化LLM客户端 - 使用全局config对象
        self.client = AsyncOpenAI(
            base_url=config.SCI_MODEL_BASE_URL,
            api_key=config.SCI_MODEL_API_KEY
        )
        
        # 初始化嵌入客户端 - 使用全局config对象
        self.embedding_client = AsyncOpenAI(
            base_url=config.SCI_EMBEDDING_BASE_URL,
            api_key=config.SCI_EMBEDDING_API_KEY
        )
        
        # 思考步骤记录
        self.thought_steps: List[ThoughtStep] = []
        
        # 会话状态
        self.session_start_time = datetime.now()
        self.is_streaming = agent_config.enable_streaming

    @abstractmethod
    async def execute(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """执行智能体任务 - 抽象方法，子类必须实现"""
        pass

    async def _thinking_loop(self) -> AsyncGenerator[str, None]:
        """思考循环 - 子类可以重写此方法实现自定义思考逻辑"""
        steps = []
        
        # 分析阶段
        analysis_step = ThoughtStep(
            step_type="analysis",
            content="分析用户查询和任务需求...",
            timestamp=datetime.now()
        )
        steps.append(analysis_step)
        yield self._format_thought(analysis_step.content)
        
        # 检索阶段
        retrieval_step = ThoughtStep(
            step_type="retrieval",
            content="检索相关知识和信息...",
            timestamp=datetime.now()
        )
        steps.append(retrieval_step)
        yield self._format_thought(retrieval_step.content)
        
        # 生成阶段
        generation_step = ThoughtStep(
            step_type="generation",
            content="生成解决方案...",
            timestamp=datetime.now()
        )
        steps.append(generation_step)
        yield self._format_thought(generation_step.content)
        
        self.thought_steps = steps

    async def _fallback_strategy(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """降级策略 - 当主要方法失败时使用"""
        self.logger.warning(f"使用降级策略处理查询: {query}")
        
        yield self._format_thought("⚠️ 系统遇到问题，使用简化模式处理...")
        
        # 简化提示词
        simple_prompt = f"""
        请基于以下查询提供简要回答：
        
        查询：{query}
        
        要求：
        1. 提供简洁明了的回答
        2. 专注于核心要点
        3. 避免复杂分析
        """
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": simple_prompt}],
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        yield self._format_content(delta_content)
                        
        except Exception as e:
            error_msg = f"降级策略也失败了: {str(e)}"
            self.logger.error(error_msg)
            yield self._format_thought(f"❌ {error_msg}")

    def _format_sse_data(self, content: str) -> str:
        """格式化SSE数据"""
        response_data = StreamChunk(
            choices=[{"delta": {"content": content}}]
        )
        return f"data: {response_data.json()}\n\n"

    def _format_thought(self, content: str) -> str:
        """格式化思考内容"""
        thought_data = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": f"\n\n🤔 {content}\n\n"
                }
            }]
        }
        return f"data: {json.dumps(thought_data)}\n\n"

    def _format_content(self, content: str) -> str:
        """格式化普通内容"""
        return self._format_sse_data(content)

    async def _handle_timeout(self) -> AsyncGenerator[str, None]:
        """超时处理"""
        self.logger.warning("任务执行超时")
        yield self._format_thought("⏰ 时间限制已到，输出当前最佳结果...")
        
        # 输出当前思考步骤的总结
        if self.thought_steps:
            summary = "基于当前分析，主要发现包括：\n"
            for step in self.thought_steps[-3:]:  # 取最后3个步骤
                summary += f"- {step.content}\n"
            yield self._format_content(summary)

    def _validate_input(self, query: str, **kwargs) -> bool:
        """输入验证"""
        if not query or not query.strip():
            self.logger.error("查询内容为空")
            return False
            
        if len(query.strip()) < 3:
            self.logger.error("查询内容过短")
            return False
            
        return True

    def _log_operation(self, operation: str, duration: float, success: bool):
        """记录操作日志"""
        self.logger.info(
            f"Agent operation completed",
            extra={
                "agent": self.config.name,
                "operation": operation,
                "duration": duration,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        try:
            response = await self.embedding_client.embeddings.create(
                model=config.SCI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"获取嵌入向量失败: {str(e)}")
            return []

    async def _call_llm(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """调用LLM模型"""
        try:
            model_to_use = model or self.config.model
            temperature = kwargs.get('temperature', self.config.temperature)
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
            
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM调用失败: {str(e)}")
            raise

    async def _stream_llm(self, prompt: str, model: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """流式调用LLM模型"""
        try:
            model_to_use = model or self.config.model
            temperature = kwargs.get('temperature', self.config.temperature)
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
            
            stream = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        yield self._format_content(delta_content)
                        
        except Exception as e:
            self.logger.error(f"流式LLM调用失败: {str(e)}")
            yield self._format_thought(f"❌ 模型调用失败: {str(e)}")

    def _calculate_session_duration(self) -> float:
        """计算会话持续时间"""
        return (datetime.now() - self.session_start_time).total_seconds()
