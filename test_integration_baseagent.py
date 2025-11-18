#!/usr/bin/env python3
"""
智能体基类集成测试模块
进行真实的HTTP收发包测试，验证智能体基类与API服务的实际交互
"""

import asyncio
import json
import logging
import sys
import os
import base64
from datetime import datetime
from typing import AsyncGenerator, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.base import AgentBase
from src.models import AgentConfig, ThoughtStep


class IntegrationTestAgent(AgentBase):
    """集成测试用的具体智能体实现"""
    
    async def execute(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """测试执行方法 - 使用真实的API调用"""
        yield self._format_thought(f"开始处理查询: {query}")
        
        # 使用真实的思考循环
        async for thought in self._thinking_loop():
            yield thought
        
        # 使用真实的LLM调用
        prompt = f"""
        请基于以下查询提供简要回答：
        
        查询：{query}
        
        要求：
        1. 提供简洁明了的回答
        2. 专注于核心要点
        3. 避免复杂分析
        """
        
        # 测试流式LLM调用
        async for chunk in self._stream_llm(prompt):
            yield chunk
        
        yield self._format_content(f"\n\n✅ 查询 '{query}' 处理完成")


class IntegrationTestAgentBase:
    """智能体基类集成测试套件"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.agent_config = AgentConfig(
            name="integration_test_agent",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=500,
            timeout=30,
            enable_streaming=True
        )
        
        # 创建测试智能体
        self.agent = IntegrationTestAgent(self.agent_config)
    
    async def test_real_llm_call(self):
        """测试真实的LLM调用"""
        print("\n🤖 测试真实LLM调用...")
        
        try:
            # 使用真实的API调用
            prompt = "请简要介绍一下人工智能的发展历史"
            response = await self.agent._call_llm(prompt, max_tokens=200)
            
            # 验证响应格式
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"✅ 真实LLM调用成功，响应长度: {len(response)} 字符")
            print(f"响应内容: {response[:100]}...")
            
        except Exception as e:
            print(f"❌ 真实LLM调用失败: {e}")
            # 检查是否是API密钥问题
            if "API key" in str(e) or "authentication" in str(e):
                print("⚠️ 可能是API密钥配置问题，请检查环境变量")
            raise
    
    async def test_real_stream_llm(self):
        """测试真实的流式LLM调用"""
        print("\n🌊 测试真实流式LLM调用...")
        
        try:
            # 使用真实的流式API调用
            prompt = "请用流式方式介绍机器学习的基本概念"
            responses = []
            
            async for chunk in self.agent._stream_llm(prompt, max_tokens=300):
                responses.append(chunk)
                # 验证每个chunk的格式
                assert "data: " in chunk
                assert isinstance(chunk, str)
            
            # 验证响应数量
            assert len(responses) > 0
            print(f"✅ 真实流式LLM调用成功，收到 {len(responses)} 个chunk")
            
            # 打印部分响应内容
            if responses:
                content = "".join([json.loads(chunk.replace("data: ", ""))["choices"][0]["delta"].get("content", "") 
                                 for chunk in responses if "content" in chunk])
                print(f"流式响应内容: {content[:100]}...")
                
        except Exception as e:
            print(f"❌ 真实流式LLM调用失败: {e}")
            raise
    
    async def test_real_embedding(self):
        """测试真实的嵌入向量获取"""
        print("\n🔤 测试真实嵌入向量获取...")
        
        try:
            # 使用真实的嵌入API调用
            text = "这是一个测试文本，用于验证嵌入向量功能"
            embedding = await self.agent._get_embedding(text)
            
            # 验证嵌入向量格式
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
            
            print(f"✅ 真实嵌入向量获取成功，向量维度: {len(embedding)}")
            print(f"嵌入向量示例: {embedding[:5]}...")
            
        except Exception as e:
            print(f"❌ 真实嵌入向量获取失败: {e}")
            # 检查是否是API密钥问题
            if "API key" in str(e) or "authentication" in str(e):
                print("⚠️ 可能是嵌入API密钥配置问题，请检查环境变量")
            raise
    
    async def test_complete_agent_execution(self):
        """测试完整的智能体执行流程"""
        print("\n🚀 测试完整智能体执行流程...")
        
        try:
            query = "请帮我分析一下深度学习在计算机视觉中的应用"
            responses = []
            
            async for response in self.agent.execute(query):
                responses.append(response)
                # 验证响应格式
                assert "data: " in response
                assert isinstance(response, str)
            
            # 验证响应数量
            assert len(responses) > 0
            print(f"✅ 完整智能体执行成功，生成 {len(responses)} 个响应chunk")
            
            # 分析响应内容
            thought_count = sum(1 for r in responses if "🤔" in r)
            content_count = sum(1 for r in responses if "🤔" not in r and "[DONE]" not in r)
            
            print(f"思考步骤: {thought_count} 个")
            print(f"内容chunk: {content_count} 个")
            
        except Exception as e:
            print(f"❌ 完整智能体执行失败: {e}")
            raise
    
    async def test_error_handling(self):
        """测试错误处理机制"""
        print("\n🔄 测试错误处理机制...")
        
        try:
            # 测试空文本的嵌入向量获取
            empty_embedding = await self.agent._get_embedding("")
            assert empty_embedding == []
            print("✅ 空文本嵌入向量处理正确")
            
        except Exception as e:
            print(f"❌ 错误处理测试失败: {e}")
            raise
    
    async def test_performance_metrics(self):
        """测试性能指标"""
        print("\n⏱️ 测试性能指标...")
        
        try:
            start_time = datetime.now()
            
            # 执行一个简单的LLM调用
            await self.agent._call_llm("测试性能", max_tokens=50)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ API调用耗时: {duration:.2f} 秒")
            assert duration < 10, "API调用超时"
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            raise


async def run_integration_tests():
    """运行所有集成测试"""
    print("=" * 70)
    print("🧪 开始智能体基类集成测试 - 真实HTTP收发包测试")
    print("=" * 70)
    
    test_suite = IntegrationTestAgentBase()
    test_suite.setup_method()  # 初始化测试套件
    
    # 运行集成测试
    await test_suite.test_real_embedding()
    await test_suite.test_real_llm_call()
    await test_suite.test_real_stream_llm()
    await test_suite.test_complete_agent_execution()
    await test_suite.test_error_handling()
    await test_suite.test_performance_metrics()
    
    print("=" * 70)
    print("🎉 所有集成测试通过！智能体基类与API服务交互正常")
    print("=" * 70)


def check_environment():
    """检查环境配置"""
    print("\n🔍 检查环境配置...")
    
    try:
        from src.config import config
        
        # 检查必要的环境变量
        required_vars = [
            'SCI_MODEL_BASE_URL',
            'SCI_EMBEDDING_BASE_URL', 
            'SCI_MODEL_API_KEY',
            'SCI_EMBEDDING_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = getattr(config, var, None)
            if not value or value.startswith('your-'):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ 缺少必要的环境变量: {missing_vars}")
            print("请设置以下环境变量:")
            for var in missing_vars:
                print(f"  - {var}")
            return False
        
        print("✅ 环境配置检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 环境配置检查失败: {e}")
        return False


if __name__ == "__main__":
    # 检查环境配置
    env_ok = check_environment()
    
    if env_ok:
        # 运行集成测试
        asyncio.run(run_integration_tests())
    else:
        print("\n⚠️ 环境配置不完整，跳过集成测试")
        print("请先设置必要的API密钥和环境变量")
        sys.exit(1)
