#!/usr/bin/env python3
"""
简化版研究构思智能体测试
测试重构后的ideation.py功能
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.ideation import IdeationAgent
from src.models import AgentConfig


async def test_simple_ideation():
    """测试简化版研究构思功能"""
    print("🧪 测试简化版研究构思智能体")
    print("=" * 50)
    
    # 创建智能体配置
    agent_config = AgentConfig(
        name="simple_ideation_test",
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=1000,
        timeout=30,
        enable_streaming=True
    )
    
    # 创建智能体
    agent = IdeationAgent(agent_config)
    
    # 测试研究构思
    print("\n🤖 测试研究构思功能...")
    query = "人工智能在教育领域的应用"
    
    print(f"查询: {query}")
    print("响应:")
    
    response_count = 0
    async for chunk in agent.execute(query):
        if "[DONE]" not in chunk:
            print(chunk, end="", flush=True)
            response_count += 1
    
    print(f"\n✅ 收到 {response_count} 个响应chunk")
    
    # 测试文献综述
    print("\n📚 测试文献综述功能...")
    query = "深度学习在医疗诊断中的应用"
    
    print(f"查询: {query}")
    print("响应:")
    
    response_count = 0
    async for chunk in agent.execute(query, task_type="literature_review"):
        if "[DONE]" not in chunk:
            print(chunk, end="", flush=True)
            response_count += 1
    
    print(f"\n✅ 收到 {response_count} 个响应chunk")
    
    # 测试错误处理
    print("\n🔄 测试错误处理...")
    
    # 测试空查询
    print("测试空查询:")
    async for chunk in agent.execute(""):
        if "[DONE]" not in chunk:
            print(chunk, end="", flush=True)
    
    # 测试过短查询
    print("\n测试过短查询:")
    async for chunk in agent.execute("AI"):
        if "[DONE]" not in chunk:
            print(chunk, end="", flush=True)
    
    print("\n" + "=" * 50)
    print("🎉 简化版研究构思智能体测试完成")


if __name__ == "__main__":
    asyncio.run(test_simple_ideation())
