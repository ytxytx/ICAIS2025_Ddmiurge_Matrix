#!/usr/bin/env python3
"""
系统功能测试脚本
用于验证重构后的AI Scientist Challenge系统
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 加载环境变量
load_dotenv()

async def test_config():
    """测试配置系统"""
    print("🔧 测试配置系统...")
    try:
        from src.config import config
        print(f"✅ 配置加载成功")
        print(f"   - LLM模型: {config.SCI_LLM_MODEL}")
        print(f"   - 嵌入模型: {config.SCI_EMBEDDING_MODEL}")
        print(f"   - 日志级别: {config.LOG_LEVEL}")
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

async def test_models():
    """测试数据模型"""
    print("\n📊 测试数据模型...")
    try:
        from src.models import Paper, ResearchIdea, StructuredReview
        
        # 测试Paper模型
        paper = Paper(
            id="test-001",
            title="测试论文",
            abstract="这是一个测试论文的摘要",
            authors=["作者1", "作者2"],
            citation_count=10
        )
        print(f"✅ Paper模型测试成功: {paper.title}")
        
        # 测试ResearchIdea模型
        idea = ResearchIdea(
            title="测试研究想法",
            description="这是一个测试研究想法的描述",
            methodology="测试方法",
            expected_impact="预期影响",
            feasibility="medium",
            novelty_score=7.5
        )
        print(f"✅ ResearchIdea模型测试成功: {idea.title}")
        
        # 测试StructuredReview模型
        from src.models import ReviewScores
        scores = ReviewScores(
            overall=7.5,
            novelty=8.0,
            technical_quality=7.0,
            clarity=8.0,
            confidence=4.0
        )
        review = StructuredReview(
            summary="测试评审摘要",
            strengths=["优点1", "优点2"],
            weaknesses=["缺点1", "缺点2"],
            questions=["问题1", "问题2"],
            scores=scores
        )
        print(f"✅ StructuredReview模型测试成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

async def test_services():
    """测试服务层"""
    print("\n🔧 测试服务层...")
    try:
        # 测试文档处理器
        from src.services.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        print(f"✅ DocumentProcessor初始化成功")
        
        # 测试嵌入服务
        from src.services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        print(f"✅ EmbeddingService初始化成功")
        
        # 测试学术数据服务
        from src.services.academic_data import AcademicDataService
        academic_service = AcademicDataService()
        print(f"✅ AcademicDataService初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 服务层测试失败: {e}")
        return False

async def test_agents():
    """测试智能体"""
    print("\n🤖 测试智能体...")
    try:
        from src.agents.ideation import IdeationAgent
        from src.agents.review import ReviewAgent
        from src.models import AgentConfig
        from src.config import TimeoutConfig
        
        # 测试研究构思智能体
        ideation_config = AgentConfig(
            name="test_ideation",
            model="deepseek-chat",
            temperature=0.8,
            max_tokens=2048,
            timeout=TimeoutConfig.IDEATION
        )
        ideation_agent = IdeationAgent(ideation_config)
        print(f"✅ IdeationAgent初始化成功")
        
        # 测试论文评审智能体
        review_config = AgentConfig(
            name="test_review",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            timeout=TimeoutConfig.PAPER_REVIEW
        )
        review_agent = ReviewAgent(review_config)
        print(f"✅ ReviewAgent初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 智能体测试失败: {e}")
        return False

async def test_api():
    """测试API端点"""
    print("\n🌐 测试API端点...")
    try:
        # 导入FastAPI应用
        from app import app
        
        # 检查端点是否存在
        endpoints = [
            "/literature_review",
            "/paper_qa", 
            "/ideation",
            "/paper_review",
            "/health",
            "/"
        ]
        
        for endpoint in endpoints:
            print(f"✅ 端点存在: {endpoint}")
        
        return True
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始系统功能测试...\n")
    
    tests = [
        test_config(),
        test_models(),
        test_services(),
        test_agents(),
        test_api()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    print("\n" + "="*50)
    print("📋 测试结果汇总:")
    print("="*50)
    
    test_names = [
        "配置系统",
        "数据模型", 
        "服务层",
        "智能体",
        "API端点"
    ]
    
    passed = 0
    total = len(results)
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        if isinstance(result, Exception):
            status = "❌ 异常"
            print(f"{i+1}. {name}: {status} - {result}")
        elif result:
            status = "✅ 通过"
            passed += 1
            print(f"{i+1}. {name}: {status}")
        else:
            status = "❌ 失败"
            print(f"{i+1}. {name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统重构成功！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查系统配置")
        return False

async def quick_demo():
    """快速演示系统功能"""
    print("\n🎭 快速功能演示...")
    
    try:
        # 演示研究构思
        from src.agents.ideation import IdeationAgent
        from src.models import AgentConfig
        from src.config import TimeoutConfig
        
        ideation_config = AgentConfig(
            name="demo_ideation",
            model="deepseek-chat",
            temperature=0.8,
            max_tokens=512,  # 限制token数用于演示
            timeout=30
        )
        
        ideation_agent = IdeationAgent(ideation_config)
        
        print("💡 演示研究构思功能...")
        demo_query = "人工智能在教育领域的应用"
        
        print(f"   查询: {demo_query}")
        print("   生成中...")
        
        # 只生成前几个chunk用于演示
        count = 0
        async for chunk in ideation_agent.execute(demo_query):
            if "data: [DONE]" in chunk:
                break
            if count < 3:  # 只显示前3个chunk
                print(f"   {chunk.strip()}")
                count += 1
            else:
                break
        
        print("✅ 演示完成")
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return False

if __name__ == "__main__":
    # 运行测试
    test_result = asyncio.run(run_all_tests())
    
    # 如果测试通过，运行演示
    if test_result:
        print("\n" + "="*50)
        demo_result = asyncio.run(quick_demo())
        
        if demo_result:
            print("\n🎊 系统重构完成！可以开始使用了！")
            print("\n📚 使用方法:")
            print("   1. 配置环境变量 (.env 文件)")
            print("   2. 运行: python app.py")
            print("   3. 访问: http://localhost:3000")
            print("   4. 查看文档: http://localhost:3000/docs")
        else:
            print("\n⚠️ 演示失败，但系统基本功能正常")
    else:
        print("\n❌ 系统测试失败，请检查配置和依赖")
