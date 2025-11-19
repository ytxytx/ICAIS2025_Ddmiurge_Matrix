# import aiohttp
import asyncio

from app import get_related_papers_from_keywords, generate_search_keywords #, format_references

if __name__ == "__main__":
    async def main():
        query = "人工智能在教育领域的应用"
        print(f"查询: {query}")

        # 生成搜索关键词
        keywords = await generate_search_keywords(query)
        print(f"生成的搜索关键词: {keywords}")

        # 获取相关论文
        related_papers = await get_related_papers_from_keywords(keywords)
        print(related_papers)
        # print(f"找到的相关论文数量: {len(related_papers)}")
        # for paper in related_papers:
        #     print(f"- {paper['title']} ({paper['url']})")
        
        # 格式化参考文献
        # formatted_references = format_references(related_papers)
        # print("\n格式化的参考文献:")
        # for ref in formatted_references:
        #     print(ref)
    asyncio.run(main())