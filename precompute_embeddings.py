import os
import json
import asyncio
from typing import List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# 从reference_ideas_str.py导入参考idea
from reference_ideas_str import reference_ideas

embedding_client = AsyncOpenAI(
    base_url=os.getenv("SCI_EMBEDDING_BASE_URL"),
    api_key=os.getenv("SCI_EMBEDDING_API_KEY")
)

async def get_embedding(text: str) -> List[float]:
    """获取文本的embedding向量"""
    try:
        response = await embedding_client.embeddings.create(
            model=os.getenv("SCI_EMBEDDING_MODEL"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        return []

async def precompute_all_embeddings():
    """预计算所有参考idea的embedding向量"""
    print(f"开始预计算 {len(reference_ideas)} 个参考idea的embedding向量...")
    
    embeddings_data = {}
    
    # 分批处理，避免一次性请求过多
    batch_size = 10
    for i in range(0, len(reference_ideas), batch_size):
        batch = reference_ideas[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(reference_ideas)-1)//batch_size + 1}: {i+1}-{min(i+batch_size, len(reference_ideas))}")
        
        # 为批次中的每个idea获取embedding
        tasks = [get_embedding(idea) for idea in batch]
        embeddings = await asyncio.gather(*tasks)
        
        # 将结果存储到字典中
        for j, idea in enumerate(batch):
            embeddings_data[idea] = embeddings[j]
        
        # 添加延迟以避免API限制
        await asyncio.sleep(1)
    
    # 保存到JSON文件
    output_file = "reference_ideas_embeddings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"完成！embedding向量已保存到 {output_file}")
    print(f"成功计算了 {len(embeddings_data)} 个embedding向量")
    
    # 检查是否有失败的embedding
    failed_count = sum(1 for emb in embeddings_data.values() if not emb)
    if failed_count > 0:
        print(f"警告：有 {failed_count} 个embedding计算失败")

if __name__ == "__main__":
    asyncio.run(precompute_all_embeddings())
