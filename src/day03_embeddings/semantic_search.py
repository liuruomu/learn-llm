import os
import httpx
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化客户端 (保持 Day 1/2 的代理修复逻辑)
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url=os.getenv("ZHIPUAI_BASE_URL"),
    http_client=http_client
)

def get_embedding(text: str):
    """将字符串转换为向量"""
    # 注意：DeepSeek 目前可能不提供 embedding 模型，
    # 如果 DeepSeek 报错，可以临时使用 OpenAI 的 'text-embedding-3-small' 
    # 或者本地库。这里假设你使用支持 embedding 的接口。
    response = client.embeddings.create(
        model="embedding-3", # 如果用 DeepSeek 需确认其模型名
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- 实战演练 ---

# 1. 我们的“知识库”
knowledge_base = [
    "大熊猫喜欢吃竹子",
    "人工智能正在改变世界",
    "明天的天气预报说有雨",
    "如何制作一份美味的意大利面",
    "计算机科学是一门研究算法的学科"
]

# 2. 将知识库全部向量化
print("正在将知识库转换成向量...")
kb_embeddings = [get_embedding(text) for text in knowledge_base]

def search(query: str):
    print(f"\n用户提问: '{query}'")
    # 3. 将用户问题向量化
    query_embedding = get_embedding(query)
    
    # 4. 计算相似度并排序
    similarities = [cosine_similarity(query_embedding, kb_v) for kb_v in kb_embeddings]
    
    # 5. 找出得分最高的一个
    best_idx = np.argmax(similarities)
    print(f"🎯 最匹配的结果: '{knowledge_base[best_idx]}' (相似度得分: {similarities[best_idx]:.4f})")

if __name__ == "__main__":
    # 测试语义搜索（注意：即使没有关键词重合，也能搜到意思接近的）
    search("那只黑白相间的动物吃什么？") 
    search("帮我写个菜谱")
    search("写代码相关的")