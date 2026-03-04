import os
import httpx
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- 1. 配置代理客户端 (解决之前的 SOCKS 代理问题) ---
http_client = httpx.Client(trust_env=False)

# --- 2. 初始化 API 客户端 ---
# 客户端 A: 智谱 AI (负责向量化)
embed_client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url=os.getenv("ZHIPUAI_BASE_URL"), # 通常是 https://open.bigmodel.cn/api/paas/v4/
    http_client=http_client
)

# 客户端 B: DeepSeek (负责大脑对话)
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
)

# --- 3. 初始化向量数据库 ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# get_or_create_collection 确保不会因为重复运行而报错
collection = chroma_client.get_or_create_collection(name="rag_bot_data")

def get_embedding(text: str):
    """调用智谱 AI 获取向量"""
    response = embed_client.embeddings.create(
        model="embedding-3", # 智谱最新的向量模型
        input=text
    )
    return response.data[0].embedding

def ingest_data(file_path: str):
    """读取文档，向量化并存入数据库"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 简单切片：按行切分（实际开发可以用更复杂的切分器）
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    print(f"正在向量化并导入数据，共 {len(lines)} 条...")
    for i, line in enumerate(lines):
        vector = get_embedding(line)
        collection.add(
            ids=[f"id_{i}"],
            embeddings=[vector],
            documents=[line]
        )
    print(f"✅ 成功导入知识数据")

def rag_answer(query: str):
    """RAG 核心流程：检索 -> 增强 -> 生成"""
    
    # 第一步：检索 (Retrieve)
    query_vector = get_embedding(query)
    # 搜索最相似的前 2 条内容
    results = collection.query(query_embeddings=[query_vector], n_results=2)
    
    # 提取参考资料
    retrieved_context = "\n".join(results['documents'][0])
    print(f"\n🔍 [检索到的参考资料]：\n{retrieved_context}\n")

    # 第二步：增强 (Augment) - 构建 Prompt
    prompt = f"""你是一个专业的助手。请根据以下提供的【参考资料】来回答用户的问题。
如果你在参考资料中找不到答案，请诚实地告诉用户你不知道，不要胡编乱造。

【参考资料】：
{retrieved_context}

【用户问题】：
{query}
"""

    # 第三步：生成 (Generate) - 让 DeepSeek 回答
    response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 # 保持严谨，低随机性
    )
    
    return response.choices[0].message.content

# --- 5. 运行入口 ---
if __name__ == "__main__":
    # 第一次运行时，请确保根目录有 my_data.txt，并取消下面一行的注释
    ingest_data("my_data.txt")
    
    print("🚀 RAG 机器人已启动 (输入 'q' 退出)")
    while True:
        user_input = input("👤 用户：")
        if user_input.lower() in ['q', 'exit', 'quit']:
            break
        
        try:
            answer = rag_answer(user_input)
            print(f"🤖 AI：{answer}\n")
        except Exception as e:
            print(f"❌ 发生错误: {e}")