import os
import httpx
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv()

# 处理代理及初始化 OpenAI 格式的智谱客户端
# 注意：确保环境变量 ZHIPUAI_BASE_URL 是 https://open.bigmodel.cn/api/paas/v4
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url=os.getenv("ZHIPUAI_BASE_URL"),
    http_client=http_client
)

# 新增：通过 API 获取向量的函数
def get_embedding_from_api(text: str):
    """调用智谱 API 获取向量"""
    # 清理换行符以提高效果
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model="embedding-3",  # 确保你的账号有此模型权限
        input=text
    )
    return response.data[0].embedding

# 2. 文本切片逻辑
# def split_text(text, chunk_size=100, overlap=20):
#     chunks = []
#     for i in range(0, len(text), chunk_size - overlap):
#         chunks.append(text[i:i+chunk_size])
#     return chunks

def smart_split(text, max_size=50):
    # 尝试按句号切分，而不是按字符数死切
    sentences = text.split('。') 
    chunks = []
    current_chunk = ""
    
    for s in sentences:
        if not s.strip(): continue
        # 如果当前累积的句子长度加上新句子没超过限制，就合并
        if len(current_chunk) + len(s) <= max_size:
            current_chunk += s + "。"
        else:
            # 超过了，就把当前的存起来，开个新的
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = s + "。"
            
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 3. 初始化向量数据库
# 注意：如果之前用过本地模型，建议先删除 ./chroma_db 文件夹，因为 API 的向量维度不同
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
db = chromadb_client.get_or_create_collection(name="my_knowledge_base")

long_text = """
人工智能（AI）是模拟人类智能的技术。它的核心包括机器学习、深度学习等。
机器学习是 AI 的一个子集，它让计算机能够从数据中学习，而不是通过显式编程。
深度学习则模仿人类大脑的神经元结构。
2023年是大模型爆发的一年，GPT-4、Claude 和 DeepSeek 等模型表现优异。
Agent（智能体）是目前 AI 领域的高级形态，它不仅能对话，还能使用工具完成复杂任务。
"""

# 4. 处理并存入数据库
print("正在对文档进行切片...")
chunks = smart_split(long_text)

print(f"正在通过 API 获取向量并存入数据库，共 {len(chunks)} 个切片...")

for idx, chunk in enumerate(chunks):
    # --- 关键修改：调用 API 获取向量 ---
    embedding = get_embedding_from_api(chunk)
    db.add(
        ids=[f"chunk_{idx}"],
        documents=[chunk],
        embeddings=[embedding]
    )

# 5. 查询逻辑
def query_knowledge(user_query: str):
    # --- 关键修改：查询时也需要调用 API 将问题向量化 ---
    query_embedding = get_embedding_from_api(user_query)
    
    results = db.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    print(f"\n🔍 用户提问: {user_query}")
    print("🤖 数据库检索到的最相关片段:")
    for doc in results['documents'][0]:
        print(f"--- {doc.strip()} ---")

if __name__ == "__main__":
    try:
        query_knowledge("什么是智能体？")
        query_knowledge("学习数据的方法叫什么？")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查 ZHIPUAI_BASE_URL 是否正确（应为 https://open.bigmodel.cn/api/paas/v4）")