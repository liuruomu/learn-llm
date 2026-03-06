import os
import httpx
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

http_client = httpx.Client(trust_env=False)
embed_client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url=os.getenv("ZHIPUAI_BASE_URL"),
    http_client=http_client
)

# --- 1. 初始化客户端 (沿用之前的配置) ---
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
    
)
# 假设你已经有 Day 6 运行好的 ChromaDB
# 修改这一行，指向 day06 的数据库路径
chroma_client = chromadb.PersistentClient(path="../day06_pdf_rag/chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_knowledge")

# --- 2. 记忆管理类 ---
class ChatMemory:
    def __init__(self, window_size=5):
        # window_size=5 表示只记得最近 5 轮对话，防止 Token 爆炸
        self.messages = []
        self.window_size = window_size

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        # 如果对话太长，删除最早的对话（保留 system prompt 除外）
        if len(self.messages) > self.window_size * 2:
            self.messages = self.messages[-self.window_size * 2:]

    def get_all_messages(self, system_prompt):
        return [{"role": "system", "content": system_prompt}] + self.messages

# --- 3. 增强版 RAG 问答逻辑 ---
memory = ChatMemory(window_size=3)


def get_embedding(text: str):
    response = embed_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding
def search_chroma(query: str, top_k: int = 3):
    # 1. 向量化问题
    query_vector = get_embedding(query)
    
    # 2. 从集合中查询
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    # 3. 提取文本内容 (results['documents'] 是一个嵌套列表)
    if results['documents']:
        return "\n".join(results['documents'][0])
    return "没有找到相关的参考资料。"

def ask_with_memory(query: str):
    # A. 检索 (这里为了演示简化，假设 get_embedding 函数已定义)
    # 实际开发中请复用 Day 6 的 get_embedding
    retrieved_context = search_chroma(query) 
    retrieved_context = "这里是之前 PDF 里的参考内容..." # 模拟检索结果

    # B. 构建当前的 System Prompt (包含参考资料)
    system_prompt = f"""你是一个有记忆的助手。参考资料如下：
{retrieved_context}
请结合参考资料和对话历史来回答。"""

    # C. 将用户问题加入记忆
    memory.add_message("user", query)

    # D. 调用 DeepSeek (传入完整的对话历史)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=memory.get_all_messages(system_prompt),
        temperature=0.7
    )

    answer = response.choices[0].message.content
    
    # E. 将 AI 的回答也加入记忆！
    memory.add_message("assistant", answer)
    
    return answer

if __name__ == "__main__":
    print("🤖 记忆助手已就绪（输入 'q' 退出）")
    while True:
        user_input = input("👤 用户：")
        if user_input.lower() == 'q': break
        
        result = ask_with_memory(user_input)
        print(f"🤖 AI：{result}\n")