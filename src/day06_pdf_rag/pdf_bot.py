import os
import httpx
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. 初始化客户端 ---
http_client = httpx.Client(trust_env=False)
embed_client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url=os.getenv("ZHIPUAI_BASE_URL"),
    http_client=http_client
)
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
)

# --- 2. 初始化向量数据库 ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# 建议每天用不同的 collection 名字，或者清空旧的
collection = chroma_client.get_or_create_collection(name="pdf_knowledge")

def get_embedding(text: str):
    response = embed_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding

# --- 3. PDF 解析与切片函数 ---
def process_pdf(file_path: str):
    print(f"📄 正在解析 PDF: {file_path}")
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    # 使用递归字符切片器
    # chunk_size: 每个块大概 500 字符
    # chunk_overlap: 块与块之间重叠 50 字符，防止语义断裂
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(full_text)
    print(f"✂️ 已将 PDF 切分为 {len(chunks)} 个片段")
    
    # 存入向量库
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        collection.add(
            ids=[f"pdf_chunk_{i}"],
            embeddings=[vector],
            documents=[chunk]
        )
    print("✅ 知识库更新完毕")

# # --- 4. 问答逻辑 (复用并优化) ---
# def ask_pdf(query: str):
#     query_vector = get_embedding(query)
#     # 增加检索数量到 3，获取更丰富的背景
#     results = collection.query(query_embeddings=[query_vector], n_results=3)
#     context = "\n---\n".join(results['documents'][0])

#     prompt = f"""你是一个基于文档的助手。请阅读以下参考资料并回答问题。
# 资料中没提到的不要乱说。

# 【参考资料】：
# {context}

# 【问题】：
# {query}
# """
#     response = chat_client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=1
#     )
#     return response.choices[0].message.content




# --- 4. 问答逻辑 (针对已存在向量库的 30 万字小说全面优化) ---
def ask_pdf(query: str):
    # 1. 判断是否为“全局总结类”问题
    global_keywords =["大纲", "细纲", "总结", "脉络", "全书", "情节", "主线", "梗概"]
    is_global = any(kw in query for kw in global_keywords)

    if is_global:
        print("⏳ 检测到全局大纲/总结需求，正在从向量库提取全书数据...")
        
        # 绕过相似度检索，直接从 ChromaDB 获取所有已存的小说切片
        all_data = collection.get(include=['documents'])
        if not all_data['ids']:
            return "❌ 错误：向量数据库为空，请确认数据是否成功入库。"
        
        # 将切片及其 ID 配对，并按照入库时的数字顺序 (pdf_chunk_i) 重新排序
        # 这一步非常关键，它能把打碎的切片重新拼成一本顺序正确的小说
        docs_with_ids = list(zip(all_data['ids'], all_data['documents']))
        docs_with_ids.sort(key=lambda x: int(x[0].split('_')[-1]))
        all_chunks =[doc for _, doc in docs_with_ids]
        
        print(f"📦 共捞取到 {len(all_chunks)} 个小说切片，开始分段提炼（此过程可能需要几分钟，请耐心等待）...")
        
        # 2. Map阶段：分段提炼剧情（每 60 个切片约 3 万字为一部分）
        # 避免 30 万字一次性塞入大模型导致细节丢失或超出限制
        chunk_group_size = 60 
        part_summaries =[]
        
        for i in range(0, len(all_chunks), chunk_group_size):
            group = all_chunks[i:i + chunk_group_size]
            part_text = "\n".join(group)
            
            progress = min((i + chunk_group_size), len(all_chunks))
            print(f"   -> 正在提炼剧情进度: {progress}/{len(all_chunks)} 切片...")
            
            map_prompt = f"""你是一个小说编辑。请用 500 字左右，详细且连贯地总结以下小说片段中的核心情节发展、出场人物及关键事件：
            
【小说片段】：
{part_text}"""
            
            try:
                response = chat_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": map_prompt}],
                    temperature=0.3 # 总结阶段建议低温度，防止瞎编
                )
                part_summaries.append(response.choices[0].message.content)
            except Exception as e:
                print(f"提炼该部分时出错: {e}")
                part_summaries.append("（该部分总结缺失）")

        # 3. Reduce阶段：全局汇总生成大纲
        print("✨ 分卷提炼完成，正在生成最终的大纲/细纲...")
        full_summary_text = "\n\n=== 小说各阶段剧情汇总 ===\n\n".join(part_summaries)
        
        reduce_prompt = f"""你是一个专业的小说主编。以下是我为你整理的这部小说各个阶段的详细剧情汇总：
        
{full_summary_text}

请根据以上剧情汇总，满足用户的具体需求：
【用户需求】：{query}

【要求】：
1. 如果用户要求大纲/细纲，请严格按照故事发展的先后顺序排版。
2. 包含故事起因、发展、高潮、结局的阶段划分。
3. 梳理出主线剧情和主要人物的核心动向。
"""
        response = chat_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": reduce_prompt}],
            temperature=0.6 # 生成大纲时可以适当提高温度，让排版更灵活
        )
        return response.choices[0].message.content

    else:
        # 4. 常规局部问答（例如：“主角的武器是什么？”）
        print("🔍 正在数据库中检索相关片段...")
        query_vector = get_embedding(query)
        
        # 优化点：将原先的 n_results=3 提高到 20，获取大约 10000 字的上下文
        results = collection.query(query_embeddings=[query_vector], n_results=20)
        
        if not results['ids'][0]:
            return "未能检索到相关内容。"

        # 优化点：检索出的片段往往是乱序的（按相关度排），这会导致 AI 看不懂剧情。
        # 我们按照原始的切片编号对检索出来的这 20 个片段重新排序，恢复上下文连贯性
        docs_with_ids = list(zip(results['ids'][0], results['documents'][0]))
        docs_with_ids.sort(key=lambda x: int(x[0].split('_')[-1]))
        context = "\n---\n".join([doc for _, doc in docs_with_ids])

        prompt = f"""你是一个基于小说内容的问答助手。请仔细阅读以下抽取的小说参考片段，回答用户的问题。
注意：资料中没提到的不要乱说。

【小说参考片段】：
{context}

【用户问题】：
{query}
"""
        response = chat_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content




if __name__ == "__main__":
    # 第一次运行需解析 PDF
    # process_pdf("《不死》作者：妖舟.pdf")
    
    print("🤖 PDF 助手已就绪（输入 q 退出）")
    while True:
        q = input("👤 提问：")
        if q.lower() == 'q': break
        print(f"🤖 AI：{ask_pdf(q)}\n")