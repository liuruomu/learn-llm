import chromadb

# 1. 连接到已有的数据库目录
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
db = chromadb_client.get_or_create_collection(name="my_knowledge_base")

# 2. 直接读取
all_data = db.get()

if not all_data['ids']:
    print("数据库目前是空的！")
else:
    print(f"✅ 成功连接数据库。当前条数: {len(all_data['ids'])}")
    print("-" * 30)
    res = db.get(ids=["chunk_0"], include=["documents", "embeddings"])

    print(f"📄 文本内容: {res['documents'][0]}")
    print(f"🔢 向量长度: {len(res['embeddings'][0])}")
    print(f"📍 前 10 位坐标: {res['embeddings'][0][:10]}")
    for i in range(len(all_data['ids'])):
        print(f"[{all_data['ids'][i]}] -> {all_data['documents'][i]}")