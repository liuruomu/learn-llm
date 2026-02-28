import os
import httpx  # 1. 新增导入
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 2. 创建一个忽略环境变量代理的 httpx 客户端
http_client = httpx.Client(trust_env=False)

# 3. 将这个 client 传给 OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client  # 使用我们自定义的客户端
)

def get_ai_response():
    # 下面的代码保持不变...
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个助手。"},
                {"role": "user", "content": "你好，我是你的 AI 助手"}
            ]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    get_ai_response()