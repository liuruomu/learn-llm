import os
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# --- 1. 初始化客户端 ---
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
)
# 初始化 Tavily 搜索客户端
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- 2. 定义真实的搜索工具 ---
def web_search(query: str):
    """
    在互联网上进行实时搜索，获取最新的信息。
    """
    print(f"🌐 正在搜索：{query}...")
    # search 接口会返回多个搜索结果的摘要
    search_result = tavily.search(query=query, search_depth="advanced", max_results=3)
    
    # 简化结果，只把内容提取出来发给 AI
    context = [obj['content'] for obj in search_result['results']]
    return "\n\n".join(context)

# --- 3. 定义工具描述 (Tool Definition) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "当用户问及实时新闻、天气、或不确定的事实时，使用此工具进行联网搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_query: str):
    messages = [{"role": "user", "content": user_query}]
    
    # 第一次提问：让 AI 决定是否需要搜索
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    
    msg = response.choices[0].message
    
    # 检查是否需要调用工具
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                # 执行真实搜索
                search_content = web_search(args['query'])
                
                # 反馈给 AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "web_search",
                    "content": search_content
                })
        
        # 第二次提问：AI 结合搜索结果总结
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return final_response.choices[0].message.content
    else:
        return msg.content

if __name__ == "__main__":
    print("🚀 联网搜索助手已启动")
    while True:
        user_q = input("👤 用户：")
        if user_q.lower() == 'q': break
        
        answer = run_agent(user_q)
        print(f"🤖 AI：{answer}\n")