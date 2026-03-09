import os
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- 1. 定义本地 Python 函数 (Agent 的“手”) ---
def get_weather(city: str):
    """获取指定城市的天气"""
    # 实际开发中这里会调用第三方 API，现在先 Mock 一个结果
    if "上海" in city:
        return "25度，晴天"
    else:
        return "30度，多云"

def calculate_tax(salary: float):
    """计算个人所得税"""
    tax = salary * 0.2 # 假设税率 20%
    return f"工资 {salary} 的税额是 {tax}"

# --- 2. 初始化客户端 ---
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
)

# --- 3. 向 AI 描述这些工具 (Tool Definition) ---
# 这是最关键的一步，必须符合 OpenAI 的 JSON 格式要求
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如：北京, 上海"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tax",
            "description": "根据工资计算个人所得税",
            "parameters": {
                "type": "object",
                "properties": {
                    "salary": {"type": "number", "description": "月收入金额"}
                },
                "required": ["salary"]
            }
        }
    }
]

def run_conversation(user_query: str):
    print(f"👤 用户：{user_query}")
    
    # 步骤 A: 第一次提问，带上工具描述
    messages = [{"role": "user", "content": user_query}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools, # 告诉 AI 你有哪些工具
        tool_choice="auto" 
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 步骤 B: 检查 AI 是否决定调用工具
    if tool_calls:
        print("🤖 AI 决定调用工具...")
        
        # 将 AI 的回复（包含工具调用请求）加入对话历史
        messages.append(response_message)
        
        # 步骤 C: 在本地执行函数
        # 一个请求中可能包含多个工具调用
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"🛠️ 执行本地函数: {function_name}(**{function_args})")
            
            if function_name == "get_weather":
                result = get_weather(city=function_args.get("city"))
            elif function_name == "calculate_tax":
                result = calculate_tax(salary=function_args.get("salary"))
            
            # 将函数执行结果传回给 AI
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(result)
            })
            
        # 步骤 D: 第二次提问，让 AI 根据结果总结
        second_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return second_response.choices[0].message.content
    else:
        return response_message.content

if __name__ == "__main__":
    # 测试 1: 触发天气工具
    print(f"🤖 AI 回复：{run_conversation('上海今天天气怎么样？')}\n")
    
    # 测试 2: 触发税务工具
    print(f"🤖 AI 回复：{run_conversation('我月薪 30000 元，要交多少税？')}\n")
    
    # 测试 3: 不触发工具
    print(f"🤖 AI 回复：{run_conversation('你好')}\n")