import os
import httpx
from dotenv import load_dotenv

# 1. 必须在加载任何 AI 库之前运行这个
load_dotenv()

# --- 【关键修复：物理屏蔽系统代理环境变量】 ---
# 这会防止底层 httpx 去读取那个格式错误的 socks:// 变量
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# 2. 手动创建一个不信任环境变量的客户端
http_client = httpx.Client(trust_env=False)

# 3. 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,  # 传入自定义客户端
    temperature=0
)
print("DEEPSEEK_API_KEY =", os.getenv("DEEPSEEK_API_KEY"))
# --- 2. 使用 LangChain 的方式定义工具 ---
# 方式 A：直接使用社区现成的工具 (Tavily)
search_tool = TavilySearchResults(k=3) # k=3 表示返回 3 条结果

# 方式 B：使用 @tool 装饰器自定义工具 (非常方便！)
@tool
def multiply(a: int, b: int) -> int:
    """当你需要计算两个数字相乘时使用此工具。"""
    return a * b

# 将工具放入列表
tools = [search_tool, multiply]

# --- 3. 核心：将工具绑定到模型 ---
# 这步相当于 Day 8/9 里的 tools=[] 参数，但 LangChain 会处理得更好
llm_with_tools = llm.bind_tools(tools)

# --- 4. 执行流程 ---
def run_simple_chain(user_input: str):
    print(f"👤 用户: {user_input}")
    
    # 这一步会自动处理 Prompt 和 Tool Calling 的请求生成
    ai_msg = llm_with_tools.invoke(user_input)
    
    # 检查是否有工具调用
    if ai_msg.tool_calls:
        print(f"🛠️ AI 决定调用工具: {ai_msg.tool_calls[0]['name']}")
        
        # 此时你可以看到 LangChain 帮你解析好的参数
        # 实际生产中我们会配合 AgentExecutor 自动运行，今天先手动运行一下观察：
        for tool_call in ai_msg.tool_calls:
            # 自动寻找工具并运行
            selected_tool = {"tavily_search_results_json": search_tool, "multiply": multiply}[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"]) # .invoke 是langchain中的标准方法，用于调用工具，它等同于执行search_tool(args)或multiply(args)
            print(f"📦 工具返回结果长度: {len(str(tool_output))}")
            
            # 在 LangChain 中，我们会把结果传回给模型进行最终总结
            # 我们通过组合消息数组来实现
            final_msg = llm_with_tools.invoke([
                ("human", user_input),
                ai_msg,
                {"role": "tool", "content": str(tool_output), "tool_call_id": tool_call["id"]}
            ])
            return final_msg.content
    else:
        return ai_msg.content

if __name__ == "__main__":
    # 测试联网搜索
    print(f"🤖 AI: {run_simple_chain('DeepSeek 现在的最新模型是什么？')}\n")
    
    # 测试自定义计算工具
    print(f"🤖 AI: {run_simple_chain('999 乘以 888 等于多少？')}\n")



    # 这个文件跑起来，需要注册tavily账号,并获取到key 放在同目录下的.env文件中，
    # search_tool = TavilySearchResults(k=3) # k=3 表示返回 3 条结果 的时候会自动去加载这个key，进行联网搜索，获取到最新的信息。