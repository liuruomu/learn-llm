import os
import asyncio
import httpx
import time
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # 官方提供的工具执行节点，支持并发

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

# --- 1. 定义异步工具 (模拟耗时操作) ---

@tool
async def fetch_stock_price(ticker: str):
    """查询实时股价"""
    print(f"📉 开始查询股价: {ticker}...")
    await asyncio.sleep(2)  # 模拟 IO 耗时
    return f"{ticker} 当前价格为 $150.00"

@tool
async def fetch_weather(city: str):
    """查询实时天气"""
    print(f"☁️ 开始查询天气: {city}...")
    await asyncio.sleep(2)  # 模拟 IO 耗时
    return f"{city} 今天晴转多云，25度"

@tool
async def fetch_news(topic: str):
    """查询最新新闻摘要"""
    print(f"📰 开始查询新闻: {topic}...")
    await asyncio.sleep(2)  # 模拟 IO 耗时
    return f"关于 {topic} 的最新消息：大模型技术取得新突破"

tools = [fetch_stock_price, fetch_weather, fetch_news]

# --- 2. 构建图逻辑 ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat", # DeepSeek 擅长代码和指令遵循，适合 Agent 的大脑
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0 
).bind_tools(tools)
# 定义 Node
async def call_model(state: AgentState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# 使用 LangGraph 内置的 ToolNode
# 它会自动检查 LLM 的 tool_calls 并并行执行所有 async 类型的 tool
tool_node = ToolNode(tools)

# --- 3. 编排图 ---

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

# 路由逻辑：检查是否需要调用工具
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 4. 运行并计时 ---

async def main():
    start_time = time.time()
    
    query = "帮我查一下英伟达(NVDA)的股价，顺便看看上海的天气和最近的AI新闻。"
    print(f"🚀 用户提问: {query}\n")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # 异步流式运行
    async for output in app.astream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            print(f"\n--- 节点 {key} 执行完毕 ---")
            last_msg = value["messages"][-1]
            
            # 如果是工具节点，它返回的是 ToolMessage
            if key == "tools":
                for msg in value["messages"]:
                    print(f"工具原始输出: {msg.content}")
            
            # 如果是 Agent 节点，它返回的是 AIMessage
            else:
                print(f"Agent 回复: {last_msg.content}")
            
    end_time = time.time()
    print(f"\n⏱️ 总耗时: {end_time - start_time:.2f} 秒")
    print("提示：如果同步执行 3 个工具，耗时应 > 6秒；异步执行应在 2-3 秒左右。")

if __name__ == "__main__":
    asyncio.run(main())


#     1. 消息链的结构（ReAct 模式）
# 在 LangGraph 的 ReAct 架构中，消息是按顺序堆叠的。当你问股价时，消息链如下：

# HumanMessage: "帮我查一下 NVDA..."

# AIMessage (from agent node): 包含 tool_calls（我要查股价、天气、新闻）。

# ToolMessage (from tools node): 包含 f"{ticker} 当前价格为 $150.00"。 <-- 你的数据在这里

# AIMessage (from agent node): LLM 读了 ToolMessage 后生成的总结：“NVDA 现在的价格是 150 美元，上海天气...”




# 整个 Workflow 的执行流转图
# 用户输入 (User Start): 产生 HumanMessage。

# State: [HumanMessage]

?# 决策阶段 (Agent Node - 第一次): LLM 接收到 HumanMessage，决定调用工具。

# 输出: 一个带有 tool_calls 列表的 AIMessage（此时 content 通常为空）。

# State: [HumanMessage, AIMessage(tool_calls)]

# 执行阶段 (Tools Node): ToolNode 看到上一个消息有 tool_calls，并行执行你的 3 个异步函数。

# 输出: 3 个 ToolMessage（每个包含一个工具的 return 字符串，比如 $150.00）。

# State: [HumanMessage, AIMessage, ToolMessage1, ToolMessage2, ToolMessage3]

# 总结阶段 (Agent Node - 第二次): LLM 重新读入整个消息链（包括刚才那 3 个 ToolMessage），然后组织语言。

# 输出: 最后一个 AIMessage（纯文本总结）。

# State: [..., ToolMessages, AIMessage(Final Summary)]


答案：# 在 LangGraph 中，这个决策过程可以拆解为以下三个核心环节：

# 1. 赋予能力：工具的“身份证” (Tool Definition)
# 当你定义 @tool 时，你其实给 LLM 提供了一份详细的说明书。

# 函数名 (fetch_stock_price)：告诉 LLM 这是干什么的。

# 文档字符串 ("""查询实时股价""")：这是 LLM 读懂工具用途的唯一依据。

# 参数类型 (ticker: str)：告诉 LLM 调用时需要提供什么数据。

# 当你执行 llm.bind_tools(tools) 时，这些信息被转换成了一组特定的 JSON Schema（类似于 OpenAI 的 Function Calling 协议），并作为“背景知识”塞进了给 LLM 的 System Prompt 里。

# 2. 语义触发：意图识别 (Intent Recognition)
# 当你输入“帮我查一下英伟达的股价”时，LLM 会进行以下逻辑推理：

# 分析需求：用户想要“股价”。

# 检索工具库：我手头有三个工具：fetch_stock_price、fetch_weather、fetch_news。

# 计算相关性：fetch_stock_price 的描述（“查询实时股价”）与用户的需求高度匹配。

# 识别槽位：“英伟达”对应参数 ticker，根据其常识，它知道英伟达的代码是 NVDA。

# 3. 强迫输出：强制结构化响应 (Structured Output)
# 这是最关键的一步。普通的 LLM 喜欢闲聊，但 bind_tools 告诉模型：“如果你发现用户的需求可以用工具解决，请不要直接说话，而是输出一个特定格式的 JSON 对象。”

# 于是，DeepSeek 在“决策阶段”并没有返回 HumanMessage（文本），而是返回了一个特殊的 additional_kwargs，里面包含：

# JSON

# {
#   "tool_calls": [
#     {
#       "id": "call_123",
#       "type": "function",
#       "function": {
#         "name": "fetch_stock_price",
#         "arguments": "{\"ticker\": \"NVDA\"}"
#       }
#     }
#   ]
# }
# 4. LangGraph 的接力：路由判断 (The Router)
# LLM 只是“提议”要用工具，真正执行跳转的是你在代码里写的 should_continue 函数：

# Python

# def should_continue(state: AgentState):
#     last_message = state["messages"][-1]
#     # 检查 LLM 刚才吐出来的消息里有没有 tool_calls
#     if last_message.tool_calls:
#         return "tools" # 发现有工具调用意图，转向 tools 节点
#     return END # 没有工具调用，直接结束
# 总结
# LLM 决定调用工具是因为：

# 它看到了工具描述（你写的 docstring）。

# 它识别了用户意图（语义匹配）。

# 它被协议约束了输出格式（JSON 结构而不是纯文本）。

# 有趣的一点： 如果你把 fetch_stock_price 的描述改成 """用来做红烧肉的工具"""，当你问股价时，LLM 绝对不会调用它。这就是为什么作为开发者，写好 Tool 的 Docstring 比写代码本身更重要。