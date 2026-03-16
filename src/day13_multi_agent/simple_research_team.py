import os
import json
import httpx
from typing import TypedDict, Annotated,List,Union
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage,ToolMessage
from langchain_core.tools import tool # 定义工具
from langchain_community.tools.tavily_search import TavilySearchResults #搜索工具


load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)


# --- 1. 定义 AgentState (全局状态) ---
# 这是一个更复杂的全局状态，用于多 Agent 之间的消息传递

class AgentState(TypedDict):
    """AgentState"""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]  # 存储所有 Agent 之间的消息
    next_agent: str

# --- 2. 初始化模型与工具 ---
http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat", # DeepSeek 擅长代码和指令遵循，适合 Agent 的大脑
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0.7 # Agent 之间对话可以稍微高一点点随机性
)

# 搜索工具 (给研究员用)
search_tool = TavilySearchResults(k=3)
tools = [search_tool]

# 将工具绑定到 LLM (这是 LangChain 绑定工具的标准方式)
llm_with_tools = llm.bind_tools(tools)


class Agent:
    def __init__(self, llm_with_tools: ChatOpenAI, tools: List[tool], name: str):
        self.llm = llm_with_tools
        self.tools = tools
        self.name = name

    def __call__(self, state: AgentState):
        """每个 Agent 都是一个 LangGraph 节点，接收状态，返回新的状态"""
        # 1. 生成消息
        messages = state["messages"]
        # 2. 调用 LLM 生成新的消息
        response = self.llm.invoke(messages)
        if(response.tool_calls):
            print(f"[{self.name} Agent] 决定调用工具...")
            tool_outputs = []
            for tool_call in response.tool_calls:
                # 原来是 tool_call.function.name，现在改为 tool_call["name"]
                tool_name = tool_call["name"] 
                # 原来是 tool_call.function.args，现在改为 tool_call["args"]
                tool_args = tool_call["args"]
                selected_tool = {"tavily_search_results_json": search_tool}[tool_call["name"]] # 这里根据你的工具名做映射
                output = selected_tool.invoke(tool_call["args"])
                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
            # 将工具调用请求和结果都加入消息历史
            return {"messages": [response] + tool_outputs}

        else:
            print(f"[{self.name} Agent] 没有工具调用，直接返回消息...")
            return {"messages": [response]}

# 创建两个 Agent 实例
researcher_agent = Agent(llm_with_tools, tools, "研究员")
reporter_agent = Agent(llm_with_tools, [], "报告员") # 报告员没有工具

# --- 4. 定义路由函数 (决定下一个 Agent) ---
def route_next_agent(state: AgentState):
    """根据最新消息决定下一个 Agent"""
    last_message = state['messages'][-1]
    
    # 如果最后一个消息是工具结果，通常是研究员还在工作
    if isinstance(last_message, ToolMessage):
        print("[Router] 上一条消息是工具结果，继续给研究员。")
        return "researcher"
    
    # 如果 AI 回复中提到了“完成”或者不再需要工具，则转给报告员
    # 这是一个简单的判断逻辑，实际可以更复杂
    if "完成" in last_message.content or not last_message.tool_calls:
        print("[Router] 研究完成或无工具调用，转给报告员。")
        return "reporter"
    
    print("[Router] 继续给研究员处理。")
    return "researcher" # 默认继续给研究员

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("reporter", reporter_agent)

workflow.set_entry_point("researcher")

# 研究员可以循环调用自己 (进行多轮搜索)
# workflow.add_edge("researcher", "reporter") # 6，我发现这2行是互斥的。这是一个硬编码，直接转给下一个节点了。
# 这相对来说就灵活一点，会根据路由函数的判断来决定是继续给研究员还是转给报告员。
workflow.add_conditional_edges(
    "researcher",
    route_next_agent,
    {
        "researcher": "researcher", # 循环自己
        "reporter": "reporter",     # 转给报告员
        "end": END
    }
)

# 报告员完成任务就结束
workflow.add_edge("reporter", END)

# 编译图
app = workflow.compile()

if __name__ == "__main__":
    initial_state = {"messages": [HumanMessage(content="请帮我搜索 DeepMind 公司最新的研究成果，并总结一下。")]}
    
    # app.stream 可以实时打印每个节点的结果
    print("🚀 启动多智能体研究团队...")
    for s in app.stream(initial_state):
        if "__end__" not in s:
            print(s)
            print("---")
    
    final_state = app.invoke(initial_state)
    print("\n" + "="*50)
    print("✅ 最终报告：")
    print(final_state['messages'][-1].content)

    # 尝试了一下，感觉调用不可控，容易一直循环下去，可能造成token的浪费，并且得到的数据本身结果比较乱，也需要额外的处理；