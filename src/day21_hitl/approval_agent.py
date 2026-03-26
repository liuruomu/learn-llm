import os
import asyncio
import httpx
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
# --- 关键修改 1: 使用 aio 版本的 checkpointer ---
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

http_client = httpx.Client(trust_env=False)
# --- 1. 定义工具 ---
@tool
def execute_transfer(amount: float, recipient: str):
    """执行转账操作"""
    return f"✅ 已成功向 {recipient} 转账 ${amount}"

tools = [execute_transfer]
tool_node = ToolNode(tools)

# --- 2. 定义状态与模型 ---
class AgentState(TypedDict):
    # 使用列表追加模式
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# 使用异步 Client
# Inside your call_model function
async def call_model(state: AgentState):
    # Create the client right here
    async with httpx.AsyncClient(trust_env=False) as client:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            http_client=client, # Use the local 'client'
            temperature=0 
        ).bind_tools(tools)
        
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}
# --- 3. 构建图 ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")

def route(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "action"
    return END

workflow.add_conditional_edges("agent", route)
workflow.add_edge("action", "agent")

# --- 4. 运行逻辑 ---
async def run_step_by_step():
    # --- 关键修改 2: 在异步上下文管理器中初始化 checkpointer ---
    async with AsyncSqliteSaver.from_conn_string("approval.db") as memory:
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=["action"] 
        )
        
        thread_config = {"configurable": {"thread_id": "tx_999"}}
        
        print("\n--- 步骤 1: 用户请求转账 ---")
        initial_input = {"messages": [HumanMessage(content="帮我转账 500 元给老王")]}
        
        # 第一次运行：会停在 action 之前
        async for event in app.astream(initial_input, thread_config):
            print(f"DEBUG Event: {list(event.keys())}")

        # 检查当前挂起的状态
        state = await app.get_state(thread_config)
        if state.next:
            print(f"\n📢 当前挂起的节点: {state.next}") 
            last_msg = state.values['messages'][-1]
            print(f"⚠️ 待审批的参数: {last_msg.tool_calls[0]['args']}")
            
            print("\n--- 模拟：后端已向前端发送审批请求，现在用户点击【批准】 ---")
            
            # 第二步：恢复执行
            # 传 None 表示从当前 Checkpoint 继续
            async for event in app.astream(None, thread_config):
                if "action" in event:
                    print(f"Action Output: {event['action']['messages'][0].content}")
                elif "agent" in event:
                    print(f"Final AI Response: {event['agent']['messages'][0].content}")

if __name__ == "__main__":
    try:
        asyncio.run(run_step_by_step())
    finally:
        # 记得关闭异步 client
        asyncio.run(async_http_client.aclose())