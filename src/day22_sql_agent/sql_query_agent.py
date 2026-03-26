import os
import httpx
from typing import Annotated, TypedDict
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

# --- 1. 环境与数据库配置 ---
http_client = httpx.Client(trust_env=False)
DB_PATH = "sales_data.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

# (保持你之前的 setup_mock_db 和工具定义不变)
@tool
def get_db_schema():
    """获取数据库的表结构信息，包括表名和字段名"""
    return "Table: sales\nColumns: id (INTEGER), product (TEXT), amount (FLOAT), date (TEXT)"

@tool
def run_sql_query(query: str):
    """执行 SQL 查询语句并返回结果。注意：只允许 SELECT 语句。"""
    if not query.strip().lower().startswith("select"):
        return "错误：只允许 SELECT 查询。"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return str(result.fetchall())
    except Exception as e:
        return f"SQL 执行出错: {str(e)}"

tools = [get_db_schema, run_sql_query]
tool_node = ToolNode(tools)

# --- 2. 定义状态 (State) ---
class AgentState(TypedDict):
    # add_messages 会将新消息追加到历史记录中，而不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]

# --- 3. 定义逻辑节点 ---
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0
).bind_tools(tools)

def call_model(state: AgentState):
    """让 LLM 决定下一步做什么"""
    messages = state['messages']
    # 如果没有系统提示词，可以加一个
    if not any(isinstance(m, HumanMessage) for m in messages):
         pass # 可以在初始化时注入
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """条件边：判断是继续调用工具还是结束"""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 4. 构建图 ---
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 设置入口
workflow.set_entry_point("agent")

# 添加条件边：agent -> tools (如果有工具调用) 或 agent -> END
workflow.add_conditional_edges("agent", should_continue)

# 添加普通边：tools 执行完后总是回到 agent
workflow.add_edge("tools", "agent")

# 编译
app = workflow.compile()

# --- 5. 运行测试 ---
async def main():
    inputs = {
        "messages": [
            ("system", "你是一个专业数据分析师。通过查询数据库回答问题。"),
            ("user", "帮我统计一下 iPhone 15 的总销售额是多少？")
        ]
    }
    
    # 使用 stream 模式可以看到 AI 的思考过程
    async for output in app.astream(inputs):
        for key, value in output.items():
            print(f"\n--- Node: {key} ---")
            if "messages" in value:
                last_msg = value["messages"][-1]
                # 打印出主要内容，过滤掉复杂的 metadata
                print(last_msg.content if last_msg.content else f"Tool Calls: {last_msg.tool_calls}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



    ### 我在示例的基础上进行了修改，改成了langgraph的方式，让它自动通过多个步骤来完成用户的目的。nice！