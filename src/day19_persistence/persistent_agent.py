import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

from dotenv import load_dotenv
import httpx

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

# 1. 定义状态
class AgentState(TypedDict):
    # operator.add 表示新消息会自动追加到列表末尾
    messages: Annotated[List[BaseMessage], operator.add]

# 2. 初始化持久化层 (使用 SQLite 模拟生产环境的 Postgres/Redis)
# 在实际全栈应用中，数据库连接通常在应用启动时初始化
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
http_client = httpx.Client(trust_env=False)
# 3. 定义一个简单的节点
def call_model(state: AgentState):
    llm = ChatOpenAI(
    model="deepseek-chat", # DeepSeek 擅长代码和指令遵循，适合 Agent 的大脑
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0.7 # Agent 之间对话可以稍微高一点点随机性
   )
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. 构建图并绑定持久化逻辑
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# 【关键点】在编译时传入 checkpointer
app = workflow.compile(checkpointer=memory)

# 5. 模拟前端多会话调用
def run_session(thread_id: str, user_input: str):
    # 配置参数：指定当前对话的 thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    input_message = HumanMessage(content=user_input)
    
    # 执行图
    for event in app.stream({"messages": [input_message]}, config):
        for value in event.values():
            print(f"Thread [{thread_id}] AI: {value['messages'][-1].content}")

if __name__ == "__main__":
    # 模拟用户 A 的对话
    run_session("user_123", "你好，我是老王。")
    
    # 模拟用户 B 的对话
    run_session("user_456", "你好，我是小李。")
    
    # 再次调用用户 A (Agent 应该记得他是老王，即使没有手动传 context)
    run_session("user_123", "请问我刚才说我叫什么？")





#     输入：你调用 app.stream(..., config)。

# 加载 (Load)：checkpointer 根据 thread_id 从 DB 读取历史 State。

# 合并 (Merge)：将历史消息与本次输入消息合并。

# 运行 (Run)：进入 call_model 节点，LLM 看到完整的历史。

# 保存 (Save)：节点输出结果，checkpointer 将更新后的 State 存回 DB。

# 输出：将结果流式返回给你。