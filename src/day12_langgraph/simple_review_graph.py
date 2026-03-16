import os
import httpx
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ.pop("ALL_PROXY", None)

# --- 1. 定义状态 (State) ---
# 这是节点之间传递的“账本”
class AgentState(TypedDict):
    topic: str
    draft: str
    critique: str
    revision_count: int

# --- 2. 初始化模型 ---
http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0.7
)

# --- 3. 定义节点 (Nodes) ---

def writer_node(state: AgentState):
    """写手：负责创作或修改"""
    print(f"✍️ 写手开始工作 (第 {state.get('revision_count', 0) + 1} 次尝试)...")
    
    prompt = f"主题: {state['topic']}\n"
    if state.get("critique"):
        prompt += f"反馈意见: {state['critique']}\n请根据反馈修改文案。"
    
    response = llm.invoke(prompt)
    return {
        "draft": response.content, 
        "revision_count": state.get("revision_count", 0) + 1
    }

def critic_node(state: AgentState):
    """审核员：检查字数是否达标"""
    print("🧐 审核员检查中...")
    draft = state["draft"]
    
    if len(draft) < 50:
        return {"critique": "文案太短了，请扩充内容，增加一些细节描述，必须超过 50 字。"}
    else:
        return {"critique": "合格"}

# --- 4. 定义跳转逻辑 (Conditional Edge) ---

def should_continue(state: AgentState):
    """根据审核结果决定去向"""
    if state["critique"] == "合格":
        return "end" # 结束
    else:
        return "rewrite" # 回到写手

# --- 5. 构建图 (Graph) ---

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

# 设置起点
workflow.set_entry_point("writer")

# 连接节点 (Writer 写完必去 Critic)
workflow.add_edge("writer", "critic")

# 设置条件边 (Critic 检查完后选择路径)
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "rewrite": "writer",
        "end": END
    }
)

# 编译图
app = workflow.compile()

if __name__ == "__main__":
    # 运行
    initial_state = {"topic": "写一段关于学习 Python 的感悟", "revision_count": 0}
    print("🚀 启动 LangGraph 流程...")
    
    final_output = app.invoke(initial_state)
    
    print("\n" + "="*30)
    print("✅ 最终成品：")
    print(final_output["draft"])
    print(f"📊 修改次数：{final_output['revision_count']}")