import os
import httpx
import operator
from typing import Annotated, List, TypedDict, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

# --- 1. 定义团队状态 ---
class TeamState(TypedDict):
    # Annotated[..., operator.add] 确保消息是增量追加的，而不是覆盖
    #Annotated[...]: 这是 Python 的元数据标注工具。它本身不改变数据类型，但可以给变量贴上“额外标签”，供框架（如 LangChain/LangGraph）在后台读取并执行特殊逻辑。

#operator.add: 这是 Python 的内置加法函数。在列表语境下，[1, 2] + [3] 的结果是 [1, 2, 3]。
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str  # 主管决定的下一个执行者

# --- 2. 初始化模型 ---
http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0
)

# --- 3. 定义主管节点 (The Manager) ---
members = ["Searcher", "Writer"]
options = ["FINISH"] + members  # options = ["FINISH", "Searcher", "Writer"]

system_prompt = (
    "你是一个团队主管。根据用户的问题和团队的进展，决定下一步由谁工作。"
    "每个工人在完成任务后都会向你汇报。如果任务已圆满完成，请输出 FINISH。"
    "可选的下属有: {members}"
)

# 使用 LangChain 的工具调用功能来强制主管输出选项
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "当前你应该调度哪位成员？或者已经完成(FINISH)了？")
]).partial(members=", ".join(members))# partial() 方法用于创建一个包含指定参数的函数,这里就是用来填充system_prompt中的 {members} 占位符，使得系统提示中会显示可选的成员列表。

def supervisor_node(state: TeamState):

#
    # 强制 LLM 只能从 options 中选一个
    chain = prompt | llm.bind_tools(
        tools=[], # 这里只是为了演示，实际可以用结构化输出
    )
    # 简单起见，我们直接让 LLM 返回名字
    response = llm.invoke(prompt.format(messages=state["messages"])) # 这里我们直接调用 llm 来生成回复，传入格式化后的提示，其中包含了当前的消息历史。这将让 LLM 根据用户的问题和团队的进展来决定下一步由谁工作。
    ## prompt中就要求了llm输出一个答案，所以下面就是按照这个答案来进行逻辑提取了。
    
    # 逻辑提取：从 AI 的回复中提取成员名字
    content = response.content
    next_ = "FINISH"
    for m in options:
        if m.lower() in content.lower():
            next_ = m
            break
    print(f"🚩 [主管] 下一步决策: {next_}")
    return {"next_agent": next_}

# --- 4. 定义工人节点 (The Workers) ---

def searcher_node(state: TeamState):
    print("🔎 [搜索工] 正在搜集资料...")
    search_tool = TavilySearchResults(k=2)
    # 获取最后一条人类指令进行搜索
    query = state["messages"][0].content
    result = search_tool.invoke(query)
    return {"messages": [HumanMessage(content=f"搜索结果如下: {result}", name="Searcher")]}

def writer_node(state: TeamState):
    print("✍️ [写手] 正在润色内容...")
    last_msg = state["messages"][-1].content
    response = llm.invoke(f"请根据以下资料写一篇 100 字的短文：{last_msg}")
    return {"messages": [HumanMessage(content=response.content, name="Writer")]}

# --- 5. 构建图 ---
workflow = StateGraph(TeamState)

# 添加节点
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Searcher", searcher_node)
workflow.add_node("Writer", writer_node)

# 设置工人干完活必须回到主管这里
workflow.add_edge("Searcher", "Supervisor")
workflow.add_edge("Writer", "Supervisor")

# 设置主管的条件边（入口永远是主管）
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next_agent"],
    {
        "Searcher": "Searcher",
        "Writer": "Writer",
        "FINISH": END
    }
)

workflow.set_entry_point("Supervisor")
app = workflow.compile()

if __name__ == "__main__":
    print("🚀 团队启动...")
    input_msg = "帮我查一下 2024 年巴黎奥运会金牌榜第一名是谁，并写一段简短的新闻评论。"
    
    for chunk in app.stream({"messages": [HumanMessage(content=input_msg)]}):
        print(chunk)
        print("---")



        # next_agent 决定逻辑的下一步
        # add_edge 是工人节点的下一步