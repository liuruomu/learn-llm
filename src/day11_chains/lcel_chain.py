import os
import httpx
from dotenv import load_dotenv

# 1. 基础配置与环境清理
load_dotenv()
os.environ.pop("ALL_PROXY", None) # 清理代理报错
os.environ.pop("all_proxy", None)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 2. 初始化模型
http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    temperature=0.7
)

# --- 3. 定义第一条链：关键词生成器 ---
keyword_prompt = ChatPromptTemplate.from_template(
    "你是一个 SEO 专家。请针对用户的主题：{topic}，提供 3 个核心搜索关键词，用逗号隔开。"
)

# 这是一个简单的链：Prompt -> LLM -> 字符串解析
keyword_chain = keyword_prompt | llm | StrOutputParser()

# --- 4. 定义第二条链：内容创作者 ---
write_prompt = ChatPromptTemplate.from_template(
    "你是一个社交媒体专家。根据这些关键词：{keywords}，写一段吸引人的小红书文案。"
)
write_chain = write_prompt | llm | StrOutputParser()

# --- 5. 组合成一个复杂的序列流 (Sequential Chain) ---
def run_workflow(topic: str):
    print(f"🚀 开始处理主题: {topic}")
    
    # 第一步：运行关键词链
    keywords = keyword_chain.invoke({"topic": topic})
    print(f"🔑 生成的关键词: {keywords}")
    
    # 第二步：将第一步的结果传给第二步
    final_content = write_chain.invoke({"keywords": keywords})
    return final_content

if __name__ == "__main__":
    topic = "全栈 LLM Agent 开发学习"
    result = run_workflow(topic)
    print("\n📝 最终生成的文案：")
    print(result)