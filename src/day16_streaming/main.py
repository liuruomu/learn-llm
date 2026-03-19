import os 
import httpx
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
import logging

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEBUG_AGENT")

app = FastAPI()

# 初始化 LLM
http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    streaming=True # 开启流式支持
)


class ChatRequest(BaseModel):
    message: str

# --- 核心：异步生成器函数 ---
async def generate_chat_responses(message: str):
    """
    这个函数会像挤牙膏一样，把 AI 的回复一节一节地‘吐’出来
    """
    # 使用 LangChain 的 astream 方法进行流式调用
    async for chunk in llm.astream([HumanMessage(content=message)]):
        # chunk.content 是当前这一小块文本内容
        content = chunk.content
        if content:
            # 按照 SSE 的标准格式发送：data: 内容\n\n
            yield f"data: {content}\n\n"
            # 稍微停顿一下（可选），模拟人类打字感
            await asyncio.sleep(0.01)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    # 使用 StreamingResponse 返回生成器
    # media_type 必须是 text/event-stream
    return StreamingResponse(
        generate_chat_responses(request.message), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)