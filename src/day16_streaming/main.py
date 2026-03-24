import os 
import httpx
import asyncio
from typing import Dict, Optional,List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境建议改为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 初始化 LLM
http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client,
    streaming=True # 开启流式支持
)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] # SDK 默认发送的是 messages 数组
# --- 核心：异步生成器函数 ---
async def generate_chat_responses(message: str):
    async for chunk in llm.astream([HumanMessage(content=message)]):
        content = chunk.content
        if content:
            # 直接返回文本内容，不要加 "data:" 和 "\n\n"
            yield content

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    # 使用 StreamingResponse 返回生成器
    # media_type 必须是 text/event-stream
    last_message = request.messages[-1].content
    return StreamingResponse(
        generate_chat_responses(last_message), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
