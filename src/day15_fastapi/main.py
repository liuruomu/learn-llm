import os 
import httpx
from typing import Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[Dict]] = []

class ChatResponse(BaseModel):
    status: str
    reply: str

app = FastAPI(title="Chat API with DeepSeek")

http_client = httpx.Client(trust_env=False)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
)

@app.get("/")
def read_root():
    return {"message": "AI Agent 服务已启动"}

from fastapi import FastAPI, HTTPException
import traceback
# ... other imports

@app.post("/chat", response_model=ChatResponse) # Added validation
async def chat_endpoint(request: ChatRequest):
    logger.info(f"📩 收到消息: {request.message}")
    try:
        # LangChain handles the message conversion
        response = await llm.ainvoke([HumanMessage(content=request.message)])
        
        logger.info("📡 API 调用成功")
        return {
            "status": "success", 
            "reply": response.content
        }
    
    except Exception as e:
        logger.error("🔥 接口执行崩溃！")
        # Print the stack trace so you can actually see what happened
        error_trace = traceback.format_exc()
        print(f"{'-'*50}\n{error_trace}{'-'*50}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )