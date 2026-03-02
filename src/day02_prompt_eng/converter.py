import os
import json
import httpx  # 修正 1: 必须导入 httpx
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# --- 1. 定义数据结构 ---
class ExpenseItem(BaseModel):
    item: str = Field(description="商品或服务的名称")
    # 建议 category 和 amount 的描述也用中文，方便国内模型理解
    category: str = Field(..., description="开支类别（例如：饮食、交通）")
    amount: float = Field(..., description="开支金额")
    
class ExpenseReport(BaseModel):
    expenses: List[ExpenseItem] = Field(..., description="开支明细列表")
    total_count: int = Field(..., description="总条数") # 修正 2: 统一字段名

# --- 2. 初始化配置 (移出类定义) ---
load_dotenv()
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    http_client=http_client
)

# --- 3. 编写 Prompt ---
SYSTEM_PROMPT = """你是一个专业的财务记账专家。你的任务是将用户凌乱的记账笔记转换为标准的 JSON 格式。
### 规则：
1. 必须只输出 JSON，不要包含任何解释文字。
2. 如果金额中没有单位，默认为元。
3. 如果无法识别类别，请归类为'其他'。

### 例子 (Few-shot)：
用户：今天打车花了20，中午吃面15。
助手：{"expenses": [{"item": "打车", "amount": 20.0, "category": "交通"}, {"item": "吃面", "amount": 15.0, "category": "饮食"}], "total_count": 2}
"""

def extract_expenses(user_input: str): # 修正 3: 统一函数名
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ], # 修正 4: 这里原本漏掉了一个逗号
            response_format={"type": "json_object"},
            temperature=0.3
        )
        json_str = response.choices[0].message.content
        print(f"AI 输出的原始字符串: {json_str}")
       
        # 修正 5: 使用 Pydantic 验证并转换
        return ExpenseReport.model_validate_json(json_str)
    except ValidationError as e:
        print(f"数据验证错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 测试输入
    test_input = "昨天下午去超市买菜花了56.5，然后给公交卡充了100，刚才路边买了个煎饼果子8块"
    
    result = extract_expenses(test_input)
    
    if result:
        print("\n✅ 提取成功！")
        for i, exp in enumerate(result.expenses, 1):
            print(f"  {i}. 【{exp.category}】{exp.item}: ￥{exp.amount}")
        print(f"📊 总计条数: {result.total_count}")