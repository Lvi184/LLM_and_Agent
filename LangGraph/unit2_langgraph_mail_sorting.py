import os
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ----------------------
# glm-5.1（推荐方式）
# ----------------------
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="glm-5.1",
    temperature=0
)

# ----------------------
# 状态（完全对齐）
# ----------------------
class EmailState(TypedDict):
    email: Dict[str, Any]
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]

# ----------------------
# 节点1：读取邮件
# ----------------------
def read_email(state: EmailState):
    email = state["email"]
    print(f"📧 来自: {email['sender']} | {email['subject']}")
    return {}

# ----------------------
# 节点2：分类
# ----------------------
def classify_email(state: EmailState):
    email = state["email"]

    prompt = f"""
判断邮件是垃圾邮件还是正常邮件。

邮件：
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

只返回：
SPAM 或 HAM
"""

    result = llm.invoke(prompt).content.strip().upper()

    is_spam = "SPAM" in result

    return {
        "is_spam": is_spam,
        "spam_reason": result if is_spam else None
    }

# ----------------------
# 节点3：垃圾邮件处理
# ----------------------
def handle_spam(state: EmailState):
    print("🚫 垃圾邮件，已丢弃")
    return {}

# ----------------------
# 节点4：生成回复
# ----------------------
def drafting_response(state: EmailState):
    email = state["email"]

    prompt = f"""
帮我写一封简短专业回复：

From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
"""

    reply = llm.invoke(prompt).content

    return {"email_draft": reply}

# ----------------------
# 节点5：通知
# ----------------------
def notify(state: EmailState):
    print("\n📨 回复草稿：")
    print(state["email_draft"])
    return {}

# ----------------------
# 路由
# ----------------------
def route(state: EmailState):
    return "spam" if state["is_spam"] else "ok"

# ----------------------
# 构建图
# ----------------------
graph = StateGraph(EmailState)

graph.add_node("read", read_email)
graph.add_node("classify", classify_email)
graph.add_node("spam", handle_spam)
graph.add_node("reply", drafting_response)
graph.add_node("notify", notify)

graph.add_edge(START, "read")
graph.add_edge("read", "classify")

graph.add_conditional_edges(
    "classify",
    route,
    {
        "spam": "spam",
        "ok": "reply"
    }
)

graph.add_edge("spam", END)
graph.add_edge("reply", "notify")
graph.add_edge("notify", END)

app = graph.compile()

# ----------------------
# 测试
# ----------------------
if __name__ == "__main__":
    app.invoke({
        "email": {
            "sender": "Lucius",
            "subject": "Meeting",
            "body": "Can we meet tomorrow?"
        },
        "is_spam": None,
        "spam_reason": None,
        "email_draft": None,
        "messages": []
    })
    # 测试1：工作邮件
    app.invoke({"email":{
        "sender": "Lucius",
        "subject": "Meeting",
        "body":"请于明日上午10点参加项目评审会议，查看附件中的会议材料"
    }})
    
    # 测试2：个人邮件
    app.invoke({"email":{
        "sender": "Lucius",
        "subject": "Personal",
        "body":"兄弟，周末聚餐，老地方见！"
    }})
    
    # 测试3：垃圾邮件
    app.invoke({"email":{
        "sender": "Lucius",
        "subject": "Spam",
        "body":"恭喜您中奖！免费领取iPhone16，点击链接立即领取"
    }})
