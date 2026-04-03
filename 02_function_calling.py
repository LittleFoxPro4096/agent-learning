# ==========================================================
# 02_function_calling.py
# ==========================================================
# Function Calling -- 使用 LangChain @tool 注册工具
# 流程:
#   用户提问
#     -> LLM 判断需要调用哪些工具 (返回 tool_calls)
#     -> 程序执行对应的 Python 函数
#     -> 函数结果以 ToolMessage 回传给 LLM
#     -> LLM 综合结果生成最终回答
# ==========================================================

import json
import math
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from config import langchain_client, logger


# ===========================================================
# 第一步: 用 @tool 装饰器注册工具
# 只需: 函数 + 类型注解 + docstring, schema 自动生成
# ===========================================================
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: Timezone name, e.g. 'UTC', 'Asia/Shanghai'.
    """
    now = datetime.now()
    return json.dumps(
        {
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone,
        }
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Supports: +, -, *, /, sqrt(), sin(), cos(), pow(), log(), pi, e.

    Args:
        expression: The math expression, e.g. 'sqrt(144) + pow(2, 10)'.
    """
    allowed = {
        "sqrt": math.sqrt,
        "pow": math.pow,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"error: {str(e)}"})


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body content.
    """
    logger.info(f"[EMAIL SENT] To: {to} | Subject: {subject} | Body: {body}")
    return json.dumps({"status": "success", "message": f"Email sent to {to}"})


# ===========================================================
# 第二步: 查看自动生成的 schema (可选, 调试用)
# ===========================================================
ALL_TOOLS = [get_current_time, calculate, send_email]


def show_tool_schemas():
    """打印所有工具的自动生成 schema"""
    print("\n[Registered Tools]")
    print("-" * 60)
    for t in ALL_TOOLS:
        print(f"  Name:   {t.name}")
        print(f"  Desc:   {t.description[:80]}")
        schema = t.args_schema.model_json_schema()
        print(f"  Schema: {json.dumps(schema, indent=4, ensure_ascii=False)}")
        print("-" * 60)


# ===========================================================
# 第三步: 绑定工具到 LLM, 构建 Agent 循环
# ===========================================================
# 工具名 -> 工具对象映射 (用于执行)
TOOL_MAPS = {t.name: t for t in ALL_TOOLS}

# 绑定工具到 LangChain LLM
enhanced_langchain_client = langchain_client.bind_tools(ALL_TOOLS)


def agent_loop(user_input, max_rounds=5):
    """
    Function Calling Agent 主循环

    Args:
        user_input: 用户问题
        max_rounds: 最大轮数 (默认 5)
    """
    print(f"[User] {user_input}")
    messages = [
        SystemMessage(
            content=(
                "你是一个拥有使用工具权限的助手。\n"
                "当你需要真实数据或计算时, 使用工具来获取结果。\n"
                "必要时可以一次使用多个工具。\n"
                "当没有匹配的工具时, 直接回答问题。"
            )
        ),
        HumanMessage(content=user_input),
    ]
    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        # LLM 推理
        response = enhanced_langchain_client.invoke(messages)
        messages.append(response)
        # 情况 1: 没有工具调用 -> 最终回答
        if not response.tool_calls:
            print(f"[最终回答] {response.content}")
            return response.content
        # 情况 2: 执行工具调用
        for tool in response.tool_calls:
            tool_name = tool["name"]
            tool_args = tool["args"]
            logger.info(
                f"[工具调用] {tool_name} ({json.dumps(tool_args, ensure_ascii=False)})"
            )
            # 执行工具
            tool_func = TOOL_MAPS.get(tool_name)
            if tool_func:
                result = tool_func.invoke(tool_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            print(f"  [工具结果] {str(result)[:200]}")
            # 将结果回传给 LLM
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool["id"],
                )
            )
    return "达到最大尝试次数"


if __name__ == "__main__":
    # 查看自动生成的 schema
    show_tool_schemas()
    # 测试 1: 数学计算
    print(f"\n{'='*60}")
    agent_loop("表达式sqrt(256) + pow(3, 4)的值为?")
    print(f"{'='*60}")
    # 测试 2: 获取时间
    agent_loop("What time is it now?")
    print(f"{'='*60}")
    # 测试 3: 多工具调用
    agent_loop(
        "计算表达式abs(-2) * sqrt (25)的值, "
        "然后将结果使用邮件发送给link@example.com, 标题为'表达式的值'"
    )
    print(f"\n{'='*60}")
    # 测试 4: 无工具调用
    agent_loop("搜索有关AI Agent架构的知识库")
    print(f"\n{'='*60}")
    agent_loop("你好, 你是谁?")
    print(f"\n{'='*60}")
