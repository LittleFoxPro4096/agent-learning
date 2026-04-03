# ==========================================================
# 01_prompt_engineering.py
# ==========================================================
from config import chat


# ==========================================================
# 示例1: System Prompt 定义 Agent 角色
# ==========================================================
def role_based_agent(user_query):
    """一个被严格约束的 SQL 助手 Agent"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个 SQL 专家助手。\n"
                "规则:\n"
                "1. 只回答与 SQL 相关的问题\n"
                "2. 如果用户问非 SQL 问题, 礼貌拒绝\n"
                "3. 回答格式必须包含: [分析] -> [SQL] -> [解释] 三个部分\n"
                "4. 使用标准 SQL 语法 (兼容 PostgreSQL)\n"
                "5. 不要编造不存在的函数"
            ),
        },
        {"role": "user", "content": user_query},
    ]
    response = chat(messages, temperature=0.3)
    return response.content


# ==========================================================
# 示例2: Few-Shot + 结构化输出
# ==========================================================
def sentiment_analyzer(text):
    """情感分析 Agent, 强制 JSON 输出"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是情感分析引擎。分析用户输入的文本情感。\n"
                "必须严格返回 JSON 格式, 不要返回任何其他内容:\n"
                '{"sentiment": "positive|negative|neutral", '
                '"confidence": 0.0-1.0, '
                '"keywords": ["关键词1", "关键词2"]}'
            ),
        },
        # Few-Shot 示例
        {
            "role": "user",
            "content": "这家餐厅的菜太难吃了, 再也不来了",
        },
        {
            "role": "assistant",
            "content": '{"sentiment": "negative", "confidence": 0.95, "keywords": ["难吃", "再也不来"]}',
        },
        {
            "role": "user",
            "content": "今天天气还行吧, 没什么特别的",
        },
        {
            "role": "assistant",
            "content": '{"sentiment": "neutral", "confidence": 0.8, "keywords": ["还行", "没什么特别"]}',
        },
        # 真正的用户输入
        {"role": "user", "content": text},
    ]
    response = chat(messages, temperature=0.1)
    return response.content


# ==========================================================
# 示例3: Chain-of-Thought 推理
# ==========================================================
def math_reasoning_agent(problem):
    """数学推理 Agent, 使用 CoT"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是数学推理助手。\n"
                "解题时必须遵循以下格式:\n\n"
                "[理解题意]\n(用自己的话复述问题)\n\n"
                "[逐步推理]\n"
                "步骤1: ...\n步骤2: ...\n...\n\n"
                "[最终答案]\n答案: xxx\n\n"
                "注意: 每一步都要写出计算过程, 不要跳步。"
            ),
        },
        {"role": "user", "content": problem},
    ]
    response = chat(messages, temperature=0.2)
    return response.content


# ==========================================================
# 示例4: Self-Reflection (自我反思/自我纠错)
# ==========================================================
def self_reflecting_agent(question):
    """带自我反思能力的 Agent"""
    # 第一轮: 初始回答
    messages = [
        {
            "role": "system",
            "content": "你是一个知识助手, 请回答用户的问题。",
        },
        {"role": "user", "content": question},
    ]
    first_answer = chat(messages, temperature=0.7)

    # 第二轮: 自我审查
    messages.append({"role": "assistant", "content": first_answer.content})
    messages.append(
        {
            "role": "user",
            "content": (
                "请审查你刚才的回答:\n"
                "1. 有没有事实性错误?\n"
                "2. 有没有逻辑漏洞?\n"
                "3. 有没有遗漏重要信息?\n\n"
                "如果有问题, 请给出修正后的完整回答。\n"
                "如果没有问题, 请说明为什么你认为回答是正确的。"
            ),
        },
    )
    refined_answer = chat(messages, temperature=0.3)

    return {
        "initial_answer": first_answer.content,
        "refined_answer": refined_answer.content,
    }


# ==========================================================
# 运行测试
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("[1] SQL Agent")
    print(role_based_agent("查询 orders 表中每个用户的总消费金额, 只要前10名"))
    print(role_based_agent("帮我计算10MB等于多少bit"))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("[2] Sentiment Analyzer")
    print(sentiment_analyzer("这个产品质量非常好, 物流也很快, 五星好评!"))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("[3] Math CoT")
    print(
        math_reasoning_agent(
            "一个水池有两个水管, A管每小时注水3吨, B管每小时排水1吨。水池容量20吨, 初始状态是3吨, 问需要多久能装满?"
        )
    )
    print("=" * 60)

    print("\n" + "=" * 60)
    print("[4] Self-Reflection")
    result = self_reflecting_agent("为什么叶子大部分是绿色的?")
    print("--- 初始回答 ---")
    print(result["initial_answer"])
    print("--- 反思修正 ---")
    print(result["refined_answer"])
    print("=" * 60)
