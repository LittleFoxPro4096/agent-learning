# ==========================================================
# config.py
# 全局统一配置: LLM 客户端 / Embedding 客户端 / LangChain LLM
# ==========================================================

import logging
from os import getenv
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI

# ==========================================================
# 加载环境变量文件
# ==========================================================
load_dotenv()


# ==========================================================
# 日志配置
# ==========================================================
def get_logger():
    log_level = getattr(logging, getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_format = getenv(
        "LOG_FORMAT", "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    logging.basicConfig(level=log_level, format=log_format)
    return logging.getLogger(__name__)


logger = get_logger()


# ==========================================================
# 环境变量验证
# ==========================================================
def required_env(key: str) -> str:
    value = getenv(key)
    if not value:
        logger.error("Missing required environment variable: %s", key)
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


# ==========================================================
# LLM 配置
# ==========================================================
LLM_BASE_URL = required_env("LLM_BASE_URL")
LLM_API_KEY = required_env("LLM_API_KEY")
LLM_MODEL = required_env("LLM_MODEL")

# ==========================================================
# Embedding 配置
# ==========================================================
EMBED_BASE_URL = required_env("EMBED_BASE_URL")
EMBED_API_KEY = required_env("EMBED_API_KEY")
EMBED_MODEL = required_env("EMBED_MODEL")


# ==========================================================
# 获取客户端实例
# ==========================================================
def get_llm_client():
    logger.info("初始化 LLM 客户端, base_url=%s", LLM_BASE_URL)
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return llm_client


def get_langchain_client():
    logger.info("初始化 LangChain 客户端, model=%s", LLM_MODEL)
    langchain_client = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.3,
    )
    return langchain_client


def get_embed_client():
    logger.info("初始化 Embedding 客户端, base_url=%s", EMBED_BASE_URL)
    embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)
    return embed_client


# ==========================================================
# 统一调用封装
# ==========================================================
def chat(messages, temperature=0.7, max_tokens=65536):
    """
    调用 LLM 进行聊天/推理 (不含工具调用)
    用于: Prompt Engineering, RAG 生成, Multi-Agent 对话
    """
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message


def embed(text):
    """
    调用 Embedding 模型进行向量化
    用于: RAG 文档索引和查询
    """
    is_batch = isinstance(text, list)
    input_text = text if is_batch else [text]
    response = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=input_text,
    )
    if is_batch:
        return [item.embedding for item in response.data]
    return response.data[0].embedding


# ==========================================================
# 客户端实例
# ==========================================================
llm_client = get_llm_client()
langchain_client = get_langchain_client()
embed_client = get_embed_client()
