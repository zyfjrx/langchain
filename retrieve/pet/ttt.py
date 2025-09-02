import re

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


def split_document(content):
    """按一级标题分割文档为6块"""
    import re
    # 匹配### 数字. 格式的标题作为分割点
    blocks = re.split(r'(?=### \d+\.)', content.strip())
    # 过滤空内容并保留标题
    return [block for block in blocks if block.strip()]


def save_to_chroma(blocks):
    """将分块内容存储到Chroma向量数据库"""
    # 初始化Chroma客户端（持久化存储到本地）
    client = chromadb.PersistentClient(path="./chroma_db")

    # 定义嵌入模型（使用中文支持较好的模型）
    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key="hk-e6y54310000462160a54b4d0097e29e44d3ec90d2e06988e",
        model_name="text-embedding-3-small",
        api_base="https://api.openai-hk.com/v1",
    )

    # 创建或获取集合
    collection = client.get_or_create_collection(
        name="abyssinian_cat_vet_info",
        embedding_function=embedding_func,
        metadata={"description": "阿比西尼亚猫医疗护理信息分块"}
    )

    # 为每个块准备数据
    documents = []
    metadatas = []
    ids = []

    for i, block in enumerate(blocks, 1):
        # 提取标题作为元数据
        title_match = re.search(r'### (\d+\. .+)', block)
        title = title_match.group(1) if title_match else f"第{i}块内容"

        documents.append(block)
        metadatas.append({"block_id": i, "title": title})
        ids.append(f"block_{i}")

    # 添加到向量数据库
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"成功将{len(blocks)}块内容存入Chroma数据库")
    print(f"集合名称: abyssinian_cat_vet_info")
    print(f"存储路径: ./chroma_db")


if __name__ == "__main__":
    # 读取文档内容（这里直接使用提供的完整内容）
    with open("/Users/zhangyf/Documents/宠物数据/宠物垂直领域大模型数据源-中文/健康与医疗-中文/阿比西尼亚猫.md", "r", encoding="utf-8") as f:
        doc_content = f.read()

    # 分块处理
    split_blocks = split_document(doc_content)
    print(f"文档分割完成，共{len(split_blocks)}块")

    # 存入Chroma
    if len(split_blocks) == 6:  # 确认是预期的6块
        save_to_chroma(split_blocks)
    else:
        print(f"分割结果不符合预期（{len(split_blocks)}块），请检查文档格式")