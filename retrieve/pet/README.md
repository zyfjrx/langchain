# 阿比西尼亚猫医疗护理RAG系统

这是一个基于检索增强生成(RAG)技术的智能问答系统，专门用于回答关于阿比西尼亚猫的健康和医疗护理问题。

## 系统特性

- **文档分割与向量化**: 自动将医疗文档按章节分割并存储到ChromaDB向量数据库
- **语义检索**: 使用OpenAI的text-embedding-3-small模型进行语义相似度检索
- **重排序优化**: 集成BAAI/bge-reranker-base重排序模型，提高检索精度
- **智能问答**: 基于GPT-3.5-turbo生成专业的医疗护理建议

## 系统架构

```
用户问题 → 向量检索 → 重排序模型 → 上下文构建 → LLM生成答案
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 初始化数据库（首次使用）

如果需要重新处理文档数据，请在`rag.py`中取消注释相关代码：

```python
# 取消注释以下代码来重新处理文档
with open("/path/to/阿比西尼亚猫.md", "r", encoding="utf-8") as f:
    doc_content = f.read()
split_blocks = split_document(doc_content)
if len(split_blocks) == 6:
    save_to_chroma(split_blocks)
```

### 2. 运行演示程序

```bash
python rag.py
```

### 3. 交互式问答

```bash
python example_usage.py
```

### 4. 快速测试

```bash
python example_usage.py --test
```

## 代码结构

### 核心类：RAGSystem

```python
class RAGSystem:
    def __init__(self, chroma_db_path="./chroma_db", collection_name="abyssinian_cat_vet_info")
    def retrieve_documents(self, query: str, n_results: int = 10)
    def rerank_documents(self, query: str, documents: List[Dict], top_k: int = 5)
    def generate_answer(self, query: str, context_docs: List[Dict])
    def query(self, question: str, retrieve_k: int = 10, rerank_k: int = 5)
```

### 主要功能函数

- `split_document()`: 文档分割
- `save_to_chroma()`: 向量化存储
- `demo_rag_system()`: 系统演示

## 配置说明

### API配置

系统使用OpenAI API进行嵌入和文本生成，配置信息：

```python
api_key = "hk-e6y54310000462160a54b4d0097e29e44d3ec90d2e06988e"
api_base = "https://api.openai-hk.com/v1"
embedding_model = "text-embedding-3-small"
chat_model = "gpt-3.5-turbo"
```

### 重排序模型

```python
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
```

## 使用示例

```python
from rag import RAGSystem

# 初始化系统
rag = RAGSystem()

# 提问
question = "阿比西尼亚猫容易患什么疾病？"
result = rag.query(question)

# 显示结果
rag.print_result(result)
```

## 示例问题

- "阿比西尼亚猫容易患什么疾病？"
- "如何预防阿比西尼亚猫的遗传性疾病？"
- "阿比西尼亚猫的日常护理需要注意什么？"
- "阿比西尼亚猫的饮食有什么特殊要求吗？"

## 文件说明

- `rag.py`: 主要的RAG系统实现
- `example_usage.py`: 使用示例和交互式问答
- `requirements.txt`: 依赖包列表
- `chroma_db/`: ChromaDB数据库文件夹
- `README.md`: 本说明文件

## 技术栈

- **向量数据库**: ChromaDB
- **嵌入模型**: OpenAI text-embedding-3-small
- **重排序模型**: BAAI/bge-reranker-base
- **生成模型**: GPT-3.5-turbo
- **框架**: sentence-transformers, openai

## 注意事项

1. 首次运行需要下载重排序模型，可能需要一些时间
2. 确保网络连接正常，以便访问OpenAI API
3. 如果遇到API限制，可以调整请求频率
4. 重排序模型会占用一定的GPU/CPU资源

## 性能优化建议

- 调整`retrieve_k`和`rerank_k`参数来平衡精度和速度
- 可以缓存重排序模型的结果来提高响应速度
- 对于大量查询，建议使用批处理模式