import re
import openai
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder


class RAGSystem:
    """完整的RAG检索增强生成系统"""
    
    def __init__(self, chroma_db_path="./chroma_db", collection_name="abyssinian_cat_vet_info"):
        """初始化RAG系统"""
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # 定义嵌入模型
        self.embedding_func = embedding_functions.OpenAIEmbeddingFunction(
            api_key="hk-e6y54310000462160a54b4d0097e29e44d3ec90d2e06988e",
            model_name="text-embedding-3-small",
            api_base="https://api.openai-hk.com/v1",
        )
        
        # 尝试获取集合，如果不存在则创建
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_func
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_func,
                metadata={"description": "阿比西尼亚猫医疗护理信息分块"}
            )
        
        # 初始化重排序模型
        self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
        
        # 配置OpenAI客户端
        openai.api_key = "hk-e6y54310000462160a54b4d0097e29e44d3ec90d2e06988e"
        openai.api_base = "https://api.openai-hk.com/v1"
    
    def split_document(self, content: str) -> List[str]:
        """按一级标题分割文档为多块"""
        # 匹配### 数字. 格式的标题作为分割点
        blocks = re.split(r'(?=### \d+\.)', content.strip())
        # 过滤空内容并保留标题
        return [block for block in blocks if block.strip()]
    
    def save_to_chroma(self, blocks: List[str]) -> None:
        """将分块内容存储到Chroma向量数据库"""
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
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"成功将{len(blocks)}块内容存入Chroma数据库")
        print(f"集合名称: {self.collection_name}")
        print(f"存储路径: {self.chroma_db_path}")
    
    def process_document(self, file_path: str) -> None:
        """处理文档：读取、分割、存储到向量数据库"""
        print(f"📖 正在处理文档: {file_path}")
        
        # 读取文档内容
        with open(file_path, "r", encoding="utf-8") as f:
            doc_content = f.read()
        
        # 分块处理
        split_blocks = self.split_document(doc_content)
        print(f"文档分割完成，共{len(split_blocks)}块")
        
        # 存入Chroma
        self.save_to_chroma(split_blocks)
        print("✅ 文档处理完成！")
    
    def retrieve_documents(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """从向量数据库检索相关文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # 格式化检索结果
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            doc = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """使用重排序模型对检索到的文档进行重新排序"""
        # 准备查询-文档对
        query_doc_pairs = [(query, doc['content']) for doc in documents]
        
        # 计算重排序分数
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # 为文档添加重排序分数
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(rerank_scores[i])
        
        # 按重排序分数降序排列
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs[:top_k]
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """基于检索到的文档生成答案"""
        # 构建上下文
        context = "\n\n".join([f"文档{i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])
        
        # 构建提示词
        prompt = f"""基于以下关于阿比西尼亚猫的医疗护理信息，请回答用户的问题。

上下文信息：
{context}

用户问题：{query}

请基于上述信息提供准确、详细的回答。如果信息不足以回答问题，请说明需要更多信息。

回答："""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的宠物医疗顾问，专门回答关于阿比西尼亚猫的健康和医疗问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"生成答案时出错：{str(e)}"
    
    def query(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        print(f"\n🔍 用户问题: {question}")
        
        # 1. 检索相关文档
        print(f"\n📚 正在检索相关文档（top-{retrieve_k}）...")
        retrieved_docs = self.retrieve_documents(question, retrieve_k)
        print(f"检索到 {len(retrieved_docs)} 个相关文档")
        
        # 2. 重排序
        print(f"\n🔄 正在使用重排序模型重新排序（top-{rerank_k}）...")
        reranked_docs = self.rerank_documents(question, retrieved_docs, rerank_k)
        print(f"重排序完成，选择前 {len(reranked_docs)} 个最相关文档")
        
        # 3. 生成答案
        print(f"\n🤖 正在生成答案...")
        answer = self.generate_answer(question, reranked_docs)
        
        # 返回完整结果
        result = {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'reranked_docs': reranked_docs,
            'context_used': len(reranked_docs)
        }
        
        return result
    
    def print_result(self, result: Dict[str, Any]):
        """格式化打印查询结果"""
        print("\n" + "="*80)
        print(f"📋 问题: {result['question']}")
        print("\n" + "-"*80)
        print(f"💡 答案:\n{result['answer']}")
        print("\n" + "-"*80)
        print(f"📖 使用的上下文文档 ({result['context_used']}个):")
        
        for i, doc in enumerate(result['reranked_docs'], 1):
            print(f"\n{i}. {doc['metadata']['title']} (重排序分数: {doc['rerank_score']:.4f})")
            print(f"   内容预览: {doc['content'][:100]}...")
        
        print("\n" + "="*80)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'db_path': self.chroma_db_path
        }


def demo_rag_system():
    """演示RAG系统的使用"""
    print("🚀 初始化RAG系统...")
    rag = RAGSystem()
    
    # 检查集合信息
    info = rag.get_collection_info()
    print(f"📊 集合信息: {info['document_count']} 个文档")
    
    # 如果集合为空，提示用户处理文档
    if info['document_count'] == 0:
        print("⚠️ 数据库为空，请先使用 process_document() 方法处理文档")
        print("示例: rag.process_document('/path/to/your/document.md')")
        return
    
    # 示例查询
    test_questions = [
        "阿比西尼亚猫容易患什么疾病？",
        "如何预防阿比西尼亚猫的遗传性疾病？",
        "阿比西尼亚猫的日常护理需要注意什么？",
        "阿比西尼亚猫的饮食有什么特殊要求吗？"
    ]
    
    for question in test_questions:
        result = rag.query(question)
        rag.print_result(result)
        print("\n" + "#"*100 + "\n")


if __name__ == "__main__":
    # 演示RAG系统
    demo_rag_system()
    
    # 如果需要处理新文档，可以使用以下代码：
    # rag = RAGSystem()
    # rag.process_document("/Users/zhangyf/Documents/宠物数据/宠物垂直领域大模型数据源-中文/健康与医疗-中文/阿比西尼亚猫.md")