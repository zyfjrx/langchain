#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统使用示例

这个文件展示了如何使用已实现的RAG系统进行问答。
"""

from rag import RAGSystem

def main():
    """主函数：演示RAG系统的基本使用"""
    
    # 初始化RAG系统
    print("🚀 正在初始化RAG系统...")
    rag = RAGSystem()
    print("✅ RAG系统初始化完成！")
    
    # 交互式问答
    print("\n" + "="*60)
    print("🐱 阿比西尼亚猫医疗护理问答系统")
    print("输入 'quit' 或 'exit' 退出程序")
    print("="*60)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n❓ 请输入您的问题: ").strip()
            
            # 检查退出条件
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 感谢使用，再见！")
                break
            
            if not question:
                print("⚠️ 请输入有效的问题")
                continue
            
            # 执行RAG查询
            result = rag.query(question, retrieve_k=8, rerank_k=3)
            
            # 显示结果
            rag.print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            print("请检查系统配置或重试")

def quick_test():
    """快速测试函数"""
    print("🧪 执行快速测试...")
    
    rag = RAGSystem()
    
    test_question = "阿比西尼亚猫有哪些常见的健康问题？"
    result = rag.query(test_question)
    rag.print_result(result)
    
    print("\n✅ 快速测试完成！")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 快速测试模式
        quick_test()
    else:
        # 交互模式
        main()