from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1. 加载预训练模型（先从轻量模型入手）
# 推荐新手先试：all-MiniLM-L6-v2（轻量、速度快）
model = SentenceTransformer('all-MiniLM-L6-v2')
# 默认下载路径 C:\Users\【你的电脑用户名】\.cache\huggingface\hub
# 编码维度
# all-MiniLM-L6-v2：384 维（轻量模型，速度快、体积小，最推荐新手）
# all-MiniLM-L12-v2：384 维（和上面同维度，特征提取稍细，速度稍慢）
# all-mpnet-base-v2：768 维（中量级模型，语义表征更精准，体积约 400MB）
# paraphrase-multilingual-MiniLM-L12-v2：384 维（多语言模型，支持中英日韩等，新手做跨语言相似度首选）
# 2. 准备测试文本
texts = [
    "今天天气很好，适合出门散步",
    "今日晴空万里，出门遛弯很合适",
    "人工智能技术正在快速发展",
    "机器学习是人工智能的重要分支"
]

# 3. 生成文本向量（返回numpy数组，shape=(文本数, 向量维度)）
embeddings = model.encode(texts)
print(f"向量维度：{embeddings.shape}")  # all-MiniLM-L6-v2输出384维，输出示例：(4, 384)

# 4. 计算向量相似度（余弦相似度）
# 计算第0句和第1句的相似度（语义相似）
print(embeddings[0])
sim_0_1 = util.cos_sim(embeddings[0], embeddings[1])
# 计算第0句和第2句的相似度（语义无关）
sim_0_2 = util.cos_sim(embeddings[0], embeddings[2])

print(f"句子0和句子1的相似度：{sim_0_1.item():.4f}")  # 约0.8+（高相似）
print(f"句子0和句子2的相似度：{sim_0_2.item():.4f}")  # 约0.1-（低相似）