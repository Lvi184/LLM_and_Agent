from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1. 加载预训练模型（先从轻量模型入手）
# 推荐新手先试：all-MiniLM-L6-v2（轻量、速度快）
model = SentenceTransformer('all-MiniLM-L6-v2')

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
sim_0_1 = util.cos_sim(embeddings[0], embeddings[1])
# 计算第0句和第2句的相似度（语义无关）
sim_0_2 = util.cos_sim(embeddings[0], embeddings[2])

print(f"句子0和句子1的相似度：{sim_0_1.item():.4f}")  # 约0.8+（高相似）
print(f"句子0和句子2的相似度：{sim_0_2.item():.4f}")  # 约0.1-（低相似）