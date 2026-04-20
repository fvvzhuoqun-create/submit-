import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import os


def generate_protein_embeddings(
        csv_path='Drug_Target_Protein.csv',
        save_path='target_protein_embeddings.pt',
        model_name="facebook/esm2_t12_35M_UR50D"  # 默认使用轻量级35M参数模型，可改为更大模型
):
    """
    读取靶点序列，使用 ESM-2 模型生成蛋白质 Embedding，并保存。
    """
    print(f"正在加载数据集: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 去重：提取唯一的靶点。
    # 建议使用 uniprot_id 作为主键，因为它比 target_name 更标准、唯一
    unique_targets = df[['uniprot_id', 'target_name', 'sequence']].drop_duplicates(subset=['uniprot_id']).reset_index(
        drop=True)
    print(f"总记录数: {len(df)}, 唯一靶点蛋白质数量: {len(unique_targets)}")

    # 2. 设置计算设备 (优先使用 GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 3. 加载 HuggingFace 上的 ESM-2 模型和分词器
    print(f"正在加载 ESM 模型: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()  # 设为评估模式
    model.to(device)

    # 创建一个字典来存储特征 {uniprot_id: embedding_tensor}
    protein_embeddings_dict = {}

    print("开始提取蛋白质特征...")
    # 使用 tqdm 显示进度条
    with torch.no_grad():  # 不计算梯度，节省显存
        for idx, row in tqdm(unique_targets.iterrows(), total=len(unique_targets)):
            uid = row['uniprot_id']
            seq = row['sequence']

            # ESM 模型默认最大序列长度通常为 1024。
            # 如果序列超长，此处进行截断 (truncation=True) 防止 OOM 报错
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 前向传播
            outputs = model(**inputs)

            # 获取最后一层隐藏状态 (shape: [batch_size, sequence_length, hidden_dim])
            last_hidden_state = outputs.last_hidden_state

            # 进行 Mean Pooling (平均池化)，将整个序列融合成一个定长向量
            # dim=1 表示在 sequence_length 维度上求平均
            # 最终 sequence_embedding shape: [hidden_dim] (对于 35M 模型，dim=480)
            sequence_embedding = last_hidden_state.mean(dim=1).squeeze()

            # 转移到 CPU 并转为 float32 节省存储空间
            protein_embeddings_dict[uid] = sequence_embedding.cpu().to(torch.float32)

    # 4. 保存字典到本地
    torch.save(protein_embeddings_dict, save_path)
    print(f"✅ 特征提取完成！已保存至 {save_path}")

    # 打印一个样本看看维度
    sample_emb = list(protein_embeddings_dict.values())[0]
    print(f"生成的 Embedding 维度为: {sample_emb.shape}")


if __name__ == "__main__":
    # 你可以根据显存大小选择更大的 ESM 模型以获得更好的生物学表征：
    # "facebook/esm2_t12_35M_UR50D"  -> 隐藏层维度 480 (极快，适合测试代码)
    # "facebook/esm2_t30_150M_UR50D" -> 隐藏层维度 640 (推荐，平衡)
    # "facebook/esm2_t33_650M_UR50D" -> 隐藏层维度 1280 (性能更好，需要较大显存)

    generate_protein_embeddings(
        csv_path='Drug_Target_Protein.csv',
        save_path='target_protein_embeddings.pt',
        model_name="facebook/esm2_t30_150M_UR50D"  # 这里我默认推荐 150M 的版本
    )