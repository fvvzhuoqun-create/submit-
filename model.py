import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any


class DrugGAT(nn.Module):
    """
    药物分子图特征提取器 (基于 GATv2)
    """

    def __init__(self, in_feats: int, hidden_size: int, out_feats: int, dropout: float = 0.5, heads: int = 4):
        super().__init__()

        # --- 网络层定义 ---
        self.conv1 = GATv2Conv(in_feats, hidden_size, heads=heads, dropout=dropout, concat=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * heads)

        self.conv2 = GATv2Conv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout, concat=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size * heads)

        self.conv3 = GATv2Conv(hidden_size * heads, out_feats, heads=1, concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """图注意力网络的前向传播"""
        # Block 1
        h = self.conv1(x, edge_index)
        h = self.batch_norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.35, training=self.training)

        # Block 2
        h = self.conv2(h, edge_index)
        h = self.batch_norm2(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.35, training=self.training)

        # Output Block
        out = self.conv3(h, edge_index)
        return out


class QwenEnhancedDrugSynergyModel(nn.Module):
    """
    Qwen 增强的多模态药物协同性预测模型
    """

    def __init__(
            self,
            gcn_config: Dict[str, Any],
            num_classes: int = 1,
            target_dim: int = 560,
            cell_dim: int = 64,
            physchem_dim: int = 7,
            qwen_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    ):
        super().__init__()

        # 1. 图神经网络编码器
        self.gcn_drug1 = DrugGAT(**gcn_config)
        self.gcn_drug2 = DrugGAT(**gcn_config)

        # 2. Qwen 大语言模型初始化
        self.qwen = AutoModel.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
        self._configure_llm()

        # 3. 模态对齐投影层 (将各类特征映射到 LLM 的隐藏层维度)
        q_hid = self.qwen.config.hidden_size

        self.proj_gcn = nn.Linear(gcn_config['out_feats'], q_hid, dtype=torch.bfloat16)
        self.proj_target = nn.Linear(1024, q_hid, dtype=torch.bfloat16)
        self.proj_physchem = nn.Linear(physchem_dim, q_hid, dtype=torch.bfloat16)
        self.proj_cell = nn.Linear(cell_dim, q_hid, dtype=torch.bfloat16)

        # 4. 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(q_hid, 256, dtype=torch.bfloat16),
            nn.LayerNorm(256, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, dtype=torch.bfloat16)
        )

    def _configure_llm(self) -> None:
        """私有方法：配置 Qwen 的状态、缓存和梯度冻结策略"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.qwen.config.use_cache = False
        self.qwen.gradient_checkpointing_enable()
        self.qwen.enable_input_require_grads()

        # 冻结所有参数
        for param in self.qwen.parameters():
            param.requires_grad = False

        # 解冻最后 150 个参数以进行微调
        for param in list(self.qwen.parameters())[-150:]:
            param.requires_grad = True

    def _get_smiles_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """私有方法：提取并平均池化 SMILES 字符串的 Embedding"""
        embeds = self.qwen.get_input_embeddings()(input_ids)
        mean_token = embeds.mean(dim=1)
        return mean_token.unsqueeze(1)

    def forward(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """模型前向传播"""
        device = next(self.parameters()).device

        # --- 第一步: 提取分子图特征并池化 ---
        g1_features = self.gcn_drug1(batch_data['graph1'].x, batch_data['graph1'].edge_index)
        d1_pooled = global_mean_pool(g1_features, batch_data['graph1'].batch)

        g2_features = self.gcn_drug2(batch_data['graph2'].x, batch_data['graph2'].edge_index)
        d2_pooled = global_mean_pool(g2_features, batch_data['graph2'].batch)

        # --- 第二步: 统一数据精度 (bfloat16) ---
        d1 = d1_pooled.to(torch.bfloat16)
        d2 = d2_pooled.to(torch.bfloat16)
        t1 = batch_data['target1'].to(torch.bfloat16)
        t2 = batch_data['target2'].to(torch.bfloat16)
        p1 = batch_data['physchem1'].to(torch.bfloat16)
        p2 = batch_data['physchem2'].to(torch.bfloat16)
        ce = batch_data['cell_expr'].to(torch.bfloat16)

        # --- 第三步: 将特征投影为 Soft Tokens ---
        soft_tokens = torch.stack([
            self.proj_gcn(d1),
            self.proj_gcn(d2),
            self.proj_target(t1),
            self.proj_target(t2),
            self.proj_physchem(p1),
            self.proj_physchem(p2),
            self.proj_cell(ce)
        ], dim=1)

        # --- 第四步: 提取文本 (SMILES) Tokens ---
        s1_token = self._get_smiles_token(batch_data['drug1_input_ids'])
        s2_token = self._get_smiles_token(batch_data['drug2_input_ids'])

        # --- 第五步: 拼接所有序列特征 ---
        full_embeds = torch.cat([soft_tokens, s1_token, s2_token], dim=1)

        # --- 第六步: 创建注意力掩码并送入大模型 ---
        batch_size = full_embeds.shape[0]
        seq_length = full_embeds.shape[1]
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)

        llm_outputs = self.qwen(inputs_embeds=full_embeds, attention_mask=attention_mask)

        # --- 第七步: 提取最后一个 Token 进行分类预测 ---
        last_tokens = llm_outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(last_tokens)

        # 为兼容 Loss 计算，输出前转回 float32
        return logits.to(torch.float32)