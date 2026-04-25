import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel, AutoTokenizer


class DrugGAT(nn.Module):

    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5, heads=4):
        super(DrugGAT, self).__init__()
        self.conv1 = GATv2Conv(in_feats, hidden_size, heads=heads, dropout=dropout, concat=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * heads)
        self.conv2 = GATv2Conv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout, concat=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size * heads)
        self.conv3 = GATv2Conv(hidden_size * heads, out_feats, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(F.elu(self.batch_norm1(self.conv1(x, edge_index))), p=0.35, training=self.training)
        x = F.dropout(F.elu(self.batch_norm2(self.conv2(x, edge_index))), p=0.35, training=self.training)
        return self.conv3(x, edge_index)


class QwenEnhancedDrugSynergyModel(nn.Module):

    def __init__(self, gcn_config, num_classes=1, target_dim=560, cell_dim=64, physchem_dim=7,
                 qwen_model_name="Qwen/Qwen2.5-3B-Instruct"):
        super().__init__()
        self.gcn_drug1 = DrugGAT(**gcn_config)
        self.gcn_drug2 = DrugGAT(**gcn_config)

        self.qwen = AutoModel.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.qwen.config.use_cache = False
        self.qwen.gradient_checkpointing_enable()
        self.qwen.enable_input_require_grads()

        for param in self.qwen.parameters():
            param.requires_grad = False
        for param in list(self.qwen.parameters())[-150:]:
            param.requires_grad = True

        q_hid = self.qwen.config.hidden_size

        self.proj_gcn = nn.Linear(gcn_config['out_feats'], q_hid, dtype=torch.bfloat16)
        self.proj_target = nn.Linear(1024, q_hid, dtype=torch.bfloat16)
        self.proj_physchem = nn.Linear(physchem_dim, q_hid, dtype=torch.bfloat16)
        self.proj_cell = nn.Linear(cell_dim, q_hid, dtype=torch.bfloat16)

        self.classifier = nn.Sequential(
            nn.Linear(q_hid, 256, dtype=torch.bfloat16),
            nn.LayerNorm(256, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, dtype=torch.bfloat16)
        )

    def forward(self, batch_data):
        device = next(self.parameters()).device

        d1 = global_mean_pool(self.gcn_drug1(batch_data['graph1'].x, batch_data['graph1'].edge_index),
                              batch_data['graph1'].batch)
        d2 = global_mean_pool(self.gcn_drug2(batch_data['graph2'].x, batch_data['graph2'].edge_index),
                              batch_data['graph2'].batch)

        d1 = d1.to(torch.bfloat16)
        d2 = d2.to(torch.bfloat16)

        t1 = batch_data['target1'].to(torch.bfloat16)
        t2 = batch_data['target2'].to(torch.bfloat16)
        p1 = batch_data['physchem1'].to(torch.bfloat16)
        p2 = batch_data['physchem2'].to(torch.bfloat16)

        ce = batch_data['cell_expr'].to(torch.bfloat16)

        soft_tokens = torch.stack([
            self.proj_gcn(d1), self.proj_gcn(d2),
            self.proj_target(t1), self.proj_target(t2),
            self.proj_physchem(p1), self.proj_physchem(p2),
            self.proj_cell(ce)
        ], dim=1)

        def get_smiles_token(input_ids):
            embeds = self.qwen.get_input_embeddings()(input_ids)
            token = embeds.mean(dim=1)
            return token.unsqueeze(1)

        s1_token = get_smiles_token(batch_data['drug1_input_ids'])
        s2_token = get_smiles_token(batch_data['drug2_input_ids'])

        full_embeds = torch.cat([soft_tokens, s1_token, s2_token], dim=1)

        batch_size = full_embeds.shape[0]
        full_attention_mask = torch.ones((batch_size, 9), dtype=torch.long, device=device)

        outputs = self.qwen(inputs_embeds=full_embeds, attention_mask=full_attention_mask).last_hidden_state

        last_tokens = outputs[:, -1, :]
        return self.classifier(last_tokens).to(torch.float32)