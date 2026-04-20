import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Dict, Any


class DrugSynergyDataset(Dataset):
    """
    药物协同性 PyTorch 数据集类
    """

    def __init__(self, synergy_df: pd.DataFrame, processor: Any, tokenizer: Any, max_length: int = 128):
        self.data = synergy_df.reset_index(drop=True)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]

        # 1. 基础特征处理
        sample = self.processor.process_sample(
            row['drug1'], row['drug2'], row['cell_line'], augment=True
        )

        # 2. 文本特征处理 (SMILES Tokenization)
        token_params = {
            "padding": 'max_length',
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": 'pt'
        }

        d1_ids = self.tokenizer(sample['drug1_smiles'], **token_params).input_ids.squeeze(0)
        d2_ids = self.tokenizer(sample['drug2_smiles'], **token_params).input_ids.squeeze(0)

        # 3. 构造返回字典
        return {
            'graph1': Data(x=sample['graph1'][1], edge_index=sample['graph1'][0]),
            'graph2': Data(x=sample['graph2'][1], edge_index=sample['graph2'][0]),
            'target1': sample['target1'],
            'target2': sample['target2'],
            'physchem1': sample['physchem1'],
            'physchem2': sample['physchem2'],
            'cell_expr': sample['cell_expr'],
            'drug1_input_ids': d1_ids,
            'drug2_input_ids': d2_ids,
            'label': torch.tensor([row['label']], dtype=torch.float32)
        }