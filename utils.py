import pandas as pd
import torch
from torch_geometric.data import Data, Batch

def save_metrics_to_excel(epoch_metrics, test_metrics, filename='training_metrics.xlsx'):

    df_epochs = pd.DataFrame(epoch_metrics)
    df_test = pd.DataFrame([test_metrics])
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_epochs.to_excel(writer, sheet_name='Train_Val_Log', index=False)
        df_test.to_excel(writer, sheet_name='Final_Test_Result', index=False)
        
        workbook = writer.book
        
        format_float = workbook.add_format({'num_format': '0.0000', 'align': 'center'})
        format_center = workbook.add_format({'align': 'center'})
        format_header = workbook.add_format({
            'bold': True, 'align': 'center', 'bg_color': '#D7E4BC', 'border': 1
        })
        
        worksheet_train = writer.sheets['Train_Val_Log']
        worksheet_train.set_column('A:A', 10, format_center)  
        worksheet_train.set_column('B:Z', 13, format_float)   
        worksheet_train.freeze_panes(1, 0)                   
        
        
        for col_num, value in enumerate(df_epochs.columns.values):
            worksheet_train.write(0, col_num, value, format_header)
  
        worksheet_test = writer.sheets['Final_Test_Result']
        worksheet_test.set_column('A:Z', 15, format_float)
        for col_num, value in enumerate(df_test.columns.values):
            worksheet_test.write(0, col_num, value, format_header)
    print(f"Metrics saved to {filename}")

def collate_fn(batch):
    graph1_list, graph2_list = [], []
    for sample in batch:
        e1, x1 = sample['graph1']
        e2, x2 = sample['graph2']
        graph1_list.append(Data(x=x1, edge_index=e1))
        graph2_list.append(Data(x=x2, edge_index=e2))

    res = {
        'graph1': Batch.from_data_list(graph1_list),
        'graph2': Batch.from_data_list(graph2_list),
        'target1': torch.stack([s['target1'] for s in batch]),
        'target2': torch.stack([s['target2'] for s in batch]),
        'physchem1': torch.stack([s['physchem1'] for s in batch]),
        'physchem2': torch.stack([s['physchem2'] for s in batch]),
        'cell_expr': torch.stack([s['cell_expr'] for s in batch]),
        'labels': torch.stack([s['labels'] for s in batch]),
        'drug1_smiles': [s['drug1_smiles'] for s in batch],
        'drug2_smiles': [s['drug2_smiles'] for s in batch]
    }
    return res

def create_safe_collate_fn(processor):
    def safe_collate_fn(batch):
        valid_samples = []
        for sample in batch:
            try:
                if sample['graph1'][0].dim() == 2 and sample['target1'].dim() == 1:
                    valid_samples.append(sample)
            except Exception:
                continue
        if len(valid_samples) == 0:
            return collate_fn([processor._create_default_sample()])
        return collate_fn(valid_samples)
    return safe_collate_fn