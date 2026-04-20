import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as GeoDataLoader

from model import QwenEnhancedDrugSynergyModel
from data_processor import DrugCellDataProcessor
from dataset import DrugSynergyDataset
from trainer import ModelTrainer


def main():

    CONFIG = {
        'batch_size': 8,
        'epochs': 10,
        'lr': 1e-4,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'model_name': "Qwen/Qwen2.5-3B-Instruct",
        'save_path': "best_synergy_model.pth"
    }

    processor = DrugCellDataProcessor(
        drug_data_file='merged_drug_data_complete.csv',
        drug_target_file='Drug_Target_Protein.csv',
        cell_line_file='cell_data_clustered_mlp.csv',
        target_features_file='target_features.csv'
    )

    synergy_df = pd.read_csv('balanced_synergy_data.csv')
    train_df, val_df = train_test_split(synergy_df, test_size=0.2, random_state=42)

    gcn_config = {'in_feats': 64, 'hidden_size': 256, 'out_feats': 256}
    model = QwenEnhancedDrugSynergyModel(
        gcn_config=gcn_config,
        qwen_model_name=CONFIG['model_name'],
        target_dim=processor.target_dim,
        cell_dim=processor.cell_dim,
        physchem_dim=processor.physchem_dim
    ).to(CONFIG['device'])

    train_ds = DrugSynergyDataset(train_df, processor, model.tokenizer)
    val_ds = DrugSynergyDataset(val_df, processor, model.tokenizer)

    train_loader = GeoDataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=CONFIG['batch_size'])

    trainer = ModelTrainer(model, CONFIG['device'], CONFIG['lr'])
    best_auc = 0

    for epoch in range(CONFIG['epochs']):
        avg_loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Loss: {avg_loss:.4f}")
        print(f"Val Metrics: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            trainer.save_checkpoint(CONFIG['save_path'])


if __name__ == "__main__":
    main()