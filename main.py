import torch
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from model import QwenEnhancedDrugSynergyModel
from trainer import ImprovedDrugSynergyTrainer
from dataset import DrugSynergyDataset
from data_processor import DrugCellDataProcessor
from utils import collate_fn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    gcn_config = {
        'in_feats': 64,
        'hidden_size': 256,
        'out_feats': 256
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing data processor...")
    processor = DrugCellDataProcessor(
        'merged_drug_data_complete.csv',
        'Drug_Target_Protein.csv',
        'cell_data_clustered_mlp.csv'
    )

    print("Loading dataset...")
    full_dataset_train = DrugSynergyDataset('balanced_synergy_data.csv', processor, augment=True)
    full_dataset_eval  = DrugSynergyDataset('balanced_synergy_data.csv', processor, augment=False)

    total_size = len(full_dataset_eval)
    labels = full_dataset_eval.data['classification'].apply(
        lambda x: 1 if 'synergy' in str(x).lower() else 0
    ).tolist()
    indices = list(range(total_size))

    train_val_idx, test_idx, train_val_labels, _ = train_test_split(
        indices, labels, test_size=0.1, stratify=labels, random_state=42
    )

    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, train_val_labels, test_size=1/9, stratify=train_val_labels, random_state=42
    )

    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset   = Subset(full_dataset_eval,  val_idx)
    test_dataset  = Subset(full_dataset_eval,  test_idx)

    batch_size = 128  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True)

    print("Initializing Qwen Enhanced Drug Synergy Model...")
    model = QwenEnhancedDrugSynergyModel(
        gcn_config=gcn_config,
        num_classes=1, 
        qwen_model_name=QWEN_MODEL_NAME,
        target_dim=processor.target_dim,
        cell_dim=processor.cell_dim,
        physchem_dim=processor.physchem_dim
    )

    trainer = ImprovedDrugSynergyTrainer(
        model, train_loader, val_loader, test_loader, device,
        early_stopping_patience=25,   
        freeze_qwen_epochs=10          
    )

    print("Commencing training execution...")
    trainer.train(num_epochs=100)
    trainer.evaluate_and_save_details(test_loader, output_csv="final_test_predictions.csv")

if __name__ == '__main__':
    main()