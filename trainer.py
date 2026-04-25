import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score
)

from utils import save_metrics_to_excel


class ImprovedDrugSynergyTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device,
                 early_stopping_patience=25, freeze_qwen_epochs=10):

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.best_val_thresh = 0.5
        self.early_stopping_patience = early_stopping_patience
        self.best_val_mcc = -1.0
        self.patience_counter = 0
        self.best_epoch = 0

        self.freeze_qwen_epochs = freeze_qwen_epochs

        self.qwen_params = []
        self.new_params = []

        for name, param in model.named_parameters():
            if 'qwen' in name:
                self.qwen_params.append(param)
            else:
                self.new_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': self.qwen_params, 'lr': 5e-6},
            {'params': self.new_params, 'lr': 1e-4}
        ], weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )

    def _freeze_qwen(self):
        for name, param in self.model.named_parameters():
            if 'qwen' in name:
                param.requires_grad_(False)

    def _unfreeze_qwen(self):
        for name, param in self.model.named_parameters():
            if 'qwen' in name:
                if any(f".layers.{i}." in name for i in range(24, 36)):
                    param.requires_grad_(True)
                elif "norm" in name and "layers" not in name:
                    param.requires_grad_(True)

    def _move_batch(self, batch):
        processed = {}
        if 'drug1_smiles' in batch and 'drug2_smiles' in batch:
            tok1 = self.model.tokenizer(
                batch['drug1_smiles'], return_tensors="pt",
                padding="max_length", truncation=True, max_length=64
            )
            tok2 = self.model.tokenizer(
                batch['drug2_smiles'], return_tensors="pt",
                padding="max_length", truncation=True, max_length=64
            )
            processed['drug1_input_ids'] = tok1.input_ids.to(self.device)
            processed['drug2_input_ids'] = tok2.input_ids.to(self.device)

        for k, v in batch.items():
            if k not in ['drug1_smiles', 'drug2_smiles']:
                processed[k] = v.to(self.device) if hasattr(v, 'to') else v

        return processed

    def _compute_metrics(self, all_labels, all_probs, threshold):
        preds = (all_probs >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0
        auprc = average_precision_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, preds)
        kappa = cohen_kappa_score(all_labels, preds)
        return acc, f1, prec, rec, auroc, auprc, mcc, kappa

    def train_epoch(self, epoch):
        if epoch <= self.freeze_qwen_epochs:
            self._freeze_qwen()
            if epoch == 1:
                print(f"[*] Qwen frozen. Training GCN/Projection for {self.freeze_qwen_epochs} epochs.")
        elif epoch == self.freeze_qwen_epochs + 1:
            self._unfreeze_qwen()
            print(f"[*] Epoch {epoch}: Unfreezing the last 12 layers of Qwen. Starting end-to-end fine-tuning.")

        self.model.train()
        total_loss = 0
        all_labels, all_probs = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            batch = self._move_batch(batch)

            self.optimizer.zero_grad()
            logits = self.model(batch).squeeze(-1)
            loss = self.criterion(logits, batch['labels'].float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        acc, f1, prec, rec, auroc, auprc, mcc, kappa = self._compute_metrics(
            all_labels, all_probs, self.best_val_thresh
        )

        print(f"\n--- Epoch {epoch} Train Results ---")
        print(f"Loss: {avg_loss:.4f} | Threshold: {self.best_val_thresh:.2f}")
        print(f"ACC: {acc:.4f} | F1: {f1:.4f} | PREC: {prec:.4f} | Recall: {rec:.4f}")
        print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f} | KAPPA: {kappa:.4f}\n")

        return {
            'Epoch': epoch, 'Phase': 'Train', 'Loss': avg_loss,
            'ACC': acc, 'F1': f1, 'PREC': prec, 'Recall': rec,
            'AUROC': auroc, 'AUPRC': auprc, 'MCC': mcc, 'KAPPA': kappa
        }

    def evaluate(self, dataloader, epoch, phase="Validation"):
        self.model.eval()
        total_loss = 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch} [{phase}]"):
                batch = self._move_batch(batch)
                logits = self.model(batch).squeeze(-1)
                loss = self.criterion(logits, batch['labels'].float())
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        if phase == "Validation":
            best_score, best_thresh = -1.0, 0.5
            for thresh in np.arange(0.1, 0.95, 0.01):
                preds = (all_probs >= thresh).astype(int)
                cur_score = matthews_corrcoef(all_labels, preds)
                if cur_score > best_score:
                    best_score, best_thresh = cur_score, thresh

            self.best_val_thresh = best_thresh
            chosen_thresh = best_thresh
            print(f"[*] Validation Phase: Optimal threshold (max MCC) = {chosen_thresh:.2f}")
        else:
            chosen_thresh = getattr(self, 'best_val_thresh', 0.5)
            print(f"[*] Test Phase: Applying validation threshold = {chosen_thresh:.2f}")

        acc, f1, prec, rec, auroc, auprc, mcc, kappa = self._compute_metrics(
            all_labels, all_probs, chosen_thresh
        )

        print(f"\n--- Epoch {epoch} {phase} Results ---")
        print(f"Loss: {avg_loss:.4f} | Threshold: {chosen_thresh:.2f}")
        print(f"ACC: {acc:.4f} | F1: {f1:.4f} | PREC: {prec:.4f} | Recall: {rec:.4f}")
        print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f} | KAPPA: {kappa:.4f}\n")

        return {
            'Epoch': epoch, 'Phase': phase, 'Loss': avg_loss,
            'ACC': acc, 'F1': f1, 'PREC': prec, 'Recall': rec,
            'AUROC': auroc, 'AUPRC': auprc, 'MCC': mcc, 'KAPPA': kappa
        }

    def train(self, num_epochs):
        print("Starting training process...")
        epoch_metrics_list = []
        save_path = "best_drug_synergy_model_light_mcc.pth"

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} Train phase completed. Time elapsed: {time.time() - start_time:.1f}s")

            val_metrics = self.evaluate(self.val_loader, epoch, phase="Validation")

            self.scheduler.step(val_metrics['Loss'])

            combined = {'Epoch': epoch}
            metric_keys = ['Loss', 'ACC', 'F1', 'PREC', 'Recall', 'AUROC', 'AUPRC', 'MCC', 'KAPPA']
            for k in metric_keys:
                combined[f'Train_{k}'] = train_metrics.get(k, 0.0)
                combined[f'Val_{k}'] = val_metrics.get(k, 0.0)

            epoch_metrics_list.append(combined)

            current_mcc = val_metrics['MCC']
            if current_mcc > self.best_val_mcc:
                self.best_val_mcc = current_mcc
                self.best_epoch = epoch
                self.patience_counter = 0
                state_dict = self.model.state_dict()
                trainable_state_dict = {
                    name: state_dict[name]
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                }
                torch.save(trainable_state_dict, save_path)

                print(f"[*] New best light model saved at epoch {epoch} with Val MCC: {self.best_val_mcc:.4f}")
            else:
                self.patience_counter += 1
                print(f"[*] No MCC improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"[*] Early stopping triggered at epoch {epoch}")
                    break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "=" * 50)
        print(f"Training finished! Loading best model (Epoch {self.best_epoch}) for final testing...")
        print("=" * 50)

        self.model.load_state_dict(torch.load(save_path), strict=False)

        test_metrics = self.evaluate(self.test_loader, epoch="Final", phase="Test")
        clean_test_metrics = {k: v for k, v in test_metrics.items() if k not in ['Epoch', 'Phase']}

        print("Saving formatted metrics to Excel...")
        save_metrics_to_excel(epoch_metrics_list, clean_test_metrics, filename='training_metrics_mcc.xlsx')

        print("\n[*] Saving detailed test predictions using the best model...")
        self.evaluate_and_save_details(self.test_loader, output_csv="final_test_predictions_detail_mcc.csv")

        print(f"[*] Final light model remains saved at: {save_path}")
        print("All processes completed.")

    def evaluate_and_save_details(self, dataloader, output_csv="test_predictions_detail.csv"):
        self.model.eval()
        all_labels, all_probs = [], []
        drug1_list, drug2_list, cell_line_list = [], [], []

        print(f"\n[*] Generating detailed predictions...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = self._move_batch(batch)
                logits = self.model(batch).squeeze(-1)
                probs = torch.sigmoid(logits)

                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                if 'cell_line' in batch:
                    drug1_list.extend(batch['drug1_name'])
                    drug2_list.extend(batch['drug2_name'])
                    cell_line_list.extend(batch['cell_line'])

        chosen_thresh = getattr(self, 'best_val_thresh', 0.5)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        final_preds = (all_probs >= chosen_thresh).astype(int)

        if len(cell_line_list) == len(all_labels):
            df_final = pd.DataFrame({
                'Drug1': drug1_list,
                'Drug2': drug2_list,
                'Cell_Line': cell_line_list,
                'True_Label': all_labels,
                'Predicted_Prob': np.round(all_probs, 4),
                'Predicted_Class': final_preds
            })
        else:
            df_final = pd.DataFrame({
                'True_Label': all_labels,
                'Predicted_Prob': all_probs,
                'Predicted_Class': final_preds
            })

        df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"[*] Detailed predictions saved to: {output_csv}")