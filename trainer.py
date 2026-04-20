import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score
)

# 确保你的 utils.py 中有这个函数
from utils import save_metrics_to_excel


class ImprovedDrugSynergyTrainer:
    """
    高级药物协同性训练器 (重构对齐版)
    包含：大模型差异化学习率、两阶段微调、基于 MCC 的动态阈值搜索、以及学习率衰减。
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: Any,
            val_loader: Any,
            test_loader: Any,
            device: torch.device,
            early_stopping_patience: int = 25,
            freeze_qwen_epochs: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 损失函数: 由于输出是单维度，使用 BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()

        # 状态追踪
        self.best_val_thresh = 0.5
        self.early_stopping_patience = early_stopping_patience
        self.best_val_mcc = -1.0
        self.patience_counter = 0
        self.best_epoch = 0
        self.freeze_qwen_epochs = freeze_qwen_epochs

        # 1. 拆分参数以实现差异化学习率 (Differential Learning Rates)
        self.qwen_params = []
        self.new_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'qwen' in name:
                self.qwen_params.append(param)
            else:
                self.new_params.append(param)

        # 2. 初始化优化器 (Qwen 学习率 5e-6, GCN/Proj 学习率 1e-4)
        self.optimizer = torch.optim.AdamW([
            {'params': self.qwen_params, 'lr': 5e-6},
            {'params': self.new_params, 'lr': 1e-4}
        ], weight_decay=0.01)

        # 3. 学习率调度器 (基于验证集 Loss 衰减)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )

    def _freeze_qwen(self) -> None:
        """冻结 Qwen 参数"""
        for name, param in self.model.named_parameters():
            if 'qwen' in name:
                param.requires_grad_(False)

    def _unfreeze_qwen(self) -> None:
        """解冻 Qwen 参数"""
        for name, param in self.model.named_parameters():
            if 'qwen' in name:
                param.requires_grad_(True)

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """处理 Batch 数据，包含 SMILES 的动态 Tokenization 及设备迁移"""
        processed = {}

        # 动态将 SMILES 转换为 Token
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

        # 其他张量移至 GPU/CPU
        for k, v in batch.items():
            if k not in ['drug1_smiles', 'drug2_smiles']:
                processed[k] = v.to(self.device) if hasattr(v, 'to') else v

        return processed

    def _compute_metrics(self, all_labels: np.ndarray, all_probs: np.ndarray, threshold: float) -> Tuple:
        """计算全套机器学习评估指标"""
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

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """执行单轮训练循环"""
        # 两阶段训练逻辑
        if epoch <= self.freeze_qwen_epochs:
            self._freeze_qwen()
            if epoch == 1:
                print(f"[*] Qwen frozen. Training GCN/Projection for {self.freeze_qwen_epochs} epochs.")
        elif epoch == self.freeze_qwen_epochs + 1:
            self._unfreeze_qwen()
            print(f"[*] Epoch {epoch}: Unfreezing Qwen parameters. Starting end-to-end fine-tuning.")

        self.model.train()
        total_loss = 0
        all_labels, all_probs = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            batch = self._move_batch(batch)

            self.optimizer.zero_grad()
            # 模型输出形状为 (batch_size, 1)，需 squeeze 以匹配 labels 的形状 (batch_size,)
            logits = self.model(batch).squeeze(-1)
            loss = self.criterion(logits, batch['labels'].float())
            loss.backward()

            # 梯度裁剪防爆炸
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

        # 计算训练指标 (使用验证集选出的最优阈值)
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

    def evaluate(self, dataloader: Any, epoch: Any, phase: str = "Validation") -> Dict[str, float]:
        """执行验证或测试循环，并搜寻最佳 MCC 阈值"""
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

        # 动态阈值搜索：仅在验证集上执行，寻找使 MCC 最大的阈值
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

    def train(self, num_epochs: int) -> None:
        """主训练调度循环"""
        print("Starting training process...")
        epoch_metrics_list = []
        save_path = "best_drug_synergy_model.pth"

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} Train phase completed. Time elapsed: {time.time() - start_time:.1f}s")

            val_metrics = self.evaluate(self.val_loader, epoch, phase="Validation")

            # 学习率调度器监控验证集 Loss
            self.scheduler.step(val_metrics['Loss'])

            combined = {'Epoch': epoch}
            metric_keys = ['Loss', 'ACC', 'F1', 'PREC', 'Recall', 'AUROC', 'AUPRC', 'MCC', 'KAPPA']
            for k in metric_keys:
                combined[f'Train_{k}'] = train_metrics.get(k, 0.0)
                combined[f'Val_{k}'] = val_metrics.get(k, 0.0)

            epoch_metrics_list.append(combined)

            # Early Stopping 与权重保存 (以验证集 MCC 为准)
            current_mcc = val_metrics['MCC']
            if current_mcc > self.best_val_mcc:
                self.best_val_mcc = current_mcc
                self.best_epoch = epoch
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"[*] New best model saved at epoch {epoch} with Val MCC: {self.best_val_mcc:.4f}")
            else:
                self.patience_counter += 1
                print(f"[*] No MCC improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"[*] Early stopping triggered at epoch {epoch}")
                    break

            # 释放显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "=" * 50)
        print(f"Training finished! Loading best model (Epoch {self.best_epoch}) for final testing...")
        print("=" * 50)

        # 测试前加载验证集 MCC 最佳的模型权重
        self.model.load_state_dict(torch.load(save_path))

        test_metrics = self.evaluate(self.test_loader, epoch="Final", phase="Test")
        clean_test_metrics = {k: v for k, v in test_metrics.items() if k not in ['Epoch', 'Phase']}

        print("Saving formatted metrics to Excel...")
        save_metrics_to_excel(epoch_metrics_list, clean_test_metrics, filename='training_metrics.xlsx')
        print(f"[*] Final model remains saved at: {save_path}")
        print("All processes completed.")

    def evaluate_and_save_details(self, dataloader: Any, output_csv: str = "test_predictions_detail.csv") -> None:
        """运行推理并将详细预测结果保存为 CSV"""
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