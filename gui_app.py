import os
import sys
os.environ['PYTORCH_JIT'] = '0'
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import torch
from torch_geometric.data import Data, Batch

from model import QwenEnhancedDrugSynergyModel
from data_processor import DrugCellDataProcessor


def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容 PyInstaller 打包后的路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class SynergyPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen-Enhanced 药物协同性预测系统")
        self.root.geometry("500x450")
        self.root.configure(padx=20, pady=20)

        # 界面状态标签
        self.status_var = tk.StringVar(value="正在初始化系统，请稍候... (大模型加载可能需要一两分钟)")

        # 构建 UI (此时下拉框为空，等待数据加载完成)
        self.build_ui()

        # 禁用预测按钮直到加载完成
        self.predict_btn.config(state=tk.DISABLED)

        # 使用子线程加载模型和数据，防止界面卡死
        threading.Thread(target=self.load_resources, daemon=True).start()

    def build_ui(self):
        # 状态显示
        tk.Label(self.root, textvariable=self.status_var, fg="blue", wraplength=450).pack(pady=5)

        # 细胞系选择
        tk.Label(self.root, text="请选择或输入细胞系 (Cell Line):", font=("Arial", 10, "bold")).pack(pady=5, anchor="w")
        self.cell_entry = ttk.Combobox(self.root, width=50)
        self.cell_entry.pack()

        # 药物1选择
        tk.Label(self.root, text="请选择或输入药物 1 (Drug 1):", font=("Arial", 10, "bold")).pack(pady=5, anchor="w")
        self.drug1_entry = ttk.Combobox(self.root, width=50)
        self.drug1_entry.pack()

        # 药物2选择
        tk.Label(self.root, text="请选择或输入药物 2 (Drug 2):", font=("Arial", 10, "bold")).pack(pady=5, anchor="w")
        self.drug2_entry = ttk.Combobox(self.root, width=50)
        self.drug2_entry.pack()

        # 预测按钮
        self.predict_btn = tk.Button(self.root, text="🚀 开始协同性预测", command=self.on_predict_click,
                                     bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20, height=2)
        self.predict_btn.pack(pady=25)

        # 结果显示区
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0", bd=2, relief="groove")
        self.result_frame.pack(fill="x", pady=10, ipady=10)
        self.result_label = tk.Label(self.result_frame, text="等待预测...", font=("Arial", 14), bg="#f0f0f0")
        self.result_label.pack()

    def load_resources(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.status_var.set(f"正在加载数据集... (使用设备: {self.device})")

            # 加载数据处理器
            self.processor = DrugCellDataProcessor(
                drug_data_file=get_resource_path('merged_drug_data_complete.csv'),
                drug_target_file=get_resource_path('Drug_Target_Protein.csv'),
                cell_line_file=get_resource_path('cell_data_clustered_mlp.csv'),
                target_features_file=get_resource_path('target_features.csv')
            )

            # 更新 UI 中的下拉列表数据
            drug_list = list(self.processor.drug_smiles_map.keys())
            cell_list = self.processor.cell_line_expr.index.tolist()

            self.root.after(0, lambda: self.drug1_entry.config(values=drug_list))
            self.root.after(0, lambda: self.drug2_entry.config(values=drug_list))
            self.root.after(0, lambda: self.cell_entry.config(values=cell_list))

            self.status_var.set("数据集加载完成！正在加载 Qwen 协同预测模型 (这可能需要较长时间)...")

            #  初始化模型
            gcn_config = {'in_feats': 64, 'hidden_size': 256, 'out_feats': 256}
            self.model = QwenEnhancedDrugSynergyModel(
                gcn_config=gcn_config,
                num_classes=1,
                qwen_model_name=get_resource_path("models/Qwen2.5-3B-Instruct"),
                target_dim=self.processor.target_dim,
                cell_dim=self.processor.cell_dim,
                physchem_dim=self.processor.physchem_dim
            )

            # 加载训练好的权重
            model_path = get_resource_path('best_drug_synergy_model.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                self.root.after(0, lambda: messagebox.showwarning("警告",
                                                                  f"未找到模型权重文件: {model_path}，将使用未微调的初始权重运行！"))

            self.model.to(self.device)
            self.model.eval()

            self.status_var.set("系统就绪！请输入参数进行预测。")
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

        except Exception as e:
            error_msg = f"初始化失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set("初始化失败！请检查文件。"))
            self.root.after(0, lambda: messagebox.showerror("严重错误", error_msg))

    def on_predict_click(self):
        cell = self.cell_entry.get().strip()
        drug1 = self.drug1_entry.get().strip()
        drug2 = self.drug2_entry.get().strip()

        if not all([cell, drug1, drug2]):
            messagebox.showwarning("提示", "请完整填写细胞系、药物1 和 药物2！")
            return

        self.predict_btn.config(state=tk.DISABLED)
        self.result_label.config(text="正在推理运算中...", fg="orange")

        # 启动后台线程预测，避免卡顿 UI
        threading.Thread(target=self.run_inference, args=(drug1, drug2, cell), daemon=True).start()

    def run_inference(self, drug1, drug2, cell):
        try:
            # 使用的 data_processor 提取特征
            sample = self.processor.process_sample(drug1, drug2, cell, augment=False)

            # 将图数据封装为 torch_geometric 期望的 Batch 格式
            graph1 = Data(x=sample['graph1'][1], edge_index=sample['graph1'][0])
            graph2 = Data(x=sample['graph2'][1], edge_index=sample['graph2'][0])

            # 将 SMILES 转化为 Qwen 期望的 input_ids
            tokenizer = self.model.tokenizer
            d1_tokens = tokenizer([sample['drug1_smiles']], return_tensors='pt', padding=True,
                                  truncation=True).input_ids
            d2_tokens = tokenizer([sample['drug2_smiles']], return_tensors='pt', padding=True,
                                  truncation=True).input_ids

            # 构建模型前向传播所需的 batch_data 字典
            batch_data = {
                'graph1': Batch.from_data_list([graph1]).to(self.device),
                'graph2': Batch.from_data_list([graph2]).to(self.device),
                'target1': sample['target1'].unsqueeze(0).to(self.device),
                'target2': sample['target2'].unsqueeze(0).to(self.device),
                'physchem1': sample['physchem1'].unsqueeze(0).to(self.device),
                'physchem2': sample['physchem2'].unsqueeze(0).to(self.device),
                'cell_expr': sample['cell_expr'].unsqueeze(0).to(self.device),
                'drug1_input_ids': d1_tokens.to(self.device),
                'drug2_input_ids': d2_tokens.to(self.device)
            }

            # 执行推理
            with torch.no_grad():
                logits = self.model(batch_data)
                # 由于你的模型是 num_classes=1，这里使用 sigmoid 转化为概率 (0-1)
                prob = torch.sigmoid(logits).item()

            # 显示结果
            if prob >= 0.5:
                res_text = f"预测结果: 具有协同性\n协同概率: {prob * 100:.2f}%"
                color = "green"
            else:
                res_text = f"预测结果: 无协同性 (拮抗/相加)\n协同概率: {prob * 100:.2f}%"
                color = "red"

            self.root.after(0, lambda: self.result_label.config(text=res_text, fg=color))

        except Exception as e:
            self.root.after(0, lambda: self.result_label.config(text="预测发生错误", fg="red"))
            self.root.after(0, lambda: messagebox.showerror("推理报错", str(e)))
        finally:
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))


if __name__ == "__main__":
    root = tk.Tk()
    app = SynergyPredictorApp(root)
    root.mainloop()