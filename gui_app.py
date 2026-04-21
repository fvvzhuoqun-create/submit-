import os
import sys

os.environ['PYTORCH_JIT'] = '0'
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

import torch
from torch_geometric.data import Data, Batch
from huggingface_hub import snapshot_download

from model import QwenEnhancedDrugSynergyModel
from data_processor import DrugCellDataProcessor


def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容 PyInstaller 打包后的路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class AdvancedSynergyPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("精准医疗：联合用药协同药效体外分析系统 ")
        self.root.geometry("650x600")
        self.root.configure(bg="#f8f9fa")

        # 字体统一定义
        self.title_font = ("Microsoft YaHei", 12, "bold")
        self.label_font = ("Microsoft YaHei", 10)

        self.status_var = tk.StringVar(value="正在初始化系统与特征库，请稍候...")

        # 构建主界面的标签页
        self.build_tabs()

        # 禁用预测按钮，直到资源加载完成
        self.predict_btn.config(state=tk.DISABLED)
        self.batch_start_btn.config(state=tk.DISABLED)

        # 启动后台线程加载大模型与数据集
        threading.Thread(target=self.load_resources, daemon=True).start()

    def build_tabs(self):
        # 顶部状态栏
        status_label = tk.Label(self.root, textvariable=self.status_var, fg="#0056b3", bg="#f8f9fa",
                                font=("Microsoft YaHei", 9))
        status_label.pack(fill="x", pady=5)

        # 创建 Notebook (标签页控制器)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=15, pady=5)

        # 创建三个标签页
        self.tab_single = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_single, text=" 🔬 单次精准评估 ")
        self.notebook.add(self.tab_batch, text=" 📊 高通量批量筛选 ")
        self.notebook.add(self.tab_history, text=" 📁 评估历史记录 ")

        self.build_single_tab()
        self.build_batch_tab()
        self.build_history_tab()

        # 底部免责声明
        disclaimer_text = "临床声明：本系统基于多模态大模型生成，结果仅供体外实验参考，不可直接作为临床用药指导。"
        tk.Label(self.root, text=disclaimer_text, fg="#6c757d", bg="#f8f9fa", font=("Microsoft YaHei", 8)).pack(
            side="bottom", pady=5)

    # ---------------- 标签页 1：单次精准评估 ----------------
    def build_single_tab(self):
        input_frame = tk.LabelFrame(self.tab_single, text=" 实验设计与参数配置 (支持文本检索) ", font=self.title_font,
                                    padx=15, pady=15)
        input_frame.pack(fill="x", padx=15, pady=15)

        # 细胞与药物输入
        tk.Label(input_frame, text="细胞系：", font=self.label_font).grid(row=0, column=0, sticky="w", pady=5)
        self.cell_entry = ttk.Combobox(input_frame, width=45)
        self.cell_entry.grid(row=0, column=1, pady=5, padx=10)

        tk.Label(input_frame, text="候选药物 A：", font=self.label_font).grid(row=1, column=0, sticky="w", pady=5)
        self.drug1_entry = ttk.Combobox(input_frame, width=45)
        self.drug1_entry.grid(row=1, column=1, pady=5, padx=10)

        tk.Label(input_frame, text="候选药物 B：", font=self.label_font).grid(row=2, column=0, sticky="w", pady=5)
        self.drug2_entry = ttk.Combobox(input_frame, width=45)
        self.drug2_entry.grid(row=2, column=1, pady=5, padx=10)

        self.predict_btn = tk.Button(self.tab_single, text="⚕️ 运行计算生物学药效评估", command=self.on_predict_click,
                                     bg="#007bff", fg="white", font=("Microsoft YaHei", 11, "bold"), width=25,
                                     relief="flat", cursor="hand2")
        self.predict_btn.pack(pady=10)

        # 结果展示区
        self.result_frame = tk.LabelFrame(self.tab_single, text=" 联合药效评估报告 ", font=self.title_font, padx=15,
                                          pady=15)
        self.result_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.result_label = tk.Label(self.result_frame, text="等待录入参数...", font=("Microsoft YaHei", 12),
                                     justify="center")
        self.result_label.pack(pady=10)

    # ---------------- 标签页 2：高通量批量筛选 ----------------
    def build_batch_tab(self):
        batch_frame = tk.Frame(self.tab_batch, padx=20, pady=20)
        batch_frame.pack(fill="both", expand=True)

        tk.Label(batch_frame, text="上传包含 Drug1, Drug2, Cell_Line 列的 CSV/Excel", font=self.title_font).pack(
            pady=(10, 20))

        file_frame = tk.Frame(batch_frame)
        file_frame.pack(fill="x", pady=10)
        self.file_path_var = tk.StringVar(value="未选择文件...")
        tk.Entry(file_frame, textvariable=self.file_path_var, width=50, state="readonly").pack(side="left",
                                                                                               padx=(0, 10))
        tk.Button(file_frame, text="📂 浏览文件", command=self.select_file, bg="#6c757d", fg="white").pack(side="left")

        # 进度条区
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(batch_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=20)
        self.progress_label = tk.Label(batch_frame, text="进度: 0%", font=self.label_font)
        self.progress_label.pack()

        btn_frame = tk.Frame(batch_frame)
        btn_frame.pack(pady=20)
        self.batch_start_btn = tk.Button(btn_frame, text="🚀 开始批量测算", command=self.on_batch_click, bg="#28a745",
                                         fg="white", font=self.title_font, width=15)
        self.batch_start_btn.pack(side="left", padx=10)

    # ---------------- 标签页 3：评估历史记录 ----------------
    def build_history_tab(self):
        hist_frame = tk.Frame(self.tab_history, padx=15, pady=15)
        hist_frame.pack(fill="both", expand=True)

        columns = ("Date", "Cell", "Drug A", "Drug B", "Score", "Suggestion")
        self.tree = ttk.Treeview(hist_frame, columns=columns, show="headings", height=15)

        self.tree.heading("Date", text="时间")
        self.tree.heading("Cell", text="细胞系")
        self.tree.heading("Drug A", text="药物 A")
        self.tree.heading("Drug B", text="药物 B")
        self.tree.heading("Score", text="协同概率")
        self.tree.heading("Suggestion", text="临床建议")

        self.tree.column("Date", width=80)
        self.tree.column("Cell", width=100)
        self.tree.column("Drug A", width=100)
        self.tree.column("Drug B", width=100)
        self.tree.column("Score", width=80, anchor="center")
        self.tree.column("Suggestion", width=120)

        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        btn_frame = tk.Frame(hist_frame)
        btn_frame.pack(fill="x", pady=10)
        tk.Button(btn_frame, text="🗑️ 清空历史", command=self.clear_history, fg="red").pack(side="left")

    # ---------------- 核心逻辑区 ----------------
    def check_and_download_base_model(self):
        """检查并下载基础模型（按需下载方案）"""
        model_dir = get_resource_path("models/Qwen2.5-3B-Instruct")
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            self.root.after(0, lambda: self.status_var.set(
                "首次运行，未检测到基础大模型，正在自动拉取 (需联网且耗时较长)..."))
            try:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                os.makedirs(model_dir, exist_ok=True)
                snapshot_download(
                    repo_id="Qwen/Qwen2.5-3B-Instruct",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                self.root.after(0, lambda: self.status_var.set("大模型拉取完成！"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("网络错误", f"模型下载失败: {e}"))
                raise

    def load_resources(self):
        """加载数据和构建模型网络"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.check_and_download_base_model()

            self.status_var.set(f"正在加载靶点数据库与细胞组学特征... (计算设备: {self.device})")
            self.processor = DrugCellDataProcessor(
                drug_data_file=get_resource_path('merged_drug_data_complete.csv'),
                drug_target_file=get_resource_path('Drug_Target_Protein.csv'),
                cell_line_file=get_resource_path('cell_data_clustered_mlp.csv'),
                target_features_file=get_resource_path('target_features.csv')
            )

            # 获取所有合法名称供下拉列表和校验使用
            self.valid_drugs = list(self.processor.drug_smiles_map.keys())
            self.valid_cells = self.processor.cell_line_expr.index.tolist()

            self.root.after(0, lambda: self.drug1_entry.config(values=self.valid_drugs))
            self.root.after(0, lambda: self.drug2_entry.config(values=self.valid_drugs))
            self.root.after(0, lambda: self.cell_entry.config(values=self.valid_cells))

            self.status_var.set("数据加载完毕。正在构建 Qwen 融合图神经网络结构...")

            gcn_config = {'in_feats': 64, 'hidden_size': 256, 'out_feats': 256}
            self.model = QwenEnhancedDrugSynergyModel(
                gcn_config=gcn_config, num_classes=1,
                qwen_model_name=get_resource_path("models/Qwen2.5-3B-Instruct"),
                target_dim=self.processor.target_dim, cell_dim=self.processor.cell_dim,
                physchem_dim=self.processor.physchem_dim
            )

            model_path = get_resource_path('best_drug_synergy_model.pth')
            if not os.path.exists(model_path):
                model_path = get_resource_path('best_drug_synergy_model_light.pth')

            if os.path.exists(model_path):
                self.status_var.set("正在导入自定义微调药效权重...")
                # strict=False 允许只加载微调的那几层参数
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            else:
                self.root.after(0,
                                lambda: messagebox.showwarning("警告", "未找到微调权重文件，将使用初始大模型权重运行！"))

            self.model.to(self.device)
            self.model.eval()

            self.status_var.set("系统就绪，特征空间对齐完毕。请输入您的实验参数。")
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.batch_start_btn.config(state=tk.NORMAL))

        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("初始化失败，请查阅报错弹窗。"))
            self.root.after(0, lambda: messagebox.showerror("初始化错误", str(e)))

    def on_predict_click(self):
        """单次预测按钮点击事件，包含合法性校验"""
        cell = self.cell_entry.get().strip()
        drug1 = self.drug1_entry.get().strip()
        drug2 = self.drug2_entry.get().strip()

        # 检查是否留空
        if not all([cell, drug1, drug2]):
            messagebox.showwarning("提示", "请完整填写细胞系、候选药物 A 和 候选药物 B！")
            return

        # 增加的需求：检查数据库中是否存在该输入项
        if cell not in self.valid_cells:
            messagebox.showwarning("提示", "抱歉，当前程序数据库中并未有该细胞系数据，敬请期待！")
            return
        if drug1 not in self.valid_drugs:
            messagebox.showwarning("提示", f"抱歉，当前程序数据库中并未有药物 [{drug1}] 数据，敬请期待！")
            return
        if drug2 not in self.valid_drugs:
            messagebox.showwarning("提示", f"抱歉，当前程序数据库中并未有药物 [{drug2}] 数据，敬请期待！")
            return

        self.predict_btn.config(state=tk.DISABLED)
        self.result_label.config(text="正在进行药物分子与细胞靶点模拟计算...", fg="#17a2b8")

        threading.Thread(target=self.run_inference, args=(drug1, drug2, cell), daemon=True).start()

    def run_inference(self, drug1, drug2, cell):
        """调用真实的深度学习预测并渲染UI"""
        try:
            sample = self.processor.process_sample(drug1, drug2, cell, augment=False)

            graph1 = Data(x=sample['graph1'][1], edge_index=sample['graph1'][0])
            graph2 = Data(x=sample['graph2'][1], edge_index=sample['graph2'][0])

            tokenizer = self.model.tokenizer
            d1_tokens = tokenizer([sample['drug1_smiles']], return_tensors='pt', padding=True,
                                  truncation=True).input_ids
            d2_tokens = tokenizer([sample['drug2_smiles']], return_tensors='pt', padding=True,
                                  truncation=True).input_ids

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

            with torch.no_grad():
                logits = self.model(batch_data)
                prob = torch.sigmoid(logits).item()

            # 三级医疗评估体系
            if prob >= 0.8:
                res_text = f"【极高潜力】具有显著的协同杀伤效应\n协同概率: {prob * 100:.2f}%\n💡 建议：优先开展体外验证实验。"
                color, sug = "#28a745", "优先验证"
            elif prob >= 0.5:
                res_text = f"【潜在组合】表现出弱协同或相加效应\n协同概率: {prob * 100:.2f}%\n💡 建议：可作为备选实验方案。"
                color, sug = "#fd7e14", "备选方案"
            else:
                res_text = f"【风险提示】存在拮抗或无效叠加可能\n协同概率: {prob * 100:.2f}%\n💡 建议：不推荐作为联合用药首选，暂缓规划。"
                color, sug = "#dc3545", "暂缓规划"

            self.root.after(0, lambda: self.result_label.config(text=res_text, fg=color))

            # 自动将结果加入历史记录表格
            current_time = time.strftime("%H:%M:%S")
            self.root.after(0, lambda: self.tree.insert("", "end",
                                                        values=(current_time, cell, drug1, drug2, f"{prob * 100:.2f}%",
                                                                sug)))

        except Exception as e:
            self.root.after(0, lambda: self.result_label.config(text="预测发生错误", fg="red"))
            self.root.after(0, lambda: messagebox.showerror("推理报错", str(e)))
        finally:
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

    # ---------------- 批处理逻辑 ----------------
    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if filepath:
            self.file_path_var.set(filepath)

    def on_batch_click(self):
        path = self.file_path_var.get()
        if not os.path.exists(path):
            messagebox.showwarning("提示", "请先选择合法的待测算文件！")
            return
        self.batch_start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_batch_inference, args=(path,), daemon=True).start()

    def run_batch_inference(self, filepath):
        """真实的批处理运算框架"""
        try:
            df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
            req_cols = ['Drug1', 'Drug2', 'Cell_Line']
            if not all(col in df.columns for col in req_cols):
                raise ValueError("上传的文件必须包含表头: 'Drug1', 'Drug2', 'Cell_Line'")

            total = len(df)
            results = []

            for idx, row in df.iterrows():
                drug1, drug2, cell = str(row['Drug1']), str(row['Drug2']), str(row['Cell_Line'])

                # 校验是否存在于库中
                if cell in self.valid_cells and drug1 in self.valid_drugs and drug2 in self.valid_drugs:
                    # 复用单次推理的数据处理逻辑，提取特征
                    sample = self.processor.process_sample(drug1, drug2, cell, augment=False)

                    graph1 = Data(x=sample['graph1'][1], edge_index=sample['graph1'][0])
                    graph2 = Data(x=sample['graph2'][1], edge_index=sample['graph2'][0])
                    tokenizer = self.model.tokenizer
                    d1_tok = tokenizer([sample['drug1_smiles']], return_tensors='pt', padding=True,
                                       truncation=True).input_ids
                    d2_tok = tokenizer([sample['drug2_smiles']], return_tensors='pt', padding=True,
                                       truncation=True).input_ids

                    batch_data = {
                        'graph1': Batch.from_data_list([graph1]).to(self.device),
                        'graph2': Batch.from_data_list([graph2]).to(self.device),
                        'target1': sample['target1'].unsqueeze(0).to(self.device),
                        'target2': sample['target2'].unsqueeze(0).to(self.device),
                        'physchem1': sample['physchem1'].unsqueeze(0).to(self.device),
                        'physchem2': sample['physchem2'].unsqueeze(0).to(self.device),
                        'cell_expr': sample['cell_expr'].unsqueeze(0).to(self.device),
                        'drug1_input_ids': d1_tok.to(self.device),
                        'drug2_input_ids': d2_tok.to(self.device)
                    }

                    with torch.no_grad():
                        prob = torch.sigmoid(self.model(batch_data)).item()
                else:
                    prob = "库中无对应数据"

                results.append(prob)

                # 更新进度条
                progress = int(((idx + 1) / total) * 100)
                self.root.after(0, self.progress_var.set, progress)
                self.root.after(0, self.progress_label.config, {"text": f"测算进度: {progress}% ({idx + 1}/{total})"})

            # 将结果拼回并保存
            df['Predicted_Prob'] = results
            out_name = filepath.replace('.csv', '_results.csv').replace('.xlsx', '_results.csv')
            df.to_csv(out_name, index=False, encoding='utf-8-sig')

            self.root.after(0, lambda: self.progress_label.config(
                text=f"✅ 批量测算完成！报告已自动保存至原文件夹下:\n{os.path.basename(out_name)}", fg="#28a745"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("批处理错误", f"批处理失败: {e}"))
            self.root.after(0, lambda: self.progress_label.config(text="批处理因错误中止。", fg="red"))
        finally:
            self.root.after(0, lambda: self.batch_start_btn.config(state=tk.NORMAL))

    def clear_history(self):
        for item in self.tree.get_children():
            self.tree.delete(item)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedSynergyPredictorApp(root)
    root.mainloop()