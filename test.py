import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import time
import threading


class AdvancedSynergyAppDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("精准医疗：联合用药协同药效体外分析系统 (Pro版)")
        self.root.geometry("650x550")
        self.root.configure(bg="#f8f9fa")

        # 字体统一定义
        self.title_font = ("Microsoft YaHei", 12, "bold")
        self.label_font = ("Microsoft YaHei", 10)

        self.status_var = tk.StringVar(value="系统就绪，特征空间对齐完毕。")

        # 构建主界面的标签页
        self.build_tabs()

        # 填充模拟数据
        self.populate_mock_data()

    def build_tabs(self):
        # 顶部状态栏
        status_label = tk.Label(self.root, textvariable=self.status_var, fg="#0056b3", bg="#f8f9fa",
                                font=("Microsoft YaHei", 9))
        status_label.pack(fill="x", pady=5)

        # 创建 Notebook (标签页控制器)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=15, pady=5)

        # 创建三个标签页框架
        self.tab_single = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_single, text=" 🔬 单次精准评估 ")
        self.notebook.add(self.tab_batch, text=" 📊 高通量批量筛选 ")
        self.notebook.add(self.tab_history, text=" 📁 评估历史记录 ")

        # 分别构建三个页面的内容
        self.build_single_tab()
        self.build_batch_tab()
        self.build_history_tab()

        # 底部免责声明
        disclaimer_text = "临床声明：本系统基于多模态生信大模型生成，结果仅供体外(In vitro)实验参考，不可直接作为临床用药指导。"
        tk.Label(self.root, text=disclaimer_text, fg="#6c757d", bg="#f8f9fa", font=("Microsoft YaHei", 8)).pack(
            side="bottom", pady=5)

    # ---------------- 标签页 1：单次精准评估 ----------------
    def build_single_tab(self):
        input_frame = tk.LabelFrame(self.tab_single, text=" 实验设计与参数配置 (支持文本模糊检索) ",
                                    font=self.title_font, padx=15, pady=15)
        input_frame.pack(fill="x", padx=15, pady=15)

        # 细胞与药物输入 (ttk.Combobox 天然支持输入，加上我们会在后台做过滤联想)
        tk.Label(input_frame, text="靶向细胞模型:", font=self.label_font).grid(row=0, column=0, sticky="w", pady=5)
        self.cell_entry = ttk.Combobox(input_frame, width=40)
        self.cell_entry.grid(row=0, column=1, pady=5, padx=10)

        tk.Label(input_frame, text="候选药物 A:", font=self.label_font).grid(row=1, column=0, sticky="w", pady=5)
        self.drug1_entry = ttk.Combobox(input_frame, width=40)
        self.drug1_entry.grid(row=1, column=1, pady=5, padx=10)

        tk.Label(input_frame, text="候选药物 B:", font=self.label_font).grid(row=2, column=0, sticky="w", pady=5)
        self.drug2_entry = ttk.Combobox(input_frame, width=40)
        self.drug2_entry.grid(row=2, column=1, pady=5, padx=10)

        self.predict_btn = tk.Button(self.tab_single, text="⚕️ 运行计算生物学药效评估", command=self.on_predict_click,
                                     bg="#007bff", fg="white", font=("Microsoft YaHei", 11, "bold"), width=25,
                                     relief="flat")
        self.predict_btn.pack(pady=10)

        # 结果展示区
        self.result_frame = tk.LabelFrame(self.tab_single, text=" 联合药效评估报告 ", font=self.title_font, padx=15,
                                          pady=15)
        self.result_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.result_label = tk.Label(self.result_frame, text="等待录入参数...", font=("Microsoft YaHei", 11),
                                     justify="center")
        self.result_label.pack(pady=10)

    # ---------------- 标签页 2：高通量批量筛选 ----------------
    def build_batch_tab(self):
        batch_frame = tk.Frame(self.tab_batch, padx=20, pady=20)
        batch_frame.pack(fill="both", expand=True)

        tk.Label(batch_frame, text="上传包含实验设计的 Excel/CSV 文件进行高通量测算", font=self.title_font).pack(
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
        self.batch_start_btn = tk.Button(btn_frame, text="🚀 开始批量测算", command=self.run_batch_mock, bg="#28a745",
                                         fg="white", font=self.title_font, width=15)
        self.batch_start_btn.pack(side="left", padx=10)
        self.batch_export_btn = tk.Button(btn_frame, text="💾 导出结果报表", state=tk.DISABLED, bg="#17a2b8", fg="white",
                                          font=self.title_font, width=15)
        self.batch_export_btn.pack(side="left", padx=10)

    # ---------------- 标签页 3：评估历史记录 ----------------
    def build_history_tab(self):
        hist_frame = tk.Frame(self.tab_history, padx=15, pady=15)
        hist_frame.pack(fill="both", expand=True)

        # 表格控件
        columns = ("Date", "Cell", "Drug A", "Drug B", "Score", "Suggestion")
        self.tree = ttk.Treeview(hist_frame, columns=columns, show="headings", height=15)

        # 定义表头
        self.tree.heading("Date", text="时间")
        self.tree.heading("Cell", text="细胞系")
        self.tree.heading("Drug A", text="药物 A")
        self.tree.heading("Drug B", text="药物 B")
        self.tree.heading("Score", text="协同概率")
        self.tree.heading("Suggestion", text="临床建议")

        # 定义列宽
        self.tree.column("Date", width=80)
        self.tree.column("Cell", width=100)
        self.tree.column("Drug A", width=100)
        self.tree.column("Drug B", width=100)
        self.tree.column("Score", width=80, anchor="center")
        self.tree.column("Suggestion", width=120)

        # 滚动条
        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        # 底部按钮
        btn_frame = tk.Frame(hist_frame)
        btn_frame.pack(fill="x", pady=10)
        tk.Button(btn_frame, text="🗑️ 清空历史", command=self.clear_history, fg="red").pack(side="left")
        tk.Button(btn_frame, text="📤 导出历史记录 (Excel)", bg="#17a2b8", fg="white").pack(side="right")

    # ---------------- 模拟数据与交互逻辑 ----------------
    def populate_mock_data(self):
        mock_cells = ["MCF7 (乳腺癌细胞)", "A549 (非小细胞肺癌)", "HeLa (宫颈癌细胞)", "HepG2 (肝癌细胞)"]
        mock_drugs = ["Paclitaxel", "Cisplatin", "Aspirin", "Imatinib", "5-Fluorouracil"]
        self.cell_entry.config(values=mock_cells)
        self.drug1_entry.config(values=mock_drugs)
        self.drug2_entry.config(values=mock_drugs)
        self.cell_entry.current(0);
        self.drug1_entry.current(0);
        self.drug2_entry.current(1)

    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel/CSV Data", "*.csv *.xlsx")])
        if filepath:
            self.file_path_var.set(filepath)

    def on_predict_click(self):
        self.predict_btn.config(state=tk.DISABLED)
        self.result_label.config(text="正在进行分子对齐与生信测算...", fg="#17a2b8")
        self.root.after(1200, self.show_single_result)

    def show_single_result(self):
        prob = random.random()
        c = self.cell_entry.get();
        d1 = self.drug1_entry.get();
        d2 = self.drug2_entry.get()

        if prob >= 0.8:
            res_text = f"【极高潜力】协同概率: {prob * 100:.2f}%\n💡 建议：优先开展体外验证。"
            sug, color = "优先验证", "#28a745"
        elif prob >= 0.5:
            res_text = f"【潜在组合】协同概率: {prob * 100:.2f}%\n💡 建议：可作为备选实验方案。"
            sug, color = "备选方案", "#fd7e14"
        else:
            res_text = f"【风险提示】协同概率: {prob * 100:.2f}%\n💡 建议：存在拮抗可能，暂缓规划。"
            sug, color = "暂缓规划", "#dc3545"

        self.result_label.config(text=res_text, fg=color)
        self.predict_btn.config(state=tk.NORMAL)

        # 自动写入历史记录
        current_time = time.strftime("%H:%M:%S")
        self.tree.insert("", "end", values=(current_time, c, d1, d2, f"{prob * 100:.1f}%", sug))

    def run_batch_mock(self):
        if "未选择" in self.file_path_var.get():
            messagebox.showwarning("提示", "请先选择需要测算的 Excel 文件！")
            return
        self.batch_start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._mock_progress_task, daemon=True).start()

    def _mock_progress_task(self):
        for i in range(1, 101):
            time.sleep(0.03)  # 模拟计算耗时
            self.root.after(0, self.progress_var.set, i)
            self.root.after(0, self.progress_label.config, {"text": f"测算进度: {i}% (正在处理分子图谱)"})

        self.root.after(0, self.progress_label.config, {"text": "✅ 批量测算完成！请导出报表。"})
        self.root.after(0, self.batch_start_btn.config, {"state": tk.NORMAL})
        self.root.after(0, self.batch_export_btn.config, {"state": tk.NORMAL})

    def clear_history(self):
        for item in self.tree.get_children():
            self.tree.delete(item)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedSynergyAppDemo(root)
    root.mainloop()