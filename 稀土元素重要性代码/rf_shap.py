# import argparse
# from pathlib import Path
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import shap
# import numpy as np
# from matplotlib import rcParams
# import traceback
# import warnings
# warnings.filterwarnings("ignore")
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# import tkinter as tk
# from tkinter import simpledialog, messagebox

# # ========= 配置 =========
# INPUT_PATH_DEFAULT = r"D:/学习/文章/稀土机器学习/副本宣威组黏土岩（数据分析）_knn_filled_k10_Y.xlsx"
# SHEET_NAME_DEFAULT = 0
# # 统一输出目录
# OUTPUT_DIR_DEFAULT = Path(r"D:/学习/文章/稀土机器学习/RF/Y")
# # 可配置的特征列范围（Excel列名，包含两端），例如：B-AC
# COL_START = "B"
# COL_END = "AC"
# # ========================

# def bin_ree(ree_series: pd.Series) -> pd.Series:
#     bins = [100, 1000, 2000, 100000000]
#     labels = ["100-1000", "1000-2000", "2000-100000000"]
#     return pd.cut(pd.to_numeric(ree_series, errors="coerce"), bins=bins, right=False, labels=labels)


# # ===== 交互式选择：解析边界/列工具 =====
# def parse_bins(bins_str: str) -> list[float]:
#     parts = [p.strip() for p in bins_str.split(',') if p.strip()]
#     return [float(p) for p in parts]


# def auto_labels_from_bins(bins: list[float], right: bool) -> list[str]:
#     labels: list[str] = []
#     for i in range(len(bins) - 1):
#         a, b = bins[i], bins[i + 1]
#         labels.append(f"({a},{b}]" if right else f"[{a},{b})")
#     return labels


# def bin_target(series: pd.Series, bins: list[float], right: bool) -> pd.Series:
#     s = pd.to_numeric(series, errors="coerce")
#     labels = auto_labels_from_bins(bins, right)
#     return pd.cut(s, bins=bins, right=right, labels=labels)


# def excel_col_to_index(col_letter: str) -> int:
#     col_letter = col_letter.strip().upper()
#     value = 0
#     for ch in col_letter:
#         if not ('A' <= ch <= 'Z'):
#             raise ValueError(f"非法列名: {col_letter}")
#         value = value * 26 + (ord(ch) - ord('A') + 1)
#     return value - 1  # 0基

# def save_feature_importance(rf_model, feature_names, input_path: Path, output_dir: Path):
#     importances = rf_model.feature_importances_
#     fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
#     stem = input_path.stem
#     output_dir.mkdir(parents=True, exist_ok=True)
#     fi_csv = output_dir / f"{stem}_rf_feature_importances.csv"
#     fi_df.to_csv(fi_csv, index=False, encoding="utf-8")
#     print(f"已保存特征重要性CSV: {fi_csv}")

#     plt.figure(figsize=(10, 8))
#     plt.barh(fi_df["feature"], fi_df["importance"])
#     plt.gca().invert_yaxis()
#     plt.xlabel("Importance")
#     plt.title("RandomForest Feature Importances")
#     plt.tight_layout()
#     fi_png = output_dir / f"{stem}_rf_feature_importances.png"
#     plt.savefig(fi_png, dpi=200, bbox_inches='tight')
#     fi_svg = output_dir / f"{stem}_rf_feature_importances.svg"
#     plt.savefig(fi_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存特征重要性图: {fi_png}")
#     print(f"已保存特征重要性SVG: {fi_svg}")
    
#     return fi_df

# def save_shap_plots(explainer, X_valid, feature_names, input_path: Path, output_dir: Path, shap_values=None):
#     stem = input_path.stem
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     if shap_values is None:
#         shap_values = explainer.shap_values(X_valid.values)
    
#     # 统一为2D (n_samples, n_features) 以避免多分类在某些shap版本上的索引问题
#     # 同时保留原始结构供导出等使用
#     if isinstance(shap_values, list):
#         # list[n_classes] of (n_samples, n_features)
#         shap_values_2d = np.mean(np.abs(np.array(shap_values)), axis=0)
#     elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#         # (n_samples, n_features, n_classes)
#         shap_values_2d = np.mean(np.abs(shap_values), axis=2)
#     else:
#         # (n_samples, n_features)
#         shap_values_2d = shap_values

#     # 特征名
#     if isinstance(feature_names, np.ndarray):
#         plot_feature_names = [str(name) for name in feature_names.tolist()]
#     else:
#         plot_feature_names = [str(name) for name in feature_names]
    
#     # 方法1：尝试使用SHAP的标准方法
#     try:
#         # 条形图
#         plt.figure(figsize=(12, 10))
#         shap.summary_plot(shap_values_2d, X_valid.values, feature_names=np.array(plot_feature_names, dtype=object), 
#                          plot_type="bar", show=False, max_display=len(plot_feature_names))
#         plt.tight_layout()
#         shap_bar = output_dir / f"{stem}_shap_bar.png"
#         plt.savefig(shap_bar, dpi=200, bbox_inches='tight')
#         shap_bar_svg = output_dir / f"{stem}_shap_bar.svg"
#         plt.savefig(shap_bar_svg, format='svg', bbox_inches='tight')
#         plt.close()
#         print(f"已保存SHAP条形图: {shap_bar}")
#         print(f"已保存SHAP条形图SVG: {shap_bar_svg}")

#         # 蜂群图
#         plt.figure(figsize=(14, 10))
#         shap.summary_plot(shap_values_2d, X_valid.values, feature_names=np.array(plot_feature_names, dtype=object), 
#                          show=False, max_display=len(plot_feature_names))
#         plt.tight_layout()
#         shap_bee = output_dir / f"{stem}_shap_beeswarm.png"
#         plt.savefig(shap_bee, dpi=200, bbox_inches='tight')
#         shap_bee_svg = output_dir / f"{stem}_shap_beeswarm.svg"
#         plt.savefig(shap_bee_svg, format='svg', bbox_inches='tight')
#         plt.close()
#         print(f"已保存SHAP蜂群图: {shap_bee}")
#         print(f"已保存SHAP蜂群图SVG: {shap_bee_svg}")
#     except Exception as e:
#         print(f"SHAP标准绘图方法失败: {e}")
#         # 如果标准方法失败，尝试替代方法
#         try:
#             # 计算平均SHAP绝对值
#             if isinstance(shap_values, list):
#                 # (C, N, F) -> (F,)
#                 mean_abs_shap = np.mean(np.abs(np.array(shap_values)), axis=(0, 1))
#             elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#                 # (N, F, C) -> (F,)
#                 mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
#             else:
#                 # (N, F) -> (F,)
#                 mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
#             # 排序
#             sorted_idx = np.argsort(mean_abs_shap)
            
#             # 创建替代条形图
#             plt.figure(figsize=(12, 10))
#             plt.barh(range(len(mean_abs_shap)), mean_abs_shap[sorted_idx].astype(float))
#             plt.yticks(range(len(mean_abs_shap)), [plot_feature_names[int(i)] for i in sorted_idx])
#             plt.xlabel("Mean |SHAP|")
#             plt.title("SHAP Feature Importance (Alternative)")
#             plt.tight_layout()
#             shap_bar_alt = output_dir / f"{stem}_shap_bar_alternative.png"
#             plt.savefig(shap_bar_alt, dpi=200, bbox_inches='tight')
#             shap_bar_alt_svg = output_dir / f"{stem}_shap_bar_alternative.svg"
#             plt.savefig(shap_bar_alt_svg, format='svg', bbox_inches='tight')
#             plt.close()
#             print(f"已保存替代SHAP条形图: {shap_bar_alt}")
#             print(f"已保存替代SHAP条形图SVG: {shap_bar_alt_svg}")
#         except Exception as e2:
#             print(f"替代SHAP条形图也失败: {e2}")

#     # 自定义条形图（确保显示所有特征）
#     try:
#         if isinstance(shap_values, list):
#             sv = np.array(shap_values)  # (C, N, F)
#             mean_abs = np.mean(np.abs(sv), axis=(0, 1))  # (F,)
#         elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#             mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))  # (F,)
#         else:
#             mean_abs = np.mean(np.abs(shap_values), axis=0)  # (F,)

#         order = np.argsort(-mean_abs).flatten()
        
#         sorted_feature_names = [plot_feature_names[i] for i in order]
#         sorted_mean_abs = mean_abs[order]
        
#         n_features = len(sorted_feature_names)
#         fig_height = max(8, n_features * 0.6)
        
#         plt.figure(figsize=(12, fig_height))
#         bars = plt.barh(sorted_feature_names, sorted_mean_abs.astype(float))
#         plt.gca().invert_yaxis()
#         plt.xlabel("Mean |SHAP| (all classes)")
#         plt.title(f"SHAP mean absolute values ({n_features} features)")
        
#         # 添加数值标签
#         for i, bar in enumerate(bars):
#             width = bar.get_width()
#             plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
#                     f'{width:.4f}', ha='left', va='center')
        
#         plt.tight_layout()
#         shap_bar_all = output_dir / f"{stem}_shap_bar_all.png"
#         plt.savefig(shap_bar_all, dpi=200, bbox_inches='tight')
#         shap_bar_all_svg = output_dir / f"{stem}_shap_bar_all.svg"
#         plt.savefig(shap_bar_all_svg, format='svg', bbox_inches='tight')
#         plt.close()
#         print(f"已保存完整SHAP条形图(包含全部{n_features}个特征): {shap_bar_all}")
#         print(f"已保存完整SHAP条形图SVG: {shap_bar_all_svg}")
        
#     except Exception as e:
#         print(f"保存完整SHAP条形图失败: {e}")
#         print(traceback.format_exc())


# def save_shap_values_excel(shap_values, feature_names, input_path: Path, output_dir: Path, class_names=None):
#     """将每个样本的每个特征的 SHAP 值导出为 Excel。
#     - shap_values: list[n_classes] of (n_samples, n_features) 或 (n_samples, n_features)
#     - feature_names: list[str]
#     - class_names: 可选。若为 None 且 shap_values 为 list，则使用 range 索引
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)
#     stem = input_path.stem
#     out_path = output_dir / f"{stem}_shap_values.xlsx"

#     def clean_sheet_name(name: str) -> str:
#         """清理工作表名称，移除Excel不允许的字符"""
#         # Excel不允许的字符: [ ] : / \ ? * 
#         invalid_chars = ['[', ']', ':', '/', '\\', '?', '*']
#         clean_name = name
#         for char in invalid_chars:
#             clean_name = clean_name.replace(char, '_')
#         # 确保不超过31字符
#         return clean_name[:31]

#     try:
#         with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#             wrote = False
#             if isinstance(shap_values, list):
#                 num_classes = len(shap_values)
#                 if class_names is None:
#                     class_names = [f"class_{i}" for i in range(num_classes)]
#                 for i, sv in enumerate(shap_values):
#                     df_sv = pd.DataFrame(sv, columns=feature_names)
#                     sheet_name = clean_sheet_name(str(class_names[i]))
#                     df_sv.to_excel(writer, sheet_name=sheet_name, index=False)
#                     wrote = True
#             elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#                 n_classes = shap_values.shape[2]
#                 if class_names is None:
#                     class_names = [f"class_{i}" for i in range(n_classes)]
#                 for i in range(n_classes):
#                     df_sv = pd.DataFrame(shap_values[:, :, i], columns=feature_names)
#                     sheet_name = clean_sheet_name(str(class_names[i]))
#                     df_sv.to_excel(writer, sheet_name=sheet_name, index=False)
#                     wrote = True
#             else:
#                 df_sv = pd.DataFrame(shap_values, columns=feature_names)
#                 df_sv.to_excel(writer, sheet_name="shap", index=False)
#                 wrote = True
#             if not wrote:
#                 pd.DataFrame({"info": ["no shap values written"]}).to_excel(writer, sheet_name="info", index=False)
#         print(f"已保存逐样本逐特征的SHAP值: {out_path}")
#     except Exception as e:
#         print(f"保存SHAP值Excel失败: {e}")
#         print(traceback.format_exc())

# def train_rf_and_shap(input_path: Path, sheet_name):
#     try:
#         df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl", header=0)
        
#         if df.shape[0] < 10:
#             print("警告: 数据样本量较少，可能影响模型性能")
        
#         # 根据配置的列范围，动态限制可选目标列
#         start_idx = excel_col_to_index(COL_START)
#         end_idx = excel_col_to_index(COL_END) + 1
#         if start_idx < 0 or end_idx > df.shape[1] or start_idx >= end_idx:
#             raise ValueError(f"列范围无效: {COL_START}-{COL_END}，表共有 {df.shape[1]} 列")
#         b_to_aq_cols = [str(col) for col in df.columns[start_idx:end_idx]]

#         # 弹窗让用户输入：目标列名或索引（限 B-AQ），以及分箱边界
#         root = tk.Tk(); root.withdraw()
#         messagebox.showinfo("选择目标列", f"将从 {COL_START}-{COL_END} 列中选择目标列并设置分箱边界。")
#         target_col = simpledialog.askstring("目标列", f"请输入目标列名或1基索引({start_idx+1}-{end_idx})\n可选列: {', '.join(b_to_aq_cols)}")
#         bins_str = simpledialog.askstring("分箱边界", "请输入分箱边界（逗号分隔，例如: 100,1000,2000,100000000）")
#         right_flag = messagebox.askyesno("区间闭合方式", "是否使用右闭合 (a,b]?\n选择“否”为左闭右开 [a,b)")
#         root.destroy()

#         if not target_col or not bins_str:
#             print("未提供目标列或分箱边界，已取消。")
#             return

#         # 解析目标列
#         try:
#             if target_col.isdigit():
#                 idx = int(target_col)
#                 if idx < (start_idx+1) or idx > end_idx:
#                     raise ValueError("索引超出选择范围")
#                 target_series = df.iloc[:, idx - 1]
#             else:
#                 if target_col not in b_to_aq_cols:
#                     raise ValueError("列名不在选择范围内")
#                 target_series = df[target_col]
#         except Exception as e:
#             print(f"目标列解析失败: {e}")
#             return

#         # 解析边界并分箱
#         try:
#             bins = parse_bins(bins_str)
#             if len(bins) < 2:
#                 raise ValueError("分箱边界至少需2个数")
#             y = bin_target(target_series, bins=bins, right=right_flag)
#         except Exception as e:
#             print(f"分箱失败: {e}")
#             return

#         # 特征：配置的范围内（不包括作为目标的那一列）
#         X_raw = df.iloc[:, start_idx:end_idx]
#         if target_col.isdigit():
#             target_pos = int(target_col) - 1
#         else:
#             target_pos = list(df.columns).index(target_col)
#         feature_cols_mask = np.ones(X_raw.shape[1], dtype=bool)
#         if start_idx <= target_pos < end_idx:
#             feature_cols_mask[target_pos - start_idx] = False
#         X_raw = X_raw.loc[:, feature_cols_mask]
#         feature_names_list = [str(c) for c in X_raw.columns]

#         features_df = X_raw.apply(pd.to_numeric, errors="coerce")
#         valid_mask = y.notna() & ~features_df.isna().any(axis=1)
#         X_valid = features_df.loc[valid_mask]
#         y_valid = y.loc[valid_mask]

#         if X_valid.shape[0] == 0:
#             print("没有满足分箱条件且特征完整的样本，退出。")
#             return
        
#         class_counts = y_valid.value_counts()
#         print("各类别样本数量:")
#         for cls, count in class_counts.items():
#             print(f"  {cls}: {count}")

#         # 交叉验证 + 超参数调优（GridSearchCV）
#         base_rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
#         param_grid = {
#             "n_estimators": [200, 300, 500],
#             "max_depth": [None, 10, 20, 30],
#             "min_samples_split": [2, 5, 10],
#             "min_samples_leaf": [1, 2, 4],
#             "max_features": ["sqrt", "log2", None]
#         }
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         grid = GridSearchCV(
#             estimator=base_rf,
#             param_grid=param_grid,
#             scoring="accuracy",
#             cv=cv,
#             n_jobs=-1,
#             verbose=1,
#             refit=True,
#             return_train_score=True,
#         )

#         print("开始随机森林的网格搜索与交叉验证...")
#         grid.fit(X_valid.values, y_valid.values)
#         print(f"最佳参数: {grid.best_params_}")
#         print(f"交叉验证最佳准确率: {grid.best_score_:.4f}")

#         # 保存CV结果
#         OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
#         cv_results_path = OUTPUT_DIR_DEFAULT / f"{input_path.stem}_rf_gridcv_results.csv"
#         pd.DataFrame(grid.cv_results_).to_csv(cv_results_path, index=False, encoding="utf-8")
#         print(f"已保存CV结果: {cv_results_path}")

#         best_rf: RandomForestClassifier = grid.best_estimator_
#         train_score = best_rf.score(X_valid.values, y_valid.values)
#         print(f"在全部有效样本上的重拟合准确率: {train_score:.4f}")

#         # 基于最佳模型输出结果
#         save_feature_importance(best_rf, feature_names_list, input_path, OUTPUT_DIR_DEFAULT)
#         explainer = shap.TreeExplainer(best_rf)
#         shap_values = explainer.shap_values(X_valid.values)
#         save_shap_plots(explainer, X_valid, feature_names_list, input_path, OUTPUT_DIR_DEFAULT, shap_values)
#         # 导出每个特征的SHAP值到Excel（按类别分Sheet）
#         try:
#             class_names = None
#             if hasattr(best_rf, "classes_"):
#                 class_names = [str(c) for c in best_rf.classes_]
#             save_shap_values_excel(shap_values, feature_names_list, input_path, OUTPUT_DIR_DEFAULT, class_names)
#         except Exception as e:
#             print(f"导出SHAP值Excel时出错: {e}")
#             print(traceback.format_exc())
        
#         print("分析完成!")
        
#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")
#         print(traceback.format_exc())

# def main():
#     parser = argparse.ArgumentParser(description="RandomForest + SHAP on Excel")
#     parser.add_argument("--input", type=str, default="", help="Excel路径")
#     parser.add_argument("--sheet", type=str, default="", help="工作表名或索引")
#     args = parser.parse_args()

#     if args.input:
#         input_path = Path(args.input)
#     else:
#         input_path = Path(INPUT_PATH_DEFAULT)
    
#     if not input_path.exists():
#         raise FileNotFoundError(f"找不到文件: {input_path}")

#     if args.sheet:
#         sheet_name = int(args.sheet) if args.sheet.isdigit() else args.sheet
#     else:
#         sheet_name = SHEET_NAME_DEFAULT

#     train_rf_and_shap(input_path, sheet_name)

# if __name__ == "__main__":
#     main()







import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap
import numpy as np
import traceback
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import tkinter as tk
from tkinter import simpledialog, messagebox

# ========= 配置 =========
INPUT_PATH_DEFAULT = r"D:/学习/文章/稀土机器学习/Zr_knn_filled_k2.xlsx"
SHEET_NAME_DEFAULT = 0
OUTPUT_DIR_DEFAULT = Path(r"D:/学习/文章/稀土机器学习/RF/new/Zr1")
# 可配置的特征列范围（Excel列名，包含两端），例如：B-AC
COL_START = "B"
COL_END = "AC"
# ========================

def parse_bins(bins_str: str) -> list[float]:
    parts = [p.strip() for p in bins_str.split(',') if p.strip()]
    return [float(p) for p in parts]

def auto_labels_from_bins(bins: list[float], right: bool) -> list[str]:
    labels: list[str] = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        labels.append(f"({a},{b}]" if right else f"[{a},{b})")
    return labels

def bin_target(series: pd.Series, bins: list[float], right: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    labels = auto_labels_from_bins(bins, right)
    return pd.cut(s, bins=bins, right=right, labels=labels)

def excel_col_to_index(col_letter: str) -> int:
    col_letter = col_letter.strip().upper()
    value = 0
    for ch in col_letter:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"非法列名: {col_letter}")
        value = value * 26 + (ord(ch) - ord('A') + 1)
    return value - 1  # 0基

def save_feature_importance(rf_model, feature_names, input_path: Path, output_dir: Path):
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    fi_csv = output_dir / f"{stem}_rf_feature_importances.csv"
    fi_df.to_csv(fi_csv, index=False, encoding="utf-8")
    print(f"已保存特征重要性CSV: {fi_csv}")

    plt.figure(figsize=(10, max(6, len(fi_df)*0.5)))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("RandomForest Feature Importances")
    plt.tight_layout()
    fi_png = output_dir / f"{stem}_rf_feature_importances.png"
    plt.savefig(fi_png, dpi=200, bbox_inches='tight')
    fi_svg = output_dir / f"{stem}_rf_feature_importances.svg"
    plt.savefig(fi_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"已保存特征重要性图: {fi_png}")
    print(f"已保存特征重要性SVG: {fi_svg}")
    return fi_df

# ================== 关键：按类别蜂群图 + “Mean|SHAP|”真实值堆叠横条图 ==================
def save_shap_plots(explainer, X_valid, feature_names, input_path: Path, output_dir: Path, shap_values=None):
    """
    生成：
    1) 每个类别一张 SHAP 蜂群图（有符号）
    2) “Mean|SHAP|”堆叠横条图（每特征=各类别 mean(|SHAP|) 真实值相加）
    并导出一份 CSV（各类 mean|SHAP| 明细 + 总和）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    feat_names = [str(f) for f in (feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names)]
    F = len(feat_names)

    # 计算 SHAP
    if shap_values is None:
        shap_values = explainer.shap_values(X_valid.values)

    # 统一成 per-class 列表：class_svs: List[(N,F)]
    if isinstance(shap_values, list):
        class_svs = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_svs = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    else:  # 二分类/回归 -> 单类
        class_svs = [shap_values]
    C = len(class_svs)

    # 类别名
    try:
        class_names = [str(c) for c in explainer.model.classes_]
        if len(class_names) != C:
            class_names = [f"class_{i}" for i in range(C)]
    except Exception:
        class_names = [f"class_{i}" for i in range(C)]

    # 1) 每类一张蜂群图（有符号 SHAP）
    for ci, sv in enumerate(class_svs):
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            sv, X_valid.values,
            feature_names=np.array(feat_names, dtype=object),
            show=False, max_display=F
        )
        plt.tight_layout()
        png = output_dir / f"{stem}_shap_beeswarm_{class_names[ci]}.png"
        svg = output_dir / f"{stem}_shap_beeswarm_{class_names[ci]}.svg"
        plt.savefig(png, dpi=200, bbox_inches='tight')
        plt.savefig(svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"已保存蜂群图：{png}")

    # 2) Mean|SHAP| 堆叠横条图（非比例，真实值）
    per_class_mean_abs = np.vstack([np.mean(np.abs(sv), axis=0) for sv in class_svs])  # (C,F)
    totals = per_class_mean_abs.sum(axis=0)  # (F,)
    order = np.argsort(-totals)              # 按总重要性降序
    names_ord = np.array(feat_names, dtype=object)[order]
    per_class_mean_abs_ord = per_class_mean_abs[:, order]
    totals_ord = totals[order]

    plt.figure(figsize=(14, max(8, F * 0.5)))
    left = np.zeros(len(names_ord))
    for ci in range(C):
        plt.barh(names_ord, per_class_mean_abs_ord[ci], left=left, label=class_names[ci])
        left += per_class_mean_abs_ord[ci]
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP| Value")
    plt.title("Feature importance (stacked by class Mean |SHAP|)")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    bar_png = output_dir / f"{stem}_shap_bar_stacked_meanabs.png"
    bar_svg = output_dir / f"{stem}_shap_bar_stacked_meanabs.svg"
    plt.savefig(bar_png, dpi=200, bbox_inches='tight')
    plt.savefig(bar_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"已保存堆叠横条图（Mean|SHAP|）：{bar_png}")

    # 3) 导出明细 CSV
    df_out = pd.DataFrame({"feature": names_ord, "total_mean_abs_shap": totals_ord})
    for ci, cname in enumerate(class_names):
        df_out[cname] = per_class_mean_abs_ord[ci]
    csv_path = output_dir / f"{stem}_shap_bar_stacked_meanabs_details.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"已保存堆叠明细 CSV：{csv_path}")

def save_shap_values_excel(shap_values, feature_names, input_path: Path, output_dir: Path, class_names=None):
    """将每个样本的每个特征的 SHAP 值导出为 Excel（多分类按类分Sheet）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    out_path = output_dir / f"{stem}_shap_values.xlsx"

    def clean_sheet_name(name: str) -> str:
        invalid_chars = ['[', ']', ':', '/', '\\', '?', '*']
        for ch in invalid_chars:
            name = name.replace(ch, '_')
        return name[:31]

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            wrote = False
            if isinstance(shap_values, list):
                num_classes = len(shap_values)
                if class_names is None:
                    class_names = [f"class_{i}" for i in range(num_classes)]
                for i, sv in enumerate(shap_values):
                    pd.DataFrame(sv, columns=feature_names).to_excel(writer, sheet_name=clean_sheet_name(str(class_names[i])), index=False)
                    wrote = True
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                n_classes = shap_values.shape[2]
                if class_names is None:
                    class_names = [f"class_{i}" for i in range(n_classes)]
                for i in range(n_classes):
                    pd.DataFrame(shap_values[:, :, i], columns=feature_names).to_excel(writer, sheet_name=clean_sheet_name(str(class_names[i])), index=False)
                    wrote = True
            else:
                pd.DataFrame(shap_values, columns=feature_names).to_excel(writer, sheet_name="shap", index=False)
                wrote = True
            if not wrote:
                pd.DataFrame({"info": ["no shap values written"]}).to_excel(writer, sheet_name="info", index=False)
        print(f"已保存逐样本逐特征的SHAP值: {out_path}")
    except Exception as e:
        print(f"保存SHAP值Excel失败: {e}")
        print(traceback.format_exc())

def train_rf_and_shap(input_path: Path, sheet_name):
    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl", header=0)
        if df.shape[0] < 10:
            print("警告: 数据样本量较少，可能影响模型性能")

        # 列范围与交互式选择
        start_idx = excel_col_to_index(COL_START)
        end_idx = excel_col_to_index(COL_END) + 1
        if start_idx < 0 or end_idx > df.shape[1] or start_idx >= end_idx:
            raise ValueError(f"列范围无效: {COL_START}-{COL_END}，表共有 {df.shape[1]} 列")
        available_cols = [str(col) for col in df.columns[start_idx:end_idx]]

        root = tk.Tk(); root.withdraw()
        messagebox.showinfo("选择目标列", f"将从 {COL_START}-{COL_END} 列中选择目标列并设置分箱边界。")
        target_col = simpledialog.askstring("目标列", f"请输入目标列名或1基索引({start_idx+1}-{end_idx})\n可选列: {', '.join(available_cols)}")
        bins_str = simpledialog.askstring("分箱边界", "请输入分箱边界（逗号分隔，例如: 100,1000,2000,100000000）")
        right_flag = messagebox.askyesno("区间闭合方式", "是否使用右闭合 (a,b]?\n选择“否”为左闭右开 [a,b)")
        root.destroy()

        if not target_col or not bins_str:
            print("未提供目标列或分箱边界，已取消。")
            return

        # 解析目标列
        if target_col.isdigit():
            idx = int(target_col)
            if idx < (start_idx+1) or idx > end_idx:
                raise ValueError("索引超出选择范围")
            target_series = df.iloc[:, idx - 1]
        else:
            if target_col not in available_cols:
                raise ValueError("列名不在选择范围内")
            target_series = df[target_col]

        # 分箱
        bins = parse_bins(bins_str)
        if len(bins) < 2:
            raise ValueError("分箱边界至少需2个数")
        y = bin_target(target_series, bins=bins, right=right_flag)

        # 特征构造（去掉目标列）
        X_raw = df.iloc[:, start_idx:end_idx]
        target_pos = (int(target_col) - 1) if target_col.isdigit() else list(df.columns).index(target_col)
        feature_cols_mask = np.ones(X_raw.shape[1], dtype=bool)
        if start_idx <= target_pos < end_idx:
            feature_cols_mask[target_pos - start_idx] = False
        X_raw = X_raw.loc[:, feature_cols_mask]
        feature_names_list = [str(c) for c in X_raw.columns]

        X_num = X_raw.apply(pd.to_numeric, errors="coerce")
        valid_mask = y.notna() & ~X_num.isna().any(axis=1)
        X_valid = X_num.loc[valid_mask]
        y_valid = y.loc[valid_mask]
        if X_valid.shape[0] == 0:
            print("没有满足分箱条件且特征完整的样本，退出。")
            return

        print("各类别样本数量:")
        for cls, count in y_valid.value_counts().items():
            print(f"  {cls}: {count}")

        # 网格搜索
        base_rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
            return_train_score=True,
        )
        print("开始随机森林的网格搜索与交叉验证...")
        grid.fit(X_valid.values, y_valid.values)
        print(f"最佳参数: {grid.best_params_}")
        print(f"交叉验证最佳准确率: {grid.best_score_:.4f}")

        # 保存CV结果
        OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
        cv_results_path = OUTPUT_DIR_DEFAULT / f"{input_path.stem}_rf_gridcv_results.csv"
        pd.DataFrame(grid.cv_results_).to_csv(cv_results_path, index=False, encoding="utf-8")
        print(f"已保存CV结果: {cv_results_path}")

        best_rf: RandomForestClassifier = grid.best_estimator_
        train_score = best_rf.score(X_valid.values, y_valid.values)
        print(f"在全部有效样本上的重拟合准确率: {train_score:.4f}")

        # 输出：特征重要性 + SHAP 图 + SHAP Excel
        save_feature_importance(best_rf, feature_names_list, input_path, OUTPUT_DIR_DEFAULT)
        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X_valid.values)

        save_shap_plots(explainer, X_valid, feature_names_list, input_path, OUTPUT_DIR_DEFAULT, shap_values)

        try:
            class_names = None
            if hasattr(best_rf, "classes_"):
                class_names = [str(c) for c in best_rf.classes_]
            save_shap_values_excel(shap_values, feature_names_list, input_path, OUTPUT_DIR_DEFAULT, class_names)
        except Exception as e:
            print(f"导出SHAP值Excel时出错: {e}")
            print(traceback.format_exc())

        print("分析完成!")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="RandomForest + SHAP on Excel")
    parser.add_argument("--input", type=str, default="", help="Excel路径")
    parser.add_argument("--sheet", type=str, default="", help="工作表名或索引")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else Path(INPUT_PATH_DEFAULT)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")
    sheet_name = int(args.sheet) if (args.sheet and args.sheet.isdigit()) else (args.sheet if args.sheet else SHEET_NAME_DEFAULT)

    train_rf_and_shap(input_path, sheet_name)

if __name__ == "__main__":
    main()
