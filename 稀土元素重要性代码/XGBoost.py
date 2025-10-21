# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import shap
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# import warnings
# warnings.filterwarnings("ignore")
# import tkinter as tk
# from tkinter import simpledialog, messagebox


# # ========= 配置 =========
# INPUT_PATH_DEFAULT = r"D:/学习/文章/稀土机器学习/副本宣威组黏土岩（数据分析）_knn_filled_k10_Y.xlsx"
# SHEET_NAME_DEFAULT = 0
# OUTPUT_DIR_DEFAULT = Path(r"D:/学习/文章/稀土机器学习/XGBoost/Y")
# # 可配置的特征列范围（Excel列名，包含两端），例如：B-AC
# COL_START = "B"
# COL_END = "AC"
# # ========================


# def parse_bins(bins_str: str) -> list[float]:
#     parts = [p.strip() for p in bins_str.split(',') if p.strip()]
#     return [float(p) for p in parts]


# def auto_labels_from_bins(bins: list[float], right: bool) -> list[str]:
#     labels: list[str] = []
#     for i in range(len(bins) - 1):
#         a, b = bins[i], bins[i + 1]
#         labels.append(f"({a},{b}]" if right else f"[{a},{b})")
#     return labels


# def bin_target(series: pd.Series, bins: list[float], labels: list[str] | None, right: bool) -> pd.Series:
#     s = pd.to_numeric(series, errors="coerce")
#     if labels is None:
#         labels = auto_labels_from_bins(bins, right)
#     return pd.cut(s, bins=bins, right=right, labels=labels)


# def excel_col_to_index(col_letter: str) -> int:
#     col_letter = col_letter.strip().upper()
#     value = 0
#     for ch in col_letter:
#         if not ('A' <= ch <= 'Z'):
#             raise ValueError(f"非法列名: {col_letter}")
#         value = value * 26 + (ord(ch) - ord('A') + 1)
#     return value - 1  # 0基


# def save_feature_importance(model: XGBClassifier, feature_names: list[str], input_path: Path, output_dir: Path):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     stem = input_path.stem
#     importances = model.feature_importances_
#     order = np.argsort(-importances)
#     names = np.array(feature_names, dtype=object)[order]
#     vals = importances[order]

#     fi_df = pd.DataFrame({"feature": names, "importance": vals})
#     fi_csv = output_dir / f"{stem}_xgb_feature_importances.csv"
#     fi_df.to_csv(fi_csv, index=False, encoding="utf-8")
#     print(f"已保存特征重要性CSV: {fi_csv}")

#     plt.figure(figsize=(12, max(8, len(names)*0.6)))
#     plt.barh(names, vals)
#     plt.gca().invert_yaxis()
#     plt.xlabel("Gain/Importance")
#     plt.title("XGBoost Feature Importances")
#     plt.tight_layout()
#     fi_png = output_dir / f"{stem}_xgb_feature_importances.png"
#     plt.savefig(fi_png, dpi=200, bbox_inches='tight')
#     fi_svg = output_dir / f"{stem}_xgb_feature_importances.svg"
#     plt.savefig(fi_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存特征重要性图: {fi_png}")
#     print(f"已保存特征重要性SVG: {fi_svg}")


# def save_shap_plots(model: XGBClassifier, X: pd.DataFrame, feature_names: list[str], input_path: Path, output_dir: Path):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     stem = input_path.stem

#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X.values)

#     # 统一为2D用于绘图（多分类时对类别取绝对值均值）
#     if isinstance(shap_values, list):
#         shap_values_plot = np.mean(np.abs(np.array(shap_values)), axis=0)
#     elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#         shap_values_plot = np.mean(np.abs(shap_values), axis=2)
#     else:
#         shap_values_plot = shap_values

#     plot_feature_names = [str(n) for n in feature_names]

#     # 条形图
#     plt.figure(figsize=(12, 10))
#     shap.summary_plot(shap_values_plot, X.values, feature_names=np.array(plot_feature_names, dtype=object),
#                       plot_type="bar", show=False, max_display=len(feature_names))
#     plt.tight_layout()
#     shap_bar = output_dir / f"{stem}_xgb_shap_bar.png"
#     plt.savefig(shap_bar, dpi=200, bbox_inches='tight')
#     shap_bar_svg = output_dir / f"{stem}_xgb_shap_bar.svg"
#     plt.savefig(shap_bar_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存XGB SHAP条形图: {shap_bar}")
#     print(f"已保存XGB SHAP条形图SVG: {shap_bar_svg}")

#     # 蜂群图
#     plt.figure(figsize=(14, 10))
#     shap.summary_plot(shap_values_plot, X.values, feature_names=np.array(plot_feature_names, dtype=object),
#                       show=False, max_display=len(feature_names))
#     plt.tight_layout()
#     shap_bee = output_dir / f"{stem}_xgb_shap_beeswarm.png"
#     plt.savefig(shap_bee, dpi=200, bbox_inches='tight')
#     shap_bee_svg = output_dir / f"{stem}_xgb_shap_beeswarm.svg"
#     plt.savefig(shap_bee_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存XGB SHAP蜂群图: {shap_bee}")
#     print(f"已保存XGB SHAP蜂群图SVG: {shap_bee_svg}")

#     # 导出SHAP值到Excel（保留原始结构，list 或 3D 都分别按类别写）
#     out_path = output_dir / f"{stem}_xgb_shap_values.xlsx"
    
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
#                 class_names = [f"class_{i}" for i in range(len(shap_values))]
#                 for i, sv in enumerate(shap_values):
#                     sheet_name = clean_sheet_name(class_names[i])
#                     pd.DataFrame(sv, columns=feature_names).to_excel(writer, sheet_name=sheet_name, index=False)
#                     wrote = True
#             elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#                 n_classes = shap_values.shape[2]
#                 class_names = [f"class_{i}" for i in range(n_classes)]
#                 for i in range(n_classes):
#                     sheet_name = clean_sheet_name(class_names[i])
#                     pd.DataFrame(shap_values[:, :, i], columns=feature_names).to_excel(writer, sheet_name=sheet_name, index=False)
#                     wrote = True
#             else:
#                 pd.DataFrame(shap_values, columns=feature_names).to_excel(writer, sheet_name="shap", index=False)
#                 wrote = True
#             if not wrote:
#                 pd.DataFrame({"info": ["no shap values written"]}).to_excel(writer, sheet_name="info", index=False)
#         print(f"已保存XGB SHAP值: {out_path}")
#     except Exception as e:
#         print(f"保存XGB SHAP值失败: {e}")


# def main():
#     parser = argparse.ArgumentParser(description="XGBoost + SHAP on Excel")
#     parser.add_argument("--input", type=str, default="", help="Excel 路径")
#     parser.add_argument("--sheet", type=str, default="", help="工作表名或索引")
#     args = parser.parse_args()

#     input_path = Path(args.input) if args.input else Path(INPUT_PATH_DEFAULT)
#     if not input_path.exists():
#         raise FileNotFoundError(f"找不到文件: {input_path}")
#     sheet_name = int(args.sheet) if args.sheet.isdigit() else (args.sheet if args.sheet else SHEET_NAME_DEFAULT)

#     # 读数据
#     df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl", header=0)
    
#     # 根据配置的列范围，动态限制可选目标列
#     start_idx = excel_col_to_index(COL_START)
#     end_idx = excel_col_to_index(COL_END) + 1
#     if start_idx < 0 or end_idx > df.shape[1] or start_idx >= end_idx:
#         raise ValueError(f"列范围无效: {COL_START}-{COL_END}，表共有 {df.shape[1]} 列")
#     b_to_aq_cols = [str(col) for col in df.columns[start_idx:end_idx]]

#     # 弹窗让用户输入：目标列名或索引（限 B-AQ），以及分箱边界
#     root = tk.Tk(); root.withdraw()
#     messagebox.showinfo("选择目标列", f"将从 {COL_START}-{COL_END} 列中选择目标列并设置分箱边界。")
#     target_col = simpledialog.askstring("目标列", f"请输入目标列名或1基索引({start_idx+1}-{end_idx})\n可选列: {', '.join(b_to_aq_cols)}")
#     bins_str = simpledialog.askstring("分箱边界", "请输入分箱边界（逗号分隔，例如: 100,1000,2000,100000000）")
#     right_flag = messagebox.askyesno("区间闭合方式", "是否使用右闭合 (a,b]?\n选择\"否\"为左闭右开 [a,b)")
#     root.destroy()

#     if not target_col or not bins_str:
#         print("未提供目标列或分箱边界，已取消。")
#         return

#     # 解析目标列
#     try:
#         if target_col.isdigit():
#             idx = int(target_col)
#             if idx < (start_idx+1) or idx > end_idx:
#                 raise ValueError("索引超出选择范围")
#             target_series = df.iloc[:, idx - 1]
#         else:
#             if target_col not in b_to_aq_cols:
#                 raise ValueError("列名不在选择范围内")
#             target_series = df[target_col]
#     except Exception as e:
#         print(f"目标列解析失败: {e}")
#         return

#     # 解析边界并分箱
#     try:
#         bins = parse_bins(bins_str)
#         if len(bins) < 2:
#             raise ValueError("分箱边界至少需2个数")
#         y = bin_target(target_series, bins=bins, labels=None, right=right_flag)
#     except Exception as e:
#         print(f"分箱失败: {e}")
#         return

#     # 特征：配置的范围内（不包括作为目标的那一列）
#     X_raw = df.iloc[:, start_idx:end_idx]
#     if target_col.isdigit():
#         target_pos = int(target_col) - 1
#     else:
#         target_pos = list(df.columns).index(target_col)
#     feature_cols_mask = np.ones(X_raw.shape[1], dtype=bool)
#     # 将对应于目标列的位置置为 False（如果在范围内）
#     if start_idx <= target_pos < end_idx:
#         feature_cols_mask[target_pos - start_idx] = False
#     X = X_raw.loc[:, feature_cols_mask]
#     feature_names = [str(c) for c in X.columns]
    
#     X = X.apply(pd.to_numeric, errors="coerce")

#     valid = y.notna() & ~X.isna().any(axis=1)
#     X_valid = X.loc[valid]
#     y_valid = y.loc[valid]
#     if X_valid.empty:
#         print("没有满足条件的样本")
#         return

#     # 将字符串标签编码为 0..C-1（XGBoost 需要数值标签）
#     le = LabelEncoder()
#     y_enc = le.fit_transform(y_valid.values)
#     num_classes = len(le.classes_)
#     model = XGBClassifier(
#         objective="multi:softprob" if num_classes > 2 else "binary:logistic",
#         num_class=int(num_classes) if num_classes > 2 else None,
#         eval_metric="mlogloss" if num_classes > 2 else "logloss",
#         tree_method="hist",
#         random_state=42,
#         n_estimators=300,
#         n_jobs=-1,
#     )

#     # 网格搜索与交叉验证
#     param_grid = {
#         "n_estimators": [200, 300, 500],
#         "max_depth": [3, 5, 7],
#         "learning_rate": [0.05, 0.1, 0.2],
#         "subsample": [0.7, 0.9, 1.0],
#         "colsample_bytree": [0.7, 0.9, 1.0]
#     }
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     grid = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         scoring="accuracy",
#         cv=cv,
#         n_jobs=-1,
#         verbose=1,
#         refit=True,
#         return_train_score=True,
#     )
#     print("开始XGBoost网格搜索与交叉验证...")
#     grid.fit(X_valid.values, y_enc)
#     print(f"最佳参数: {grid.best_params_}")
#     print(f"交叉验证最佳准确率: {grid.best_score_:.4f}")

#     # 保存CV结果
#     OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
#     cv_results_path = OUTPUT_DIR_DEFAULT / f"{input_path.stem}_xgb_gridcv_results.csv"
#     pd.DataFrame(grid.cv_results_).to_csv(cv_results_path, index=False, encoding="utf-8")
#     print(f"已保存CV结果: {cv_results_path}")

#     best_model: XGBClassifier = grid.best_estimator_
#     acc = best_model.score(X_valid.values, y_enc)
#     print(f"在全部有效样本上的重拟合准确率: {acc:.4f}")

#     save_feature_importance(best_model, feature_names, input_path, OUTPUT_DIR_DEFAULT)
#     save_shap_plots(best_model, X_valid, feature_names, input_path, OUTPUT_DIR_DEFAULT)


# if __name__ == "__main__":
#     main()







# 加入shap_dependence_plot
# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import shap
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# import warnings
# import tkinter as tk
# from tkinter import simpledialog, messagebox

# warnings.filterwarnings("ignore")

# # ========= 配置 =========
# INPUT_PATH_DEFAULT = r"D:/学习/文章/稀土机器学习/副本宣威组黏土岩（数据分析）_knn_filled_k10_Y.xlsx"
# SHEET_NAME_DEFAULT = 0
# OUTPUT_DIR_DEFAULT = Path(r"D:/学习/文章/稀土机器学习/XGBoost/Y-shap_de")
# COL_START = "B"
# COL_END = "AC"
# # ========================

# def parse_bins(bins_str: str) -> list[float]:
#     parts = [p.strip() for p in bins_str.split(',') if p.strip()]
#     return [float(p) for p in parts]

# def auto_labels_from_bins(bins: list[float], right: bool) -> list[str]:
#     labels: list[str] = []
#     for i in range(len(bins) - 1):
#         a, b = bins[i], bins[i + 1]
#         labels.append(f"({a},{b}]" if right else f"[{a},{b})")
#     return labels

# def bin_target(series: pd.Series, bins: list[float], labels: list[str] | None, right: bool) -> pd.Series:
#     s = pd.to_numeric(series, errors="coerce")
#     if labels is None:
#         labels = auto_labels_from_bins(bins, right)
#     return pd.cut(s, bins=bins, right=right, labels=labels)

# def excel_col_to_index(col_letter: str) -> int:
#     col_letter = col_letter.strip().upper()
#     value = 0
#     for ch in col_letter:
#         if not ('A' <= ch <= 'Z'):
#             raise ValueError(f"非法列名: {col_letter}")
#         value = value * 26 + (ord(ch) - ord('A') + 1)
#     return value - 1

# def save_feature_importance(model: XGBClassifier, feature_names: list[str], input_path: Path, output_dir: Path):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     stem = input_path.stem
#     importances = model.feature_importances_
#     order = np.argsort(-importances)
#     names = np.array(feature_names, dtype=object)[order]
#     vals = importances[order]

#     fi_df = pd.DataFrame({"feature": names, "importance": vals})
#     fi_csv = output_dir / f"{stem}_xgb_feature_importances.csv"
#     fi_df.to_csv(fi_csv, index=False, encoding="utf-8")
#     print(f"已保存特征重要性CSV: {fi_csv}")

#     plt.figure(figsize=(12, max(8, len(names)*0.6)))
#     plt.barh(names, vals)
#     plt.gca().invert_yaxis()
#     plt.xlabel("Gain/Importance")
#     plt.title("XGBoost Feature Importances")
#     plt.tight_layout()
#     fi_png = output_dir / f"{stem}_xgb_feature_importances.png"
#     plt.savefig(fi_png, dpi=200, bbox_inches='tight')
#     fi_svg = output_dir / f"{stem}_xgb_feature_importances.svg"
#     plt.savefig(fi_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存特征重要性图: {fi_png}")
#     print(f"已保存特征重要性SVG: {fi_svg}")

# def save_shap_plots(model: XGBClassifier, X: pd.DataFrame, feature_names: list[str], input_path: Path, output_dir: Path):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     stem = input_path.stem
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X.values)

#     if isinstance(shap_values, list):
#         shap_values_plot = np.mean(np.abs(np.array(shap_values)), axis=0)
#     elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#         shap_values_plot = np.mean(np.abs(shap_values), axis=2)  # Average across classes if multi-class
#     else:
#         shap_values_plot = shap_values

#     plot_feature_names = [str(n) for n in feature_names]

#     # SHAP条形图
#     plt.figure(figsize=(12, 10))
#     shap.summary_plot(shap_values_plot, X.values, feature_names=np.array(plot_feature_names, dtype=object),
#                       plot_type="bar", show=False, max_display=len(feature_names))
#     plt.tight_layout()
#     shap_bar = output_dir / f"{stem}_xgb_shap_bar.png"
#     plt.savefig(shap_bar, dpi=200, bbox_inches='tight')
#     shap_bar_svg = output_dir / f"{stem}_xgb_shap_bar.svg"
#     plt.savefig(shap_bar_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存XGB SHAP条形图: {shap_bar}")
#     print(f"已保存XGB SHAP条形图SVG: {shap_bar_svg}")

#     # SHAP蜂群图
#     plt.figure(figsize=(14, 10))
#     shap.summary_plot(shap_values_plot, X.values, feature_names=np.array(plot_feature_names, dtype=object),
#                       show=False, max_display=len(feature_names))
#     plt.tight_layout()
#     shap_bee = output_dir / f"{stem}_xgb_shap_beeswarm.png"
#     plt.savefig(shap_bee, dpi=200, bbox_inches='tight')
#     shap_bee_svg = output_dir / f"{stem}_xgb_shap_beeswarm.svg"
#     plt.savefig(shap_bee_svg, format='svg', bbox_inches='tight')
#     plt.close()
#     print(f"已保存XGB SHAP蜂群图: {shap_bee}")
#     print(f"已保存XGB SHAP蜂群图SVG: {shap_bee_svg}")

#     # SHAP依赖图
#     for feature_idx, feature_name in enumerate(feature_names):
#         plt.figure(figsize=(12, 8))
#         # Ensure the shap_values is 2D before plotting dependence plot
#         if isinstance(shap_values, list):
#             shap.dependence_plot(feature_name, shap_values[0], X.values, feature_names=np.array(plot_feature_names, dtype=object), show=False)
#         elif isinstance(shap_values, np.ndarray):
#             if shap_values.ndim == 3:  # Multi-class scenario
#                 shap.dependence_plot(feature_name, shap_values[:, :, 0], X.values, feature_names=np.array(plot_feature_names, dtype=object), show=False)
#             else:  # Binary or regression case
#                 shap.dependence_plot(feature_name, shap_values, X.values, feature_names=np.array(plot_feature_names, dtype=object), show=False)
#         plt.tight_layout()
#         shap_dep = output_dir / f"{stem}_xgb_shap_dependence_{feature_name}.png"
#         plt.savefig(shap_dep, dpi=200, bbox_inches='tight')
#         shap_dep_svg = output_dir / f"{stem}_xgb_shap_dependence_{feature_name}.svg"
#         plt.savefig(shap_dep_svg, format='svg', bbox_inches='tight')
#         plt.close()
#         print(f"已保存XGB SHAP依赖图: {shap_dep}")
#         print(f"已保存XGB SHAP依赖图SVG: {shap_dep_svg}")

#     # 导出SHAP值到Excel
#     out_path = output_dir / f"{stem}_xgb_shap_values.xlsx"
    
#     def clean_sheet_name(name: str) -> str:
#         invalid_chars = ['[', ']', ':', '/', '\\', '?', '*']
#         clean_name = name
#         for char in invalid_chars:
#             clean_name = clean_name.replace(char, '_')
#         return clean_name[:31]
    
#     try:
#         with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#             wrote = False
#             if isinstance(shap_values, list):
#                 class_names = [f"class_{i}" for i in range(len(shap_values))]
#                 for i, sv in enumerate(shap_values):
#                     sheet_name = clean_sheet_name(class_names[i])
#                     pd.DataFrame(sv, columns=feature_names).to_excel(writer, sheet_name=sheet_name, index=False)
#                     wrote = True
#             elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
#                 n_classes = shap_values.shape[2]
#                 class_names = [f"class_{i}" for i in range(n_classes)]
#                 for i in range(n_classes):
#                     sheet_name = clean_sheet_name(class_names[i])
#                     pd.DataFrame(shap_values[:, :, i], columns=feature_names).to_excel(writer, sheet_name=sheet_name, index=False)
#                     wrote = True
#             else:
#                 pd.DataFrame(shap_values, columns=feature_names).to_excel(writer, sheet_name="shap", index=False)
#                 wrote = True
#             if not wrote:
#                 pd.DataFrame({"info": ["no shap values written"]}).to_excel(writer, sheet_name="info", index=False)
#         print(f"已保存XGB SHAP值: {out_path}")
#     except Exception as e:
#         print(f"保存XGB SHAP值失败: {e}")

# def main():
#     parser = argparse.ArgumentParser(description="XGBoost + SHAP on Excel")
#     parser.add_argument("--input", type=str, default="", help="Excel 路径")
#     parser.add_argument("--sheet", type=str, default="", help="工作表名或索引")
#     args = parser.parse_args()

#     input_path = Path(args.input) if args.input else Path(INPUT_PATH_DEFAULT)
#     if not input_path.exists():
#         raise FileNotFoundError(f"找不到文件: {input_path}")
#     sheet_name = int(args.sheet) if args.sheet.isdigit() else (args.sheet if args.sheet else SHEET_NAME_DEFAULT)

#     # 读数据
#     df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl", header=0)
    
#     # 根据配置的列范围，动态限制可选目标列
#     start_idx = excel_col_to_index(COL_START)
#     end_idx = excel_col_to_index(COL_END) + 1
#     if start_idx < 0 or end_idx > df.shape[1] or start_idx >= end_idx:
#         raise ValueError(f"列范围无效: {COL_START}-{COL_END}，表共有 {df.shape[1]} 列")
#     b_to_aq_cols = [str(col) for col in df.columns[start_idx:end_idx]]

#     # 弹窗让用户输入：目标列名或索引（限 B-AQ），以及分箱边界
#     root = tk.Tk(); root.withdraw()
#     messagebox.showinfo("选择目标列", f"将从 {COL_START}-{COL_END} 列中选择目标列并设置分箱边界。")
#     target_col = simpledialog.askstring("目标列", f"请输入目标列名或1基索引({start_idx+1}-{end_idx})\n可选列: {', '.join(b_to_aq_cols)}")
#     bins_str = simpledialog.askstring("分箱边界", "请输入分箱边界（逗号分隔，例如: 100,1000,2000,100000000）")
#     right_flag = messagebox.askyesno("区间闭合方式", "是否使用右闭合 (a,b]?\n选择\"否\"为左闭右开 [a,b)")
#     root.destroy()

#     if not target_col or not bins_str:
#         print("未提供目标列或分箱边界，已取消。")
#         return

#     # 解析目标列
#     try:
#         if target_col.isdigit():
#             idx = int(target_col)
#             if idx < (start_idx+1) or idx > end_idx:
#                 raise ValueError("索引超出选择范围")
#             target_series = df.iloc[:, idx - 1]
#         else:
#             if target_col not in b_to_aq_cols:
#                 raise ValueError("列名不在选择范围内")
#             target_series = df[target_col]
#     except Exception as e:
#         print(f"目标列解析失败: {e}")
#         return

#     # 解析边界并分箱
#     try:
#         bins = parse_bins(bins_str)
#         if len(bins) < 2:
#             raise ValueError("分箱边界至少需2个数")
#         y = bin_target(target_series, bins=bins, labels=None, right=right_flag)
#     except Exception as e:
#         print(f"分箱失败: {e}")
#         return

#     # 特征：配置的范围内（不包括作为目标的那一列）
#     X_raw = df.iloc[:, start_idx:end_idx]
#     if target_col.isdigit():
#         target_pos = int(target_col) - 1
#     else:
#         target_pos = list(df.columns).index(target_col)
#     feature_cols_mask = np.ones(X_raw.shape[1], dtype=bool)
#     if start_idx <= target_pos < end_idx:
#         feature_cols_mask[target_pos - start_idx] = False
#     X = X_raw.loc[:, feature_cols_mask]
#     feature_names = [str(c) for c in X.columns]

#     X = X.apply(pd.to_numeric, errors="coerce")

#     valid = y.notna() & ~X.isna().any(axis=1)
#     X_valid = X.loc[valid]
#     y_valid = y.loc[valid]
#     if X_valid.empty:
#         print("没有满足条件的样本")
#         return

#     le = LabelEncoder()
#     y_enc = le.fit_transform(y_valid.values)
#     num_classes = len(le.classes_)
#     model = XGBClassifier(
#         objective="multi:softprob" if num_classes > 2 else "binary:logistic",
#         num_class=int(num_classes) if num_classes > 2 else None,
#         eval_metric="mlogloss" if num_classes > 2 else "logloss",
#         tree_method="hist",
#         random_state=42,
#         n_estimators=300,
#         n_jobs=-1,
#     )

#     param_grid = {
#         "n_estimators": [200, 300, 500],
#         "max_depth": [3, 5, 7],
#         "learning_rate": [0.05, 0.1, 0.2],
#         "subsample": [0.7, 0.9, 1.0],
#         "colsample_bytree": [0.7, 0.9, 1.0]
#     }
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     grid = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         scoring="accuracy",
#         cv=cv,
#         n_jobs=-1,
#         verbose=1,
#         refit=True,
#         return_train_score=True,
#     )
#     print("开始XGBoost网格搜索与交叉验证...")
#     grid.fit(X_valid.values, y_enc)
#     print(f"最佳参数: {grid.best_params_}")
#     print(f"交叉验证最佳准确率: {grid.best_score_:.4f}")

#     # 保存CV结果
#     OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
#     cv_results_path = OUTPUT_DIR_DEFAULT / f"{input_path.stem}_xgb_gridcv_results.csv"
#     pd.DataFrame(grid.cv_results_).to_csv(cv_results_path, index=False, encoding="utf-8")
#     print(f"已保存CV结果: {cv_results_path}")

#     best_model: XGBClassifier = grid.best_estimator_
#     acc = best_model.score(X_valid.values, y_enc)
#     print(f"在全部有效样本上的重拟合准确率: {acc:.4f}")

#     save_feature_importance(best_model, feature_names, input_path, OUTPUT_DIR_DEFAULT)
#     save_shap_plots(best_model, X_valid, feature_names, input_path, OUTPUT_DIR_DEFAULT)

# if __name__ == "__main__":
#     main()





import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import tkinter as tk
from tkinter import simpledialog, messagebox


# ========= 配置 =========
INPUT_PATH_DEFAULT = r"D:/学习/文章/稀土机器学习/Zr_knn_filled_k2.xlsx"
SHEET_NAME_DEFAULT = 0
OUTPUT_DIR_DEFAULT = Path(r"D:/学习/文章/稀土机器学习/XGBoost/new/Zr1")
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


def bin_target(series: pd.Series, bins: list[float], labels: list[str] | None, right: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if labels is None:
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


def save_feature_importance(model: XGBClassifier, feature_names: list[str], input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    importances = model.feature_importances_
    order = np.argsort(-importances)
    names = np.array(feature_names, dtype=object)[order]
    vals = importances[order]

    fi_df = pd.DataFrame({"feature": names, "importance": vals})
    fi_csv = output_dir / f"{stem}_xgb_feature_importances.csv"
    fi_df.to_csv(fi_csv, index=False, encoding="utf-8")
    print(f"已保存特征重要性CSV: {fi_csv}")

    plt.figure(figsize=(12, max(8, len(names)*0.6)))
    plt.barh(names, vals)
    plt.gca().invert_yaxis()
    plt.xlabel("Gain/Importance")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    fi_png = output_dir / f"{stem}_xgb_feature_importances.png"
    plt.savefig(fi_png, dpi=200, bbox_inches='tight')
    fi_svg = output_dir / f"{stem}_xgb_feature_importances.svg"
    plt.savefig(fi_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"已保存特征重要性图: {fi_png}")
    print(f"已保存特征重要性SVG: {fi_svg}")


def save_shap_plots(model: XGBClassifier, X: pd.DataFrame, feature_names: list[str], input_path: Path, output_dir: Path):
    """
    生成：
    1) 每个类别一张 SHAP 蜂群图（有符号）
    2) “Mean|SHAP|”堆叠横条图（每特征=各类别 mean(|SHAP|) 真实值相加）
    3) 导出一份 CSV（各类 mean|SHAP| 明细 + 总和）
    同时继续将原始 SHAP 值导出到 Excel（多分类按类分sheet）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    feat_names = [str(f) for f in feature_names]
    F = len(feat_names)

    # 计算 SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)

    # 规范为 per-class 列表：class_svs: List[(N,F)]
    if isinstance(shap_values, list):
        class_svs = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:  # (N,F,C)
        class_svs = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    else:  # 二分类/回归：二维 (N,F)
        class_svs = [shap_values]
    C = len(class_svs)

    # 类别名
    try:
        class_names = [str(c) for c in model.classes_]
        if len(class_names) != C:
            class_names = [f"class_{i}" for i in range(C)]
    except Exception:
        class_names = [f"class_{i}" for i in range(C)]

    # 1) 每类一张蜂群图（有符号）
    for ci, sv in enumerate(class_svs):
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            sv, X.values,
            feature_names=np.array(feat_names, dtype=object),
            show=False, max_display=F
        )
        plt.tight_layout()
        png = output_dir / f"{stem}_xgb_shap_beeswarm_{class_names[ci]}.png"
        svg = output_dir / f"{stem}_xgb_shap_beeswarm_{class_names[ci]}.svg"
        plt.savefig(png, dpi=200, bbox_inches='tight')
        plt.savefig(svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"已保存蜂群图：{png}")

    # 2) Mean|SHAP| 堆叠横条图（真实值相加，非比例）
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
    bar_png = output_dir / f"{stem}_xgb_shap_bar_stacked_meanabs.png"
    bar_svg = output_dir / f"{stem}_xgb_shap_bar_stacked_meanabs.svg"
    plt.savefig(bar_png, dpi=200, bbox_inches='tight')
    plt.savefig(bar_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"已保存堆叠横条图（Mean|SHAP|）：{bar_png}")

    # 3) 导出堆叠明细 CSV
    df_out = pd.DataFrame({"feature": names_ord, "total_mean_abs_shap": totals_ord})
    for ci, cname in enumerate(class_names):
        df_out[cname] = per_class_mean_abs_ord[ci]
    csv_path = output_dir / f"{stem}_xgb_shap_bar_stacked_meanabs_details.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"已保存堆叠明细 CSV：{csv_path}")

    # 4) 导出 SHAP 值到 Excel（保留原始结构）
    out_path = output_dir / f"{stem}_xgb_shap_values.xlsx"
    def clean_sheet_name(name: str) -> str:
        invalid_chars = ['[', ']', ':', '/', '\\', '?', '*']
        clean_name = name
        for ch in invalid_chars:
            clean_name = clean_name.replace(ch, '_')
        return clean_name[:31]

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            wrote = False
            if isinstance(shap_values, list):
                class_names_xlsx = [f"class_{i}" for i in range(len(shap_values))]
                for i, sv in enumerate(shap_values):
                    sheet_name = clean_sheet_name(class_names_xlsx[i])
                    pd.DataFrame(sv, columns=feature_names).to_excel(writer, sheet_name=sheet_name, index=False)
                    wrote = True
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                n_classes = shap_values.shape[2]
                class_names_xlsx = [f"class_{i}" for i in range(n_classes)]
                for i in range(n_classes):
                    sheet_name = clean_sheet_name(class_names_xlsx[i])
                    pd.DataFrame(shap_values[:, :, i], columns=feature_names).to_excel(writer, sheet_name=sheet_name, index=False)
                    wrote = True
            else:
                pd.DataFrame(shap_values, columns=feature_names).to_excel(writer, sheet_name="shap", index=False)
                wrote = True
            if not wrote:
                pd.DataFrame({"info": ["no shap values written"]}).to_excel(writer, sheet_name="info", index=False)
        print(f"已保存XGB SHAP值: {out_path}")
    except Exception as e:
        print(f"保存XGB SHAP值失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="XGBoost + SHAP on Excel")
    parser.add_argument("--input", type=str, default="", help="Excel 路径")
    parser.add_argument("--sheet", type=str, default="", help="工作表名或索引")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else Path(INPUT_PATH_DEFAULT)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")
    sheet_name = int(args.sheet) if args.sheet.isdigit() else (args.sheet if args.sheet else SHEET_NAME_DEFAULT)

    # 读数据
    df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl", header=0)
    
    # 根据配置的列范围，动态限制可选目标列
    start_idx = excel_col_to_index(COL_START)
    end_idx = excel_col_to_index(COL_END) + 1
    if start_idx < 0 or end_idx > df.shape[1] or start_idx >= end_idx:
        raise ValueError(f"列范围无效: {COL_START}-{COL_END}，表共有 {df.shape[1]} 列")
    b_to_aq_cols = [str(col) for col in df.columns[start_idx:end_idx]]

    # 弹窗让用户输入：目标列名或索引（限 B-AQ），以及分箱边界
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("选择目标列", f"将从 {COL_START}-{COL_END} 列中选择目标列并设置分箱边界。")
    target_col = simpledialog.askstring("目标列", f"请输入目标列名或1基索引({start_idx+1}-{end_idx})\n可选列: {', '.join(b_to_aq_cols)}")
    bins_str = simpledialog.askstring("分箱边界", "请输入分箱边界（逗号分隔，例如: 100,1000,2000,100000000）")
    right_flag = messagebox.askyesno("区间闭合方式", "是否使用右闭合 (a,b]?\n选择\"否\"为左闭右开 [a,b)")
    root.destroy()

    if not target_col or not bins_str:
        print("未提供目标列或分箱边界，已取消。")
        return

    # 解析目标列
    try:
        if target_col.isdigit():
            idx = int(target_col)
            if idx < (start_idx+1) or idx > end_idx:
                raise ValueError("索引超出选择范围")
            target_series = df.iloc[:, idx - 1]
        else:
            if target_col not in b_to_aq_cols:
                raise ValueError("列名不在选择范围内")
            target_series = df[target_col]
    except Exception as e:
        print(f"目标列解析失败: {e}")
        return

    # 解析边界并分箱
    try:
        bins = parse_bins(bins_str)
        if len(bins) < 2:
            raise ValueError("分箱边界至少需2个数")
        y = bin_target(target_series, bins=bins, labels=None, right=right_flag)
    except Exception as e:
        print(f"分箱失败: {e}")
        return

    # 特征：配置的范围内（不包括作为目标的那一列）
    X_raw = df.iloc[:, start_idx:end_idx]
    if target_col.isdigit():
        target_pos = int(target_col) - 1
    else:
        target_pos = list(df.columns).index(target_col)
    feature_cols_mask = np.ones(X_raw.shape[1], dtype=bool)
    if start_idx <= target_pos < end_idx:
        feature_cols_mask[target_pos - start_idx] = False
    X = X_raw.loc[:, feature_cols_mask]
    feature_names = [str(c) for c in X.columns]
    
    X = X.apply(pd.to_numeric, errors="coerce")

    valid = y.notna() & ~X.isna().any(axis=1)
    X_valid = X.loc[valid]
    y_valid = y.loc[valid]
    if X_valid.empty:
        print("没有满足条件的样本")
        return

    # 将字符串标签编码为 0..C-1（XGBoost 需要数值标签）
    le = LabelEncoder()
    y_enc = le.fit_transform(y_valid.values)
    num_classes = len(le.classes_)
    model = XGBClassifier(
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        num_class=int(num_classes) if num_classes > 2 else None,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        tree_method="hist",
        random_state=42,
        n_estimators=300,
        n_jobs=-1,
    )

    # 网格搜索与交叉验证
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    print("开始XGBoost网格搜索与交叉验证...")
    grid.fit(X_valid.values, y_enc)
    print(f"最佳参数: {grid.best_params_}")
    print(f"交叉验证最佳准确率: {grid.best_score_:.4f}")

    # 保存CV结果
    OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    cv_results_path = OUTPUT_DIR_DEFAULT / f"{input_path.stem}_xgb_gridcv_results.csv"
    pd.DataFrame(grid.cv_results_).to_csv(cv_results_path, index=False, encoding="utf-8")
    print(f"已保存CV结果: {cv_results_path}")

    best_model: XGBClassifier = grid.best_estimator_
    acc = best_model.score(X_valid.values, y_enc)
    print(f"在全部有效样本上的重拟合准确率: {acc:.4f}")

    save_feature_importance(best_model, feature_names, input_path, OUTPUT_DIR_DEFAULT)
    save_shap_plots(best_model, X_valid, feature_names, input_path, OUTPUT_DIR_DEFAULT)


if __name__ == "__main__":
    main()
