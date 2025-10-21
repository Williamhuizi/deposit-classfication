import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

# ========= 配置 =========
input_path = r"D:/学习/文章/矿床分类预测文章/分类应用/数据/最终数据.xlsx"   # 修改为你的Excel路径
sheet_name = 0                        # Sheet名或索引
candidate_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]   # 备选k
weights = "distance"                   # "uniform" 或 "distance"
mask_frac = 0.1                        # 每次遮蔽比例（0.05~0.2常用）
n_repeats = 5                          # 重复次数（>1更稳健）
random_seed = 42                       # 复现实验
# =======================

# 列范围配置（Excel 列名，包含两端）。示例：B-AC
COL_START = "P"
COL_END = "AF"
# 自定义输出标记（可留空）。示例："Sc"、"v2" 等
OUTPUT_TAG = ""

def zscore_fit(X: np.ndarray):
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=0)
    stds_safe = np.where(stds == 0, 1.0, stds)
    return means, stds_safe

def zscore_transform(X: np.ndarray, means: np.ndarray, stds: np.ndarray):
    return (X - means) / stds

def zscore_inverse_transform(Xz: np.ndarray, means: np.ndarray, stds: np.ndarray):
    return Xz * stds + means

def excel_col_to_index(col_letter: str) -> int:
    col_letter = col_letter.strip().upper()
    value = 0
    for ch in col_letter:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"非法列名: {col_letter}")
        value = value * 26 + (ord(ch) - ord('A') + 1)
    return value - 1  # 转为0基索引

def evaluate_k(X: np.ndarray, k: int, weights: str, mask_frac: float, n_repeats: int, rng: np.random.RandomState):
    # 在标准化空间做KNN，误差在原始尺度上计算
    means, stds = zscore_fit(X)
    Xz = zscore_transform(X, means, stds)

    observed_idx = np.where(~np.isnan(Xz))
    total_obs = observed_idx[0].size
    if total_obs == 0:
        raise ValueError("选定的列全部为空，无法评估。")

    n_mask = max(1, int(total_obs * mask_frac))
    imputer = KNNImputer(n_neighbors=k, weights=weights)

    sq_errors = []
    for _ in range(n_repeats):
        # 随机遮蔽一部分已知值
        flat_indices = rng.choice(total_obs, size=n_mask, replace=False)
        rows = observed_idx[0][flat_indices]
        cols = observed_idx[1][flat_indices]

        Xz_masked = Xz.copy()
        Xz_masked[rows, cols] = np.nan

        # 填充并反标准化
        Xz_imputed = imputer.fit_transform(Xz_masked)
        X_imputed = zscore_inverse_transform(Xz_imputed, means, stds)

        # 计算被遮蔽单元格的平方误差（原始尺度）
        err = (X_imputed[rows, cols] - X[rows, cols]) ** 2
        sq_errors.append(err)

    rmse = np.sqrt(np.mean(np.concatenate(sq_errors)))
    return rmse

def main():
    rng = np.random.RandomState(random_seed)

    # 读取
    df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")

    # 选择列范围（例如 B-AC），包含两端
    start_idx = excel_col_to_index(COL_START)
    end_idx_inclusive = excel_col_to_index(COL_END)
    end_idx = end_idx_inclusive + 1  # iloc 右开
    if start_idx < 0 or end_idx > df.shape[1] or start_idx >= end_idx:
        raise ValueError(f"列范围无效: {COL_START}-{COL_END}，表共有 {df.shape[1]} 列")
    block = df.iloc[:, start_idx:end_idx].copy()
    block_numeric = block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # 若某些列几乎全空，会影响评估与KNN距离，可选：剔除全空列并在最后还原
    all_nan_cols = np.all(np.isnan(block_numeric), axis=0)
    if np.any(all_nan_cols):
        # 保留列索引映射
        kept_cols = np.where(~all_nan_cols)[0]
        removed_cols = np.where(all_nan_cols)[0]
        X = block_numeric[:, kept_cols]
    else:
        kept_cols = np.arange(block_numeric.shape[1])
        removed_cols = np.array([], dtype=int)
        X = block_numeric

    if X.shape[1] == 0:
        raise ValueError(f"{COL_START}-{COL_END} 列全为空或不可转为数值，无法进行KNN填充。")

    # 评估每个k
    results = []
    for k in candidate_ks:
        rmse = evaluate_k(X, k=k, weights=weights, mask_frac=mask_frac, n_repeats=n_repeats, rng=rng)
        results.append((k, rmse))
        print(f"k={k:>2}, RMSE={rmse:.6f}")

    # 选择最优k
    best_k, best_rmse = min(results, key=lambda x: x[1])
    print(f"选择最优 k={best_k} (RMSE={best_rmse:.6f})")

    # 用最优k在标准化空间拟合+填充全数据
    means, stds = zscore_fit(X)
    Xz = zscore_transform(X, means, stds)
    imputer = KNNImputer(n_neighbors=best_k, weights=weights)
    Xz_filled = imputer.fit_transform(Xz)
    X_filled = zscore_inverse_transform(Xz_filled, means, stds)

    # 将结果写回到原 DataFrame 的所选列范围
    out_block = block_numeric.copy()
    out_block[:, kept_cols] = X_filled
    # 对全空列（若有），保持为原样（全NaN），或按需设定为0/均值
    df.iloc[:, start_idx:end_idx] = out_block

    # 保存
    p = Path(input_path)
    tag = ("_" + OUTPUT_TAG) if OUTPUT_TAG else ""
    output_path = str(p.with_name(p.stem + f"_knn_filled_k{best_k}" + tag + p.suffix))
    df.to_excel(output_path, index=False)
    print(f"完成，已保存到: {output_path}")

if __name__ == "__main__":
    main()



