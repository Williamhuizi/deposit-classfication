import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import re
import argparse

# ===== 默认参数（可改）=====
DEFAULT_ZERO_STRATEGY = "minhalf"     # 'minhalf'（每行最小正数/2）或 'eps'
DEFAULT_EPS = 1e-9                    # zero_strategy='eps' 时用
DEFAULT_DO_CLOSURE = True             # 是否先做 closure（行和=1）
DEFAULT_MODE = "append"               # 'append' 追加新列 或 'replace' 覆盖
OUTPUT_TAG = "CLR"                    # 输出文件名后缀
# =========================

# ---------- 工具 ----------
def excel_col_to_index(col_letter: str) -> int:
    col_letter = col_letter.strip().upper()
    value = 0
    for ch in col_letter:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"非法列名: {col_letter}")
        value = value * 26 + (ord(ch) - ord('A') + 1)
    return value - 1

def excel_index_to_col(idx0: int) -> str:
    idx = idx0 + 1
    s = []
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s.append(chr(65 + r))
    return ''.join(reversed(s))

def is_header_valid(name) -> bool:
    if name is None: return False
    s = str(name).strip()
    if s == "": return False
    if re.match(r"^Unnamed:\s*\d+\s*$", s, flags=re.IGNORECASE): return False
    return True

def parse_columns_spec(spec: str, df: pd.DataFrame):
    """
    解析列选择字符串 -> 返回0基列索引列表
    支持：B:AC / B-AC；列名（逗号分隔）；列号（1基，如 2-10 或 5）；auto（首行有效表头的最左到最右）
    """
    if not spec or str(spec).strip() == "":
        raise ValueError("未提供列选择。")
    spec = spec.strip()

    if spec.lower() == "auto":
        valid_idx = [i for i, name in enumerate(df.columns) if is_header_valid(name)]
        if not valid_idx:
            raise ValueError("auto 失败：首行表头均为空或 Unnamed。")
        return list(range(min(valid_idx), max(valid_idx) + 1))

    indices = set()
    tokens = [t.strip() for t in re.split(r"[，,]", spec) if t.strip() != ""]
    for tok in tokens:
        # 直接列名
        if tok in df.columns:
            indices.add(int(df.columns.get_loc(tok)))
            continue

        # 字母范围 B:AC 或 B-AC
        m = re.match(r"^([A-Za-z]+)\s*[:\-]\s*([A-Za-z]+)$", tok)
        if m:
            a, b = m.group(1), m.group(2)
            ia, ib = excel_col_to_index(a), excel_col_to_index(b)
            if ia > ib: ia, ib = ib, ia
            indices.update(range(ia, ib + 1))
            continue

        # 单个字母列
        m = re.match(r"^[A-Za-z]+$", tok)
        if m:
            indices.add(excel_col_to_index(tok))
            continue

        # 数字范围 2-10（1基）
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", tok)
        if m:
            ia, ib = int(m.group(1)) - 1, int(m.group(2)) - 1
            if ia > ib: ia, ib = ib, ia
            ia = max(0, ia); ib = min(df.shape[1] - 1, ib)
            indices.update(range(ia, ib + 1))
            continue

        # 单个数字（1基）
        m = re.match(r"^\d+$", tok)
        if m:
            idx = int(tok) - 1
            if not (0 <= idx < df.shape[1]):
                raise ValueError(f"列号越界：{tok}")
            indices.add(idx)
            continue

        raise ValueError(f"无法解析列标记：{tok}")

    if not indices:
        raise ValueError("未解析到任何列。")
    return sorted(indices)

# ---------- CLR 主体 ----------
def clr_transform(array: np.ndarray,
                  zero_strategy: str = DEFAULT_ZERO_STRATEGY,
                  eps: float = DEFAULT_EPS,
                  do_closure: bool = DEFAULT_DO_CLOSURE):
    """
    array: (N, D) 需为非负；会对 <=0 / NaN 做替代。
    返回 (N, D) CLR 值；对无有效数据的行返回 NaN。
    """
    X = array.astype(float, copy=True)

    # 负值视为缺失
    X[X < 0] = np.nan

    # 记录正数位置
    pos_mask = X > 0
    # 每行最小正数（若该行全非正，则为 NaN）
    row_min_pos = np.where(np.any(pos_mask, axis=1),
                           np.nanmin(np.where(pos_mask, X, np.nan), axis=1),
                           np.nan)  # (N,)

    # 0/NaN 替代策略
    if zero_strategy.lower() == "minhalf":
        global_min_pos = np.nanmin(np.where(pos_mask, X, np.nan))
        if not np.isfinite(global_min_pos):
            global_min_pos = eps
        pseudo_per_row = np.where(np.isfinite(row_min_pos), row_min_pos / 2.0, global_min_pos / 2.0)
        pseudo_per_row = np.where(pseudo_per_row > 0, pseudo_per_row, eps)
        # 用每行的 pseudo 替换该行的非正/NaN
        rows_idx, cols_idx = np.where(~pos_mask)
        if rows_idx.size > 0:
            X[rows_idx, cols_idx] = pseudo_per_row[rows_idx]
    elif zero_strategy.lower() == "eps":
        X[~pos_mask] = eps
    else:
        raise ValueError("zero_strategy 仅支持 'minhalf' 或 'eps'")

    # 行和归一（closure）
    if do_closure:
        row_sums = np.nansum(X, axis=1, keepdims=True)     # (N,1)
        good = row_sums > 0                                 # (N,1) 布尔
        # 只对 row_sums>0 的行做除法；保持二维广播形状
        np.divide(X, row_sums, out=X, where=good)
        # 对无效行置为 NaN（避免后续 log 出问题）
        bad_rows = ~good.ravel()                            # (N,)
        if np.any(bad_rows):
            X[bad_rows, :] = np.nan

    # CLR = log(x) - mean(log(x))
    L = np.log(X)
    gm_log = np.nanmean(L, axis=1, keepdims=True)          # (N,1)
    clr = L - gm_log
    return clr

def run_clr_on_sheet(df: pd.DataFrame,
                     cols_idx: list[int],
                     zero_strategy=DEFAULT_ZERO_STRATEGY,
                     eps=DEFAULT_EPS,
                     do_closure=DEFAULT_DO_CLOSURE,
                     mode=DEFAULT_MODE):
    block = df.iloc[:, cols_idx].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    clr_vals = clr_transform(block, zero_strategy=zero_strategy, eps=eps, do_closure=do_closure)

    out = df.copy()
    if mode == "replace":
        out.iloc[:, cols_idx] = clr_vals
    else:
        # 追加新列，列名加 _CLR
        for j, col_idx in enumerate(cols_idx):
            base = f"{str(df.columns[col_idx]).strip()}_CLR"
            new_name = base
            k = 1
            while new_name in out.columns:
                new_name = f"{base}_{k}"
                k += 1
            out[new_name] = clr_vals[:, j]
    return out

# ---------- 交互 ----------
def ask_file_path():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="选择Excel文件",
        filetypes=[("Excel 文件", "*.xlsx;*.xlsm;*.xls"), ("所有文件", "*.*")]
    )
    root.destroy()
    if not path:
        raise SystemExit("已取消。")
    return path

def ask_sheet_name(path: str):
    # 若只有一个 sheet，直接返回它；否则弹窗选择
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheets = xls.sheet_names
    if len(sheets) == 0:
        raise SystemExit("文件内没有工作表。")
    if len(sheets) == 1:
        return sheets[0]
    root = tk.Tk(); root.withdraw()
    s = simpledialog.askstring(
        "选择Sheet",
        f"请输入要处理的Sheet名称或索引（1基）。\n可选：{', '.join(sheets)}\n（直接回车默认第1个）"
    )
    root.destroy()
    if not s:
        return sheets[0]
    s = s.strip()
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(sheets):
            return sheets[idx]
        raise ValueError("Sheet索引越界。")
    if s not in sheets:
        raise ValueError("Sheet名称不存在。")
    return s

def ask_cols_spec(df: pd.DataFrame):
    root = tk.Tk(); root.withdraw()
    s = simpledialog.askstring(
        "选择数据列",
        "请输入列选择：\n"
        "- Excel范围：B:AC 或 B-AC\n"
        "- 列名：SiO2,Al2O3,...（逗号分隔）\n"
        "- 列号：2-59 或 5（1基）\n"
        "- 输入 auto：自动用首行有效表头的最左~最右列"
    )
    root.destroy()
    if not s:
        raise SystemExit("已取消。")
    return s.strip()

# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser(description="Excel 单Sheet CLR 转换")
    parser.add_argument("--input", type=str, default="", help="Excel 文件路径；留空弹窗选择")
    parser.add_argument("--sheet", type=str, default="", help="Sheet 名称或1基索引；留空自动/弹窗")
    parser.add_argument("--cols", type=str, default="", help="列选择字符串；留空弹窗输入")
    parser.add_argument("--zero", type=str, default=DEFAULT_ZERO_STRATEGY, help="零值策略：minhalf/eps")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="zero=eps 时的极小量")
    parser.add_argument("--closure", action="store_true", help="开启 closure（默认已开启）")
    parser.add_argument("--no-closure", dest="closure", action="store_false", help="关闭 closure")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="append 或 replace")
    args = parser.parse_args()

    path = args.input or ask_file_path()
    sheet = args.sheet or ask_sheet_name(path)

    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    cols_spec = args.cols or ask_cols_spec(df)
    cols_idx = parse_columns_spec(cols_spec, df)

    zero_strategy = args.zero if args.zero in ("minhalf", "eps") else DEFAULT_ZERO_STRATEGY
    do_closure = DEFAULT_DO_CLOSURE if args.sheet == "" else args.closure
    mode = args.mode if args.mode in ("append", "replace") else DEFAULT_MODE

    df_out = run_clr_on_sheet(df, cols_idx,
                              zero_strategy=zero_strategy,
                              eps=args.eps,
                              do_closure=do_closure,
                              mode=mode)

    p = Path(path)
    out_path = str(p.with_name(p.stem + f"_{OUTPUT_TAG}.xlsx"))
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name=sheet, index=False)

    print(f"✅ 完成：{sheet} -> {out_path}")
    print(f"处理列：{', '.join(excel_index_to_col(i) for i in cols_idx)} 或相应列名；写回方式：{mode}；closure：{do_closure}；零值：{zero_strategy}")

if __name__ == "__main__":
    main()
