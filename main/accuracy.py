import os
import json
import glob
import pandas as pd
from collections import Counter
from pathlib import Path
from pypinyin import pinyin, Style
from sklearn.metrics import precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# Copy of the list from MemeAnalyzer.py
POLITICIAN_LIST = [
    "賴清德", "曹興誠", "柯建銘", "林智堅", "郭昱晴 (萬老師)", "陳其邁", "邱議瑩",
    "王義川", "陳吉仲", "沈伯洋", "吳崢", "李進勇", "林楚茵", "林俊憲", "賴品妤",
    "吳思瑤", "李俊俋", "蔡英文", "黃捷", "蔡其昌", "劉世芳",
    "卓榮泰", "黃偉哲", "侯友宜", "鄭文燦", "吳欣盈", "江啟臣", "蔣萬安", "蕭美琴", "柯文哲",
    "朱立倫", "韓國瑜", "林佳龍", "郭台銘", "蘇貞昌", "馬英九",
]


def get_pinyin_map():
    mapping = {}
    for name in POLITICIAN_LIST:
        # Standard pinyin with underscores, e.g., "lai_qing_de"
        py = '_'.join([''.join(p) for p in pinyin(name, style=Style.NORMAL)])
        mapping[py] = name
        
        # Handle special cases if any (e.g. 郭昱晴 (萬老師))
        if "(" in name:
            real_name = name.split(" ")[0]
            py_real = '_'.join([''.join(p) for p in pinyin(real_name, style=Style.NORMAL)])
            mapping[py_real] = name # Map "guo_yu_qing" to full string if needed
            
    return mapping

PINYIN_MAP = get_pinyin_map()

def parse_ground_truth(filename):
    """
    Filename format examples: 
    001_lai_qing_de.png
    001_lai_qing_de&zhuo_rong_tai.jpg
    """
    # Remove extension and leading number (e.g., "001_")
    stem = Path(filename).stem
    # Find where the name starts (after the first digit_underscore sequence)
    parts = stem.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        name_part = parts[1]
    else:
        name_part = stem
        
    # Split by '&' for multiple names
    raw_names = name_part.split('&')
    
    gt_names = []
    for name in raw_names:
        # 1. Check if name is already in Chinese politician list (Chinese filename case)
        if name in POLITICIAN_LIST:
            gt_names.append(name)
        # 2. Check if name is Pinyin (Legacy pinyin filename case)
        elif name in PINYIN_MAP:
            gt_names.append(PINYIN_MAP[name])
        else:
            # Fallback/Debugging
            # print(f"Warning: Could not map '{name}' in file '{filename}'")
            pass
            
    return gt_names

def calculate_metrics(df, type_, filter_none=False):
    """
    Calculates micro-averaged Precision, Recall, F1, and collects error data.
    """
    # Work on a copy
    working_df = df.copy()
    none_count = 0

    # 1. Handle Filtering
    if filter_none:
        if type_ == 'face_rec_hit' or type_ == 'face_rec_top':
            # Check if cand_details is empty/None if it exists, else use candidates
            col = 'cand_details' if 'cand_details' in working_df.columns else 'candidates'
            empty_mask = working_df[col].apply(
                lambda x: len(x) == 0 if isinstance(x, (list, set)) else True
            )
        else:
            empty_mask = working_df['llm_pred_names'].apply(
                lambda x: len(x) == 0 if isinstance(x, list) else True
            )
        none_count = empty_mask.sum()
        working_df = working_df[~empty_mask]

    # 2. Extract Predictions based on type
    if type_ == 'face_rec_hit':
        # 將候選名單中的拼音轉回中文
        preds = working_df['candidates'].apply(
            lambda x: set([PINYIN_MAP.get(n, n) for n in x]) if isinstance(x, list) else set()
        )
        
    elif type_ == 'face_rec_top':
        if 'cand_details' in working_df.columns:
            def get_top_per_face_index(details_list):
                if not isinstance(details_list, list) or len(details_list) == 0:
                    return set()
                
                best_faces = {}
                for item in details_list:
                    f_idx = item.get('face_index')
                    prob = item.get('prob', 0)
                    if f_idx not in best_faces or prob > best_faces[f_idx]['prob']:
                        best_faces[f_idx] = item
                # 這裡也將預測出來的拼音轉回中文
                return {PINYIN_MAP.get(v['name'], v['name']) for v in best_faces.values()}

            preds = working_df['cand_details'].apply(get_top_per_face_index)
        else:
            # 備用邏輯同樣加上轉換
            preds = working_df['candidates'].apply(
                lambda x: set([PINYIN_MAP.get(n, n) for n in x]) if isinstance(x, list) else set()
            )
            
    elif type_ == 'llm':
        # Use Chinese names to match ground truth (which is standardized to Chinese)
        col = 'llm_pred_names'
        preds = working_df[col].apply(
            lambda x: set(x) if isinstance(x, list) else set()
        )

    # 3. Get Ground Truth
    gts = working_df['meme_names'].apply(set)

    # 4. Use sklearn for Multi-label Metrics
    mlb = MultiLabelBinarizer(classes=POLITICIAN_LIST)
    y_true = mlb.fit_transform(gts)
    y_pred = mlb.transform(preds)

    # Micro
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    # Macro
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    # Weighted
    p_wei, r_wei, f_wei, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # 5. Calculate Counts for reporting (Micro-based)
    tp_count = (y_true & y_pred).sum()
    fp_count = (~y_true & y_pred).sum()
    fn_count = (y_true & ~y_pred).sum()

    # --- 計算 True Negative (TN) 與 Hamming Accuracy ---
    num_classes = len(POLITICIAN_LIST)
    total_samples = len(gts)
    # 總預測可能數 = 圖片數 * 總名單人數 (36)
    total_possible_predictions = total_samples * num_classes
    # TN = 總可能數 - (TP + FP + FN)
    tn_count = total_possible_predictions - (tp_count + fp_count + fn_count)
    
    # 傳統 Accuracy = (TP + TN) / 全部
    accuracy = (tp_count + tn_count) / total_possible_predictions if total_possible_predictions > 0 else 0.0

    # --- 更新 Print：加入 Micro, Macro, Weighted 與 TN 顯示 ---
    print(f'Type: {type_:<15} | Filter None: {str(filter_none):<5}')
    print(f'  [Micro]    Acc: {accuracy:.4f} | P: {p_mic:.4f} | R: {r_mic:.4f} | F1: {f_mic:.4f}')
    print(f'  [Macro]    P: {p_mac:.4f} | R: {r_mac:.4f} | F1: {f_mac:.4f}')
    print(f'  [Weighted] P: {p_wei:.4f} | R: {r_wei:.4f} | F1: {f_wei:.4f}')
    print(f'  (Counts: TP={tp_count}, FP={fp_count}, FN={fn_count}, TN={tn_count})')
    
    if filter_none and none_count > 0:
        print(f'  (Filtered out {none_count} empty predictions)')

    # 6. For Error Analysis Counters
    fp_analysis = Counter()
    fn_analysis = Counter()
    for p, g in zip(preds, gts):
        fp_analysis.update(p - g)
        fn_analysis.update(g - p)

    # ==========================================
    # 7. 多類別混淆矩陣 (Confusion Matrix)
    # ==========================================
    classes = POLITICIAN_LIST + ["None"]
    cm = pd.DataFrame(0, index=classes, columns=classes)
    
    for actual_set, pred_set in zip(gts, preds):
        # 對角線 (TP): 實際有，且預測正確
        for tp in actual_set.intersection(pred_set):
            if tp in classes:
                cm.loc[tp, tp] += 1
                
        # 右側 (FN): 實際有，但沒預測到 (被歸類為 None)
        for fn in actual_set - pred_set:
            if fn in classes:
                cm.loc[fn, "None"] += 1
                
        # 底部 (FP): 實際沒有，但模型亂猜 (從 None 變成該人物)
        for fp in pred_set - actual_set:
            if fp in classes:
                cm.loc["None", fp] += 1

    # --- 新增：建立專屬資料夾 ---
    output_dir = "confusion_matrices/combine"  # 修改子資料夾名稱以區分測試不同類型資料的結果
    os.makedirs(output_dir, exist_ok=True)  # 如果資料夾不存在就建立，存在就忽略


    # ==========================================
    # 8. 使用 sklearn 繪製混淆矩陣圖表
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支援 Mac 中文字體
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(22, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm.values, display_labels=classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
    plt.title(f"Confusion Matrix: {type_} (Filter None: {filter_none})", fontsize=16)
    
    plot_path = os.path.join(output_dir, f"cm_{type_}_fNone_{filter_none}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'  [+] 混淆矩陣圖表已匯出至: {plot_path}')

    return {
        "accuracy": accuracy, "p_mic": p_mic, "r_mic": r_mic, "f_mic": f_mic,
        "p_mac": p_mac, "r_mac": r_mac, "f_mac": f_mac,
        "p_wei": p_wei, "r_wei": r_wei, "f_wei": f_wei,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
        "fp_analysis": fp_analysis,
        "fn_analysis": fn_analysis,
    }

def print_error_analysis(results, top_n=10):
    """Prints the most common False Positives and False Negatives."""
    fp_counter = results['fp_analysis']
    fn_counter = results['fn_analysis']
    
    print(f"\n--- Top {top_n} Error Analysis for '{results['type']}' (Filter: {results['filter_none']}) ---")
    print(f"Most Common False Positives (Wrong Predictions): {fp_counter.most_common(top_n)}")
    print(f"Most Common False Negatives (Missed Predictions): {fn_counter.most_common(top_n)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate accuracy from meme analysis JSONL results.")
    parser.add_argument("--results_path", type=str, help="Path to the JSONL result file. If not provided, uses the latest file in results/.")
    args = parser.parse_args()

    target_file = args.results_path

    if not target_file:
        # 1. Find latest JSONL
        jsonl_files = glob.glob(os.path.join("results", "*.jsonl"))
        if not jsonl_files:
            print("No JSONL files found in results/ and no path provided.")
            return
        target_file = max(jsonl_files, key=os.path.getctime)
    
    print(f"Processing result file: {target_file}")
    
    # 2. Load Data
    data = []
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file {target_file}: {e}")
        return
    
    df = pd.DataFrame(data)
    
    # 3. Parse Ground Truth (Force parsing to ensure Chinese names are used for consistent evaluation)
    df['meme_names'] = df['meme_path'].apply(lambda p: parse_ground_truth(os.path.basename(p)))
    
    print("-" * 60)
    
    # 4. Run Calculations
    metrics = ['face_rec_hit', 'face_rec_top', 'llm']
    all_results = []
    
    for m in metrics:
        for f in [False, True]:
            res = calculate_metrics(df, type_=m, filter_none=f)
            res['type'] = m
            res['filter_none'] = f
            all_results.append(res)
        print("-" * 60)
    
    # 5. Print Confusion Matrix style error analysis for key models
    for res in all_results:
        # Show error analysis for the main models when they are not filtered
        if res['type'] in ['face_rec_top', 'llm'] and res['filter_none'] is False:
            print_error_analysis(res)

if __name__ == "__main__":
    main()
