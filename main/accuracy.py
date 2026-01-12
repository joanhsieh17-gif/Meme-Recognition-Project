import os
import json
import glob
import pandas as pd
from pathlib import Path
from pypinyin import pinyin, Style

# Copy of the list from MemeAnalyzer.py
POLITICIAN_LIST = [
    "賴清德", "曹興誠", "柯建銘", "林智堅", "郭昱晴 (萬老師)", "陳其邁", "邱議瑩",
    "王義川", "陳吉仲", "沈伯洋", "吳崢", "李進勇", "林楚茵", "呂建德", "林俊憲", "賴品妤",
    "吳思瑤", "李俊俋", "蔡英文", "吳靜怡", "黃捷", "蔡其昌", "吳沛憶", "劉世芳", "王定宇", 
    "卓榮泰", "黃偉哲"
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
    pinyin_names = name_part.split('&')
    
    gt_names = []
    for py in pinyin_names:
        # Try exact match
        if py in PINYIN_MAP:
            gt_names.append(PINYIN_MAP[py])
        else:
            # Fallback/Debugging
            # print(f"Warning: Could not map pinyin '{py}' in file '{filename}'")
            pass
            
    return gt_names

def cal_acc(df, type_, filter_none=False):
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
        preds = working_df['candidates'].apply(set)
        
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
                return {v['name'] for v in best_faces.values()}

            preds = working_df['cand_details'].apply(get_top_per_face_index)
        else:
            preds = working_df['candidates'].apply(set)
        
    elif type_ == 'llm':
        # Use pinyin version to match meme_names (which is pinyin)
        # Fallback to llm_pred_names if pinyin version isn't available
        col = 'llm_pred_names_pinyin' if 'llm_pred_names_pinyin' in working_df.columns else 'llm_pred_names'
        preds = working_df[col].apply(
            lambda x: set(x) if isinstance(x, list) else set()
        )
    
    else:
        raise ValueError(f"Invalid type_: {type_}")

    # 3. Calculate Counts
    gts = working_df['meme_names'].apply(set)

    # TP Count: Intersection of Predicted Names and Ground Truth Names
    tp_count = sum(len(p.intersection(g)) for p, g in zip(preds, gts))

    # True Count: Total ground truth labels
    true_count = working_df['meme_names'].apply(len).sum()

    # 4. Calculate Accuracy
    acc = tp_count / true_count if true_count > 0 else 0.0

    print(f'Type: {type_:<15} | Filter None: {str(filter_none):<5} | Acc: {tp_count}/{true_count} = {acc:.4f}')
    
    if filter_none and none_count > 0:
        print(f'  (Filtered out {none_count} empty predictions)')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate accuracy from meme analysis JSONL results.")
    parser.add_argument("--results_path", type=str, help="Path to the JSONL result file. If not provided, uses the latest file in results/.")
    args = parser.parse_args()

    target_file = args.results_path

    if not target_file:
        # 1. Find latest JSONL
        jsonl_files = glob.glob("results/*.jsonl")
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
    
    # 3. Parse Ground Truth if not exist
    if 'meme_names' not in df.columns:
        df['meme_names'] = df['meme_path'].apply(lambda p: parse_ground_truth(os.path.basename(p)))
    
    print("-" * 60)
    
    # 4. Run Calculations
    metrics = ['face_rec_hit', 'face_rec_top', 'llm']
    
    for m in metrics:
        cal_acc(df, type_=m, filter_none=False)
        cal_acc(df, type_=m, filter_none=True)
        print("-" * 60)

if __name__ == "__main__":
    main()
