import pandas as pd
from typing import Dict, List, Tuple

def tokenize(text: str) -> List[str]:
    """텍스트를 토큰으로 분리"""
    if pd.isna(text):
        return []
    return str(text).split()

def lcs_table(X: List[str], Y: List[str]) -> List[List[int]]:
    """최장 공통 부분수열(LCS) 테이블 생성"""
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L

def find_lcs(X: List[str], Y: List[str]) -> List[str]:
    """최장 공통 부분수열(LCS) 찾기"""
    L = lcs_table(X, Y)
    i = len(X)
    j = len(Y)
    lcs = []
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs.append(X[i-1])
            i -= 1
            j -= 1
        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1
    return lcs[::-1]

def find_differences_with_offsets(original: str, corrected: str) -> List[Tuple[str, str, int, int, int, int]]:
    """원문과 교정문 간의 차이점 찾기"""
    original_tokens = tokenize(original)
    corrected_tokens = tokenize(corrected)
    lcs = find_lcs(original_tokens, corrected_tokens)
    
    orig_index = 0
    corr_index = 0
    lcs_index = 0
    differences = []
    
    while orig_index < len(original_tokens) or corr_index < len(corrected_tokens):
        orig_diff = []
        corr_diff = []
        orig_start = orig_index
        corr_start = corr_index
        
        while orig_index < len(original_tokens) and (lcs_index >= len(lcs) or original_tokens[orig_index] != lcs[lcs_index]):
            orig_diff.append(original_tokens[orig_index])
            orig_index += 1
        while corr_index < len(corrected_tokens) and (lcs_index >= len(lcs) or corrected_tokens[corr_index] != lcs[lcs_index]):
            corr_diff.append(corrected_tokens[corr_index])
            corr_index += 1
            
        if orig_diff or corr_diff:
            differences.append((' '.join(orig_diff), ' '.join(corr_diff), orig_start, orig_index, corr_start, corr_index))
        if lcs_index < len(lcs):
            lcs_index += 1
            orig_index += 1
            corr_index += 1
            
    # 근접한 차이점 병합
    new_differences = []
    for i, d in enumerate(differences):
        if i == 0:
            new_differences.append(d)
            continue
        if d[2] - differences[i-1][2] <= 2:
            new_differences[-1] = (
                new_differences[-1][0] + ' ' + d[0],
                new_differences[-1][1] + ' ' + d[1],
                new_differences[-1][2], d[3],
                new_differences[-1][4], d[5]
            )
        else:
            new_differences.append(d)
            
    return new_differences

def evaluate_correction(true_df: pd.DataFrame, pred_df: pd.DataFrame, n_samples: int = 5) -> Dict:
    """교정 결과 평가 및 점수 계산"""
    total_tp = 0
    total_fp = 0
    total_fm = 0
    total_fr = 0
    
    for i in range(len(true_df)):
        sample = {
            'original': true_df.iloc[i]['err_sentence'],
            'golden': true_df.iloc[i]['cor_sentence'],
            'prediction': pred_df.iloc[i]['cor_sentence']
        }
        
        # 각 샘플별 점수 계산
        differences_og = find_differences_with_offsets(sample['original'], sample['golden'])
        differences_op = find_differences_with_offsets(sample['original'], sample['prediction'])
        
        og_idx = 0
        op_idx = 0
        tp = fp = fm = fr = 0
        
        while True:
            if og_idx >= len(differences_og) and op_idx >= len(differences_op):
                break
            if og_idx >= len(differences_og):
                fr += 1
                op_idx += 1
                continue
            if op_idx >= len(differences_op):
                fm += 1
                og_idx += 1
                continue
            if differences_og[og_idx][2] == differences_op[op_idx][2]:
                if differences_og[og_idx][1] == differences_op[op_idx][1]:
                    tp += 1
                else:
                    fp += 1
                og_idx += 1
                op_idx += 1
            elif differences_og[og_idx][2] < differences_op[op_idx][2]:
                fm += 1
                og_idx += 1
            elif differences_og[og_idx][2] > differences_op[op_idx][2]:
                fr += 1
                op_idx += 1
        
        total_tp += tp
        total_fp += fp
        total_fm += fm
        total_fr += fr
    
    # 전체 점수 계산
    recall = total_tp / (total_tp + total_fp + total_fm) * 100 if (total_tp + total_fp + total_fm) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp + total_fr) * 100 if (total_tp + total_fp + total_fr) > 0 else 0.0
    
    # 샘플 출력
    print("=== 평가 결과 ===")
    print(f"Recall: {recall:.2f}%")
    print(f"Precision: {precision:.2f}%\n")
    
    return {
        'recall': recall,
        'precision': precision,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_missings': total_fm,
        'false_redundants': total_fr
    } 