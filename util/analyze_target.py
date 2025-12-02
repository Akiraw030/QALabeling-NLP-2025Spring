import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import os

# 全局設定繪圖風格
sns.set_theme(style="white")

# 1. 定義所有目標 (30個)
ALL_TARGETS = [
    'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
    'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
    'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
    'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
    'question_type_compare', 'question_type_consequence', 'question_type_definition',
    'question_type_entity', 'question_type_instructions', 'question_type_procedure',
    'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
    'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',
    'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure',
    'answer_type_reason_explanation', 'answer_well_written'
]

# 2. 定義分組目標 (用於混合模式)
Q_TARGETS = ALL_TARGETS[:21] # 前 21 個是 Question
A_TARGETS = ALL_TARGETS[21:] # 後 9 個是 Answer

def plot_and_get_clusters(df, subset_cols, title, n_clusters, save_name):
    """
    核心分析函式：
    1. 計算相關性
    2. 畫出階層分群熱圖 (Cluster Map) 並存檔
    3. 輸出文字版的分群建議
    """
    print(f"\n{'='*20} Analyzing: {title} ({len(subset_cols)} targets) {'='*20}")
    
    # 1. 計算 Spearman 相關性矩陣
    corr = df[subset_cols].corr(method='spearman')
    
    # 2. 繪製階層分群熱圖 (Hierarchical Cluster Map)
    print(f"Plotting Cluster Map for {title}...")
    
    # 創建一個大的畫布
    plt.figure(figsize=(15, 15))
    
    # sns.clustermap 會自動進行階層聚類並重新排列矩陣
    # 這樣相關性高的變數會靠在一起，形成明顯的區塊
    g = sns.clustermap(
        corr,
        method='ward',       # 使用 Ward 法，這通常能分出大小比較平均的群
        cmap='coolwarm',     # 藍色負相關，紅色正相關
        annot=False,         # 不顯示數字，避免畫面太亂
        center=0,
        vmin=-1, vmax=1,
        figsize=(16, 16),
        dendrogram_ratio=(.15, .15), # 設定樹狀圖的比例
        cbar_pos=(0.02, 0.8, 0.03, 0.15), # Colorbar 位置
        tree_kws=dict(linewidths=1.5)
    )
    
    g.fig.suptitle(f'{title} Correlation Clusters', fontsize=20, y=1.02)
    
    # 存檔
    save_path = f"{save_name}.png"
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to '{save_path}'")
    
    # 關閉圖表以釋放記憶體
    plt.close('all') 

    # 3. 輸出分群建議 (文字版)
    # 雖然圖畫好了，但我們需要具體的列表來寫進 model.py
    Z = linkage(corr, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    clusters = {}
    for col, label in zip(subset_cols, labels):
        clusters.setdefault(label, []).append(col)
        
    print(f"\n--- Suggested Grouping (Cutting into k={n_clusters} clusters) ---")
    for label, cols in sorted(clusters.items()):
        print(f"  [Group {label}]:")
        for c in cols:
            print(f"    - {c}")
            
    return clusters

def main():
    if os.path.exists('./data/train.csv'):
        df = pd.read_csv('./data/train.csv')
    else:
        print("Error: ./data/train.csv not found. Please ensure the data file is in the current directory.")
        return

    # ---------------------------------------------------------
    # 策略 A: 全局分群 (Global Clustering)
    # 不區分 Q 和 A，看看數據本身最自然的群聚狀態
    # 假設我們想分成 6 個 Head (您可以自由調整 n_clusters)
    # ---------------------------------------------------------
    plot_and_get_clusters(
        df, 
        ALL_TARGETS, 
        title="Strategy A - Global Clustering (All Targets)", 
        n_clusters=6, 
        save_name="cluster_map_global"
    )
    
    # ---------------------------------------------------------
    # 策略 B: 混合分群 (Hybrid / Constrained Clustering)
    # 強制先切開 Q 和 A，然後在內部再細分
    # 這是為了配合我們 model.py 中的 Grouped Heads 架構
    # ---------------------------------------------------------
    
    # 1. 分析 Question Targets (建議分 4 群)
    plot_and_get_clusters(
        df, 
        Q_TARGETS, 
        title="Strategy B - Question Targets Only", 
        n_clusters=4, 
        save_name="cluster_map_question"
    )
    
    # 2. 分析 Answer Targets (建議分 2 群)
    plot_and_get_clusters(
        df, 
        A_TARGETS, 
        title="Strategy B - Answer Targets Only", 
        n_clusters=2, 
        save_name="cluster_map_answer"
    )

if __name__ == '__main__':
    main()