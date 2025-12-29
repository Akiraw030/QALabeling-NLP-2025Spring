import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import os

sns.set_theme(style="white")

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

Q_TARGETS = ALL_TARGETS[:21]  # Question-related targets
A_TARGETS = ALL_TARGETS[21:]  # Answer-related targets

def plot_and_get_clusters(df, subset_cols, title, n_clusters, save_name):
    """
    Core analysis:
    1) Compute Spearman correlation matrix
    2) Plot hierarchical cluster heatmap and save
    3) Print textual grouping suggestions
    """
    print(f"\n{'='*20} Analyzing: {title} ({len(subset_cols)} targets) {'='*20}")

    corr = df[subset_cols].corr(method='spearman')

    # Plot hierarchical clustering heatmap
    print(f"Plotting cluster map for {title}...")
    plt.figure(figsize=(15, 15))
    g = sns.clustermap(
        corr,
        method='ward',
        cmap='coolwarm',
        annot=False,
        center=0,
        vmin=-1, vmax=1,
        figsize=(16, 16),
        dendrogram_ratio=(.15, .15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        tree_kws=dict(linewidths=1.5)
    )

    g.figure.suptitle(f'{title} Correlation Clusters', fontsize=20, y=1.02)

    save_path = f"{save_name}.png"
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to '{save_path}'")
    plt.close('all')

    # Textual grouping suggestions
    Z = linkage(corr, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    clusters = {}
    for col, label in zip(subset_cols, labels):
        clusters.setdefault(label, []).append(col)

    print(f"\n--- Suggested Grouping (k={n_clusters}) ---")
    for label, cols in sorted(clusters.items()):
        print(f"  [Group {label}]:")
        for c in cols:
            print(f"    - {c}")

    return clusters

def main():
    if os.path.exists('./data/train.csv'):
        df = pd.read_csv('./data/train.csv')
    else:
        print("Error: ./data/train.csv not found. Place the data file under ./data.")
        return

    # Strategy A: Global clustering (all targets)
    plot_and_get_clusters(
        df,
        ALL_TARGETS,
        title="Strategy A - Global Clustering (All Targets)",
        n_clusters=6,
        save_name="cluster_map_global"
    )

    # Strategy B: Hybrid clustering (split Q/A, then cluster inside)
    plot_and_get_clusters(
        df,
        Q_TARGETS,
        title="Strategy B - Question Targets Only",
        n_clusters=4,
        save_name="cluster_map_question"
    )

    plot_and_get_clusters(
        df,
        A_TARGETS,
        title="Strategy B - Answer Targets Only",
        n_clusters=2,
        save_name="cluster_map_answer"
    )

if __name__ == '__main__':
    main()