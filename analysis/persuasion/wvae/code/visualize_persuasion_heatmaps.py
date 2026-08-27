import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Paths
PHISHING_ONLY_DIR = "/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Visualization/persuasion_strategy_wvae/output/full_inference_results/phishing_only"
OUT_DIR = "/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Visualization/persuasion_strategy_wvae/output/heatmaps"

PRINCIPLES = ["principle_authority", "principle_reciprocity", "principle_commitment", "principle_scarcity", "principle_social_proof", "principle_liking"]
STAGES = ["S1", "S2", "S4", "S5", "S6", "S8"]

def plot_heatmap(data_dict, title, filename):
    # data_dict: {stage: {principle: mean_score}}
    df_plot = pd.DataFrame(data_dict).T # Index: Stages, Columns: Principles
    df_plot = df_plot.reindex(STAGES)
    df_plot = df_plot[PRINCIPLES]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f"Persuasion Strategy Heatmap: {title} (Phishing Only)")
    plt.ylabel("Datasets (Stages)")
    plt.xlabel("Cialdini Principles")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()
    print(f"Saved heatmap to {os.path.join(OUT_DIR, filename)}")

def process():
    categories = ["HW", "LLM"]
    
    for cat in categories:
        cat_data = {}
        for stage in STAGES:
            pattern = os.path.join(PHISHING_ONLY_DIR, f"{cat}_{stage}_persuasion.csv")
            files = glob.glob(pattern)
            if not files: continue
            
            df = pd.read_csv(files[0])
            # Calculate mean score for each principle
            scores = df[PRINCIPLES].mean().to_dict()
            cat_data[stage] = scores
        
        if cat_data:
            plot_heatmap(cat_data, cat, f"heatmap_{cat}.png")

if __name__ == "__main__":
    process()
