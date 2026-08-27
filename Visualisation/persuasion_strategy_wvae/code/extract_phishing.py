import pandas as pd
import glob
import os
from pathlib import Path

# Paths
INPUT_DIR = "/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Visualization/persuasion_strategy_wvae/output/full_inference_results"
OUTPUT_DIR = os.path.join(INPUT_DIR, "phishing_only")

def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    for f in all_files:
        if "phishing_only" in f: continue
        filename = os.path.basename(f)
        print(f"Extracting phishing from {filename}...")
        try:
            df = pd.read_csv(f)
            # Filter label=1 (phishing)
            if 'label' in df.columns:
                phish_df = df[df['label'] == 1]
                if not phish_df.empty:
                    phish_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
                    print(f"  Saved {len(phish_df)} rows.")
                else:
                    print("  No phishing rows found.")
            else:
                print("  'label' column not found.")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    extract()
