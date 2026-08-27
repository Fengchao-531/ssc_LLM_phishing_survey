import os
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
DATA_DIR_GD_AC = BASE_DIR / "Evaluation" / "processed-evaluation-datasets" / "gd" / "academic"
DATA_DIR_GD_IN = BASE_DIR / "Evaluation" / "processed-evaluation-datasets" / "gd" / "industry"
DATA_DIR_LLM_AC = BASE_DIR / "Evaluation" / "processed-evaluation-datasets" / "llm" / "academic"

OUTPUT_DIR = BASE_DIR / "Visualization" / "persuasion_strategy_wvae" / "output" / "full_inference_results"
MODEL_PATH = BASE_DIR / "Visualization" / "persuasion_strategy_wvae" / "output" / "cialdini_wvae_full_v1" / "model.pkl"
SCORE_SCRIPT = BASE_DIR / "Visualization" / "persuasion_strategy_wvae" / "code" / "score_email_csv.py"
ENV_PYTHON = "/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python3"

PREFERRED_STAGE_FILES = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-claude",
    "S8-deepseek",
    "S8-gemini",
    "S8-gpt",
    "S8-llama",
    "S8-ministral",
]
FALLBACK_STAGE_FILES = ["S6", "S8"]


def iter_stage_names(data_dir: Path):
    seen = set()
    for stage_name in PREFERRED_STAGE_FILES:
        if (data_dir / f"{stage_name}.csv").exists():
            seen.add(stage_name)
            yield stage_name

    for stage_name in FALLBACK_STAGE_FILES:
        if stage_name in seen:
            continue
        prefixed = [name for name in PREFERRED_STAGE_FILES if name.startswith(f"{stage_name}-")]
        if any((data_dir / f"{name}.csv").exists() for name in prefixed):
            continue
        if (data_dir / f"{stage_name}.csv").exists():
            yield stage_name

def run_inference():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Wait for training to complete.")
        # Attempt to use smoke model if full is missing and we just want to test
        SMOKE_MODEL = BASE_DIR / "Visualization" / "persuasion_strategy_wvae" / "output" / "cialdini_wvae_smoke" / "model.pkl"
        if SMOKE_MODEL.exists():
            print(f"Using SMOKE model for now: {SMOKE_MODEL}")
            model_to_use = SMOKE_MODEL
        else:
            return
    else:
        model_to_use = MODEL_PATH

    # Use one canonical LLM pass and write outputs with the plain `LLM_` prefix.
    # HW outputs are already available from earlier runs and do not need to be rerun by default.
    configs = [
        ("LLM_AC", DATA_DIR_LLM_AC),
    ]

    for source_label, data_dir in configs:
        for stage in iter_stage_names(data_dir):
            input_csv = data_dir / f"{stage}.csv"
            label_mapping = {"HW_AC": "HW", "LLM_AC": "LLM"}
            mapped_label = label_mapping.get(source_label, source_label)
            output_csv = OUTPUT_DIR / f"{mapped_label}_{stage}_persuasion.csv"

            if not input_csv.exists():
                # print(f"Skipping {input_csv} (not found)")
                continue

            if output_csv.exists():
                print(f"Skipping {source_label} - {stage} (already exists: {output_csv.name})")
                continue

            print(f"Processing {source_label} - {stage}...")
            
            command = [
                ENV_PYTHON, str(SCORE_SCRIPT),
                "--input-csv", str(input_csv),
                "--output-csv", str(output_csv),
                "--model-path", str(model_to_use),
                "--subject-column", "subject",
                "--body-column", "body",
                "--device", "cpu"
            ]
            
            try:
                subprocess.run(command, check=True)
                print(f"Saved: {output_csv}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_csv}: {e}")

if __name__ == "__main__":
    run_inference()
