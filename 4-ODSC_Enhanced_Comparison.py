import os
import json
from collections import Counter
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RES_DIR1 = "data/1-Baseline"
RES_DIR2 = "data/3-FineTune-Results"

# Update these to exactly match the prefixes you used in your generation scripts
BASELINE_PREFIX = ""
FINETUNED_PREFIX = ""

# The specific question index you want to inspect side-by-side (matches IND = 70 from NB4)
COMPARISON_INDEX = 70 

# Adding path to where GRPO training saved state

GRPO_STATE_PATH = "data/2-Training/checkpoint-2500/trainer_state.json"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_evaluation_records(directory: str):
    """Loads all JSON evaluation records"""
    records = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return records

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                try:
                    records.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode {filename}")
                    
    # Sort records by their internal index so they are in consistent order
    records.sort(key=lambda x: x.get("index", 0))
    return records

def calculate_metrics(records, run_name: str):
    """Calculates Pass@1 and Majority Vote accuracy."""
    if not records:
        print(f"No records found for {run_name}. Skipping metrics.")
        return

    pass_at_1_correct = 0
    majority_vote_correct = 0
    total_questions = len(records)

    for record in records:
        outputs = record.get("model_outputs", [])
        if not outputs:
            continue
            
        # 1. Pass@1 (First attempt accuracy)
        if outputs[0].get("correct", False):
            pass_at_1_correct += 1
            
        # 2. Majority Vote
        valid_answers = [
            out.get("predicted_answer") for out in outputs 
            if out.get("valid_number", False) and out.get("predicted_answer") is not None
        ]
        
        if valid_answers:
            most_common_answer, _ = Counter(valid_answers).most_common(1)[0]
            gold_clean = record["gold_answer"].replace("$", "").replace(",", "").strip()
            pred_clean = most_common_answer.replace("$", "").replace(",", "").strip()
            
            if pred_clean == gold_clean:
                majority_vote_correct += 1

    pass_1_acc = (pass_at_1_correct / total_questions) * 100
    maj_vote_acc = (majority_vote_correct / total_questions) * 100

    print(f"EVALUATION RESULTS: {run_name}")
    print(f"Total Questions Evaluated:  {total_questions}")
    print(f"Pass@1 Accuracy:            {pass_1_acc:.2f}% ({pass_at_1_correct}/{total_questions})")
    print(f"Majority Vote Accuracy:     {maj_vote_acc:.2f}% ({majority_vote_correct}/{total_questions})")
    print("-" * 50)

def display_record_comparison(baseline_records, finetuned_records, target_index):
    """Finds a specific question index in both datasets and prints them for comparison."""
    print("\n" + "=" * 60)
    print(f" DEEP-DIVE COMPARISON: QUESTION INDEX {target_index}")
    print("=" * 60)

    # Find the records matching the target index
    base_rec = next((r for r in baseline_records if r.get("index") == target_index), None)
    ft_rec = next((r for r in finetuned_records if r.get("index") == target_index), None)

    if base_rec:
        print("\n>>> BASELINE MODEL OUTPUT <<<")
        print("-" * 40)
        print(json.dumps(base_rec, indent=2))
    else:
        print(f"\nWarning: Could not find index {target_index} in Baseline records.")

    if ft_rec:
        print("\n>>> FINE-TUNED MODEL OUTPUT <<<")
        print("-" * 40)
        print(json.dumps(ft_rec, indent=2))
    else:
        print(f"\nWarning: Could not find index {target_index} in Fine-Tuned records.")


def plot_loss_comparison(grpo_state_path: str, baseline_loss_val: float = None, save_path: str = "loss_comparison.png"):
    """
    Extracts loss data from the Hugging Face trainer_state.json and plots it.
    Optionally plots a horizontal baseline deterministic loss for comparison.
    """
    if not os.path.exists(grpo_state_path):
        print(f"\nWarning: Cannot find {grpo_state_path} to plot loss.")
        return

    with open(grpo_state_path, "r") as f:
        state_data = json.load(f)

    steps = []
    losses = []
    
    # Filter the log_history for entries that specifically contain the loss metric
    for entry in state_data.get("log_history", []):
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    if not steps:
        print("\nWarning: No loss data found in the provided state file.")
        return

    # Create the graph
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', label="GRPO Training Loss", color="#1f77b4")

    # If you have a static baseline deterministic loss, plot it as a reference line
    if baseline_loss_val is not None:
        plt.axhline(y=baseline_loss_val, color="#d62728", linestyle='--', linewidth=2, 
                    label=f"Baseline Loss ({baseline_loss_val})")

    plt.title("Model Training Loss over Time", fontsize=14, pad=15)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot locally
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccessfully generated and saved loss comparison graph to '{save_path}'")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    #print("Loading Baseline records...")
    #baseline_data = load_evaluation_records(RES_DIR1)
    
    print("Loading Fine-Tuned records...")
    finetuned_data = load_evaluation_records(RES_DIR2)
    
    print("\n" + "=" * 50)
    #calculate_metrics(baseline_data, "BASELINE MODEL")
    calculate_metrics(finetuned_data, "FINE-TUNED MODEL")

    #plot_loss_comparison(
    #    grpo_state_path=GRPO_STATE_PATH, 
    #    baseline_loss_val=0.100,  
    #    save_path="grpo_vs_baseline_loss-batch12-2500.png"
    #)
    
    #display_record_comparison(baseline_data, finetuned_data, COMPARISON_INDEX)
    #print("Check update2.")

if __name__ == "__main__":
    main()