import os
import json
from collections import Counter

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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("Loading Baseline records...")
    baseline_data = load_evaluation_records(RES_DIR1)
    
    print("Loading Fine-Tuned records...")
    finetuned_data = load_evaluation_records(RES_DIR2)
    
    print("\n" + "=" * 50)
    calculate_metrics(baseline_data, "BASELINE MODEL")
    calculate_metrics(finetuned_data, "FINE-TUNED MODEL")
    
    display_record_comparison(baseline_data, finetuned_data, COMPARISON_INDEX)
    print("Check update2.")

if __name__ == "__main__":
    main()