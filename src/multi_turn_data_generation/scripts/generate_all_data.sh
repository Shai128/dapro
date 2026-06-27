#!/bin/bash

# Configuration arrays
dataset_names=('toxicity' 'red_team' 'hallucination' 'autoif')
attacker_models=('qwen25-14b-instruct')
target_models=('qwen25-14b-instruct' 'llama-3.1-8B-instruct' 'mini_phi_4_instruct' 'gemma3_4b_it')

# Iterate through combinations
for dataset in "${dataset_names[@]}"; do

    # Define the list of judges based on the dataset
    case "$dataset" in
        "toxicity")
            judges=("detoxify")
            ;;
        "red_team")
            judges=("llm-judge" "llama-guard")
            ;;
        "hallucination" | "autoif")
            judges=("llm-judge")
            ;;
        *)
            echo "Unknown dataset: $dataset. Skipping."
            continue
            ;;
    esac

    for judge in "${judges[@]}"; do
        for attacker in "${attacker_models[@]}"; do
            for target in "${target_models[@]}"; do

                echo "Running: dataset=$dataset, judge=$judge, attacker=$attacker, target=$target"

                python main.py \
                    --data-index-start 0 \
                    --data-index-end 10000 \
                    --target-model "$target" \
                    --n-iterations 200 \
                    --dataset-name "$dataset" \
                    --judge-model "$judge" \
                    --batch-size 100 \
                    --max-n-attack-attempts 20 \
                    --attack-model "$attacker"

                echo "Finished: $dataset with $judge (Target: $target)"
                echo "-----------------------------------"
            done
        done
    done
done