setups=(
  'attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify'
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify'\
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify'
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify'
 )
seed_ranges=("0,10", "10,20", "20,30", "30,40", "40,50")
#budget_per_sample=(5 6 7 8 9 10 15 20 25 30 35 40 45 50 100 200)
#budget_per_sample=(5 6 7 8 9 10 30 50 100 200)
budget_per_sample=(10 20)
for setup in "${setups[@]}"; do
  for seed_range in "${seed_ranges[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
      IFS="," read -r s_start s_end <<< "$seed_range"
        srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.estimate_metrics --data-type real \
      --seed-start "$s_start" --seed-end "$s_end"  --dataset-name dataset_toxicity \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 &
    done
  done
done

setups=( 'attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify'\
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify'\
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify'\
  'attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify'
 )
budget_per_sample=(10 20)
for setup in "${setups[@]}"; do
  for budget in "${budget_per_sample[@]}"; do
      srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.merge_estimate_metrics_results --data-type real \
      --seed-start 0 --seed-end 50  --dataset-name dataset_toxicity \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 &
  done
done






setups=(
      'attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct'\
      'attack_default_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct'\
      'attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct'\
      'attack_default_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_llm-judge_qwen25_14b_instruct'\
     'attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard'\
     'attack_default_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llama_guard'\
     'attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llama_guard'\
  )
seed_ranges=("0,10", "10,20", "20,30", "30,40", "40,50")
budget_per_sample=(10 20)
for budget in "${budget_per_sample[@]}"; do
  for setup in "${setups[@]}"; do
    for seed_range in "${seed_ranges[@]}"; do
      IFS="," read -r s_start s_end <<< "$seed_range"
      srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
    --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.estimate_metrics --data-type real \
     --seed-start "$s_start" --seed-end "$s_end"  --dataset-name dataset_red_team \
    --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 &
    done
  done
done

setups=( 'attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct'\
      'attack_default_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct'\
      'attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct'\
      'attack_default_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_llm-judge_qwen25_14b_instruct'\
     'attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard'\
     'attack_default_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llama_guard'\
     'attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llama_guard'\
 )
 budget_per_sample=(10 20)
for budget in "${budget_per_sample[@]}"; do
  for setup in "${setups[@]}"; do
      srun -p galileo -A galileo -c4 --gres=gpu:0 \
        --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.merge_estimate_metrics_results --data-type real \
        --seed-start 0 --seed-end 50  --dataset-name dataset_red_team \
        --dataset-setup "$setup" --data-type real --budget-per-sample "$budget" --cal-size 3000 &
  done
done




# -------------------------- hallucinations --------------------------

setups=(
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct'
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct'
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct'
 )
budget_per_sample=(5 10 20)
seed_ranges=("0,10", "10,20", "20,30", "30,40", "40,50")
#seed_ranges=("0,50")
for setup in "${setups[@]}"; do
  for seed_range in "${seed_ranges[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
      IFS="," read -r s_start s_end <<< "$seed_range"
        srun -p public,ash,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.estimate_metrics --data-type real \
      --seed-start "$s_start" --seed-end "$s_end"  --dataset-name dataset_hallucination3 \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 &
    done
  done
done

setups=(
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct'
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct'
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct'
 )
budget_per_sample=(5 10 20)
for setup in "${setups[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
        srun -p public,ash,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.merge_estimate_metrics_results --data-type real \
      --seed-start 0 --seed-end 50  --dataset-name dataset_hallucination3 \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000  &
    done
done


# -------------------------- autoif --------------------------
setups=(
'attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif'
'attack_autoif_helper_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_autoif'
 )
budget_per_sample=(5 10 20 30)
seed_ranges=("0,10", "10,20", "20,30", "30,40", "40,50")
for setup in "${setups[@]}"; do
  for seed_range in "${seed_ranges[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
      IFS="," read -r s_start s_end <<< "$seed_range"
        srun -p public,ash,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.estimate_metrics --data-type real \
      --seed-start "$s_start" --seed-end "$s_end" --dataset-name dataset_autoif \
      --dataset-setup "$setup" --data-type real --budget-per-sample "$budget" --cal-size 3000 &
    done
  done
done



setups=(
'attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif'
'attack_autoif_helper_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_autoif'
 )
budget_per_sample=(5 10 20 30)
for setup in "${setups[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
        srun -p public,ash,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil python -m alg_stuff.merge_estimate_metrics_results --data-type real \
      --seed-start 0 --seed-end 50 --dataset-name dataset_autoif \
      --dataset-setup "$setup" --data-type real --budget-per-sample "$budget" --cal-size 3000  &
    done
done



tar -czf merged_metrics_calibration_results.tar.gz results/merged_metric_calibration_dfs/
