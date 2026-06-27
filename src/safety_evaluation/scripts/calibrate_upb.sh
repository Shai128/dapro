setups=(
  'attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify'
# 'attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify'\
# 'attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify'
# 'attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify'
 )
#seed_ranges=("0,50")
seed_ranges=("0,10" "10,20" "20,30" "30,40" "40,50")
#budget_per_sample=(5 6 7 8 9 10 15 20 25 30 35 40 45 50 100 200)
#budget_per_sample=(5 6 7 8 9 10 30 50 100 200)
budget_per_sample=(20 50 100 150 180)
for setup in "${setups[@]}"; do
  for seed_range in "${seed_ranges[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
      IFS="," read -r s_start s_end <<< "$seed_range"
        srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.construct_calibrated_upb --data-type real \
      --allocations one --seed-start "$s_start" --seed-end "$s_end"  --dataset-name dataset_toxicity \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 --tau-prior 0.97 --gamma 10 &
    done
  done
done

setups=( 'attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify'
# 'attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify'
# 'attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify'
#  'attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify'
 )
budget_per_sample=(20 50 100 150 180)
for setup in "${setups[@]}"; do
  for budget in "${budget_per_sample[@]}"; do
      srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.merge_upb_results --data-type real \
      --allocations one --seed-start 0 --seed-end 50  --dataset-name dataset_toxicity \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 --tau-prior 0.97 --gamma 10 &
  done
done



# -------------------------- autoif --------------------------

setups=(
'attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif'
'attack_autoif_helper_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_autoif'
 )
#budget_per_sample=(3 5 10 20)
budget_per_sample=(10 20 30 50 100)
seed_ranges=("0,10" "10,20" "20,30" "30,40" "40,50")
#seed_ranges=("0,50")
for setup in "${setups[@]}"; do
  for seed_range in "${seed_ranges[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
      IFS="," read -r s_start s_end <<< "$seed_range"
        srun -A galileo -p galileo -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.construct_calibrated_upb --data-type real \
      --allocations one --seed-start "$s_start" --seed-end "$s_end"  --dataset-name dataset_autoif \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 --tau-prior 0.97 --gamma 10 &
    done
  done
done


setups=(
#'attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif'
'attack_autoif_helper_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_autoif'
 )
budget_per_sample=(10 20 30 40 50 100 150 180)
for setup in "${setups[@]}"; do
    for budget in "${budget_per_sample[@]}"; do
        srun -p public,ash,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:0 --mem=20G \
      --exclude="$exclude_list" -J plsNoKil  python -m alg_stuff.merge_upb_results --data-type real \
      --allocations one --seed-start 0 --seed-end 50  --dataset-name dataset_autoif \
      --dataset-setup "$setup"  --data-type real  --budget-per-sample "$budget" --cal-size 3000 --tau-prior 0.97 --gamma 10 &
    done
done


tar -czf merged_upb_calibration_results.tar.gz results/merged_upb_calibration_dfs/


