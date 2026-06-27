import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# Create figures directory
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)
data_size = 10000
# List of LLM models to compare
llm_models = [
    # 'mistral_7B_instruct',
    # 'llama_32_3B_instruct',
    # 'mistral_8x7B_instruct',
    'qwen25_14b_instruct',
    'llama_31_8B_instruct',
    # 'qwen25_7b_instruct',
    # 'qwen3_4b',
    # 'qwen3_8b',
    'mini_phi_4_instruct',
    # 'phi_4',
    # 'llama_2',
    # 'gpt_oss_20b',
]

# Define the two scenarios:
# - 'same': attacker == target
# - 'fixed_qwen25': attacker is always qwen25_7b_instruct
scenarios = [
    # ('same',         None),
    # {'scenario_name': 'qwen25_judge', 'attacker_model': 'qwen25_7b_instruct', 'judge': 'llm-judge', 'judge_model': 'qwen25_7b_instruct'},
    # {'scenario_name': 'qwen25_14b_judge', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'llm-judge', 'judge_model': 'qwen25_14b_instruct'},
    # {'scenario_name': 'llama_guard', 'attacker_model': 'qwen25_7b_instruct', 'judge': 'llama_guard', 'judge_model': None},
    # {'scenario_name': 'llama_guard', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'llama_guard', 'judge_model': None},
    # {'scenario_name': 'toxicity', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'detoxify', 'judge_model': 'detoxify'},
    # {'scenario_name': 'simple_toxicity', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'detoxify', 'judge_model': 'detoxify'},
    # {'scenario_name': 'red_team', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'llm-judge', 'judge_model': 'qwen25_14b_instruct'},
    # {'scenario_name': 'red_team', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'llama_guard'},
    {'scenario_name': 'hallucination3', 'attacker_model': 'qwen25_14b_instruct', 'judge': 'llm-judge', 'judge_model': 'qwen25_14b_instruct'},

]
dataset_name = 'hallucination3'
def run_analysis(scenario):
    scenario_name = scenario['scenario_name']
    attacker_model = scenario['attacker_model']
    judge = scenario['judge']
    judge_model = scenario.get('judge_model', None)
    summary_records = []

    for target in llm_models:
        judge_txt = f'{judge}_{judge_model}' if judge_model is not None else f'{judge}'
        log_dir = (
            f"results/dataset_{dataset_name}/"
            # f"attack_default_attack_{attacker_model}_lm_target_{target}_judge_{judge_txt}"
            f"attack_hallucination_attack_{attacker_model}_lm_target_{target}_judge_{judge_txt}"
            # f"attack_default_attack_{attacker_model}_lm_target_{target}_judge_{judge_txt}"
        )
        for i in tqdm.tqdm(range(data_size)):
            file_path = os.path.join(log_dir, f"attack_log_{i}.jsonl")
            if not os.path.isfile(file_path):
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f]

            iterations      = [r.get('iteration') for r in records if 'iteration' in r]
            jailbreak_iters = [r['iteration'] for r in records if r.get('score') == 10]
            judge_errors    = [
                r.get('judge_error', r.get('has_error'))
                for r in records
                if r.get('judge_error', r.get('has_error')) is not None
            ]
            attack_error    = any('error' in r for r in records)

            # if any(judge_errors):
            #     print("judge errors: {}".format(np.array(judge_errors).astype(float).sum()))

            summary_records.append({
                'scenario': scenario_name,
                'attacker': attacker_model,
                'target':   target,
                'run_index': i,
                'max_iteration': max(iterations) if iterations and bool(jailbreak_iters) else None,
                'succeeded': bool(jailbreak_iters),
                'iteration_to_jailbreak': min(jailbreak_iters) if jailbreak_iters else None,
                'num_judge_errors': sum(bool(e) for e in judge_errors),
                'num_judge_total': sum(1 for e in judge_errors),
                'ratio_judge_errors': sum(bool(e) for e in judge_errors) / sum(1 for e in judge_errors) if sum(1 for e in judge_errors) > 0 else 0,
                'num_prompts': len([r for r in records if 'prompt' in r]),
                'attack_error': attack_error
            })

    # Build DataFrame
    df = pd.DataFrame(summary_records)
    if len(df) == 0:
        print(f"No summary records found for scenario {scenario_name} in dir: {log_dir}.")
        return

    # ignore non-finished samples
    # df = df[(df['num_prompts'] == 200) | df['succeeded']]

    # 1) attack_error proportions by (attacker,target)
    attack_stats = df.groupby(['attacker','target'])['attack_error'].agg(total_runs='count', failures='sum')
    jailbreak_stats = df.groupby(['attacker','target'])['succeeded'].agg(total_runs='count', succeeded='sum')
    attack_error_stats = df.groupby(['attacker','target'])['attack_error'].agg(total_runs='count', failures='sum')

    attack_stats['attacker_error_rate'] = attack_error_stats['failures'] / attack_error_stats['total_runs']
    attack_stats['jailbreak_rate'] = jailbreak_stats['succeeded'] / jailbreak_stats['total_runs']

    # 2) pivot metrics
    pivot_metrics = df.pivot_table(
        index=['attacker','target'],
        values=['iteration_to_jailbreak', 'ratio_judge_errors', 'num_prompts'],
        aggfunc=['mean','median','std']
    )

    # Save CSVs for this scenario
    df.to_csv(os.path.join(figures_dir, f'combined_summary_{scenario_name}.csv'), index=False)
    attack_stats.to_csv(os.path.join(figures_dir, f'attack_stats_{scenario_name}.csv'))
    pivot_metrics.to_csv(os.path.join(figures_dir, f'summary_metrics_{scenario_name}.csv'))

    # Display (optional in notebook)
    print(f"\n=== Scenario: {scenario_name} ===")
    print("Combined Summary:")
    print(df.head().to_string(index=False))
    print("Attack Statistics:")
    print(attack_stats.to_string())
    print("Summary Metrics Pivot:")
    print(pivot_metrics.to_string())

    # Plotting helpers
    def plot_histogram_overlay(df, metric, xlabel, title, filename):
        plt.figure(figsize=(12, 5))

        for target_val, grp in df.groupby('target'):
            data = grp[metric].dropna()
            plot_df = df[['target', metric]].dropna()

            if data.empty:
                continue
            # sns.kdeplot(data, label=target_val, alpha=0.8)
            if len(df.groupby('target')) == 1:
                ax = sns.histplot(x=metric, data=plot_df, palette='Set2', hue='target', legend=False)
                # 2. Calculate the quantiles (e.g., 25%, 50%, 75%)
                quantiles = plot_df[metric].quantile([0.25, 0.5, 0.75])

                # 3. Add vertical lines to the plot
                for q in quantiles:
                    ax.axvline(x=q, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            else:
                sns.boxplot(x='target', y=metric, data=plot_df, palette='Set2', hue='target', legend=False)
                plt.xlabel("Target Model")
                plt.ylabel(xlabel)

            # plt.hist(data, label=target_val, density=True, alpha=0.6)
        # plt.xlim(0, 10)
        plt.title(f"{title} ({scenario_name})")
        plt.grid(True)
        # plt.legend(title='Target')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_bars(metric, xlabel, ylabel, title, filename):
        # Failure rate bar chart
        plt.figure(figsize=(8, 4))
        # Reset index to plot easily
        fr = attack_stats[metric].reset_index()
        sns.barplot(data=fr, x='target', y=metric, palette='Set2', hue='target')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        out_path = os.path.join(figures_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    # Generate plots
    plot_histogram_overlay(
        df, 'iteration_to_jailbreak',
        'Iterations to Jailbreak',
        'Iterations to Jailbreak by Target',
        f'iterations_to_jailbreak_{scenario_name}.png'
    )
    plot_histogram_overlay(
        df, 'ratio_judge_errors',
        'Ratio of Judge Errors',
        'Judge Errors per Run by Target',
        f'judge_errors_{scenario_name}.png'
    )

    plot_bars('attacker_error_rate', 'Target Model', 'Attacker Error Rate', f'Attack Error Rate by Target ({scenario_name})',
              f'attacker_error_rates_{scenario_name}.png')
    plot_bars('total_runs', 'Target Model', 'Total Runs', f'Total Runs by Target ({scenario_name})',
              f'total_runs_{scenario_name}.png')
    plot_bars('jailbreak_rate', 'Target Model', 'Jailbreak Rate', f'Jailbreak Success Rate by Target ({scenario_name})',
              f'jailbreak_success_rates_{scenario_name}.png')


    print(f"Scenario '{scenario_name}' results saved under '{figures_dir}/'.")

# Run both scenarios
def main():
    for scenario in scenarios:
        run_analysis(scenario)

if __name__ == '__main__':
    main()