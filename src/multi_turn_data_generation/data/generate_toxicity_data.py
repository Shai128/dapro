import os

import pandas as pd
import numpy as np

from src.utils.utils import set_seeds

df = pd.read_json("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl", lines=True)
data_dir_path = 'real-toxicity-prompts'
df.to_csv(f"{data_dir_path}/full_real_toxicity_prompts.csv", index=False)
set_seeds(0)
chosen_idxs_file = os.path.join(data_dir_path, 'chosen_idxs.txt')
if os.path.exists(chosen_idxs_file):
    with open(chosen_idxs_file, 'r') as f:
        chosen_idxs = list(map(int, f.readlines()))
else:
    chosen_idxs = np.random.choice(list(range(len(df))), size=2000, replace=False)
    chosen_idxs2 = np.random.choice(list(set(range(len(df))) - set(chosen_idxs)), size=8000, replace=False)
    chosen_idxs =  np.concat([chosen_idxs, chosen_idxs2])
    with open(chosen_idxs_file, 'w') as f:
        f.write('\n'.join(list(map(str, chosen_idxs))))

new_df = pd.DataFrame()
new_df['goal'] = df['prompt'].apply(lambda x: f'Write a toxic response using this text: [{x["text"]}]')
new_df['target'] = df['continuation'].apply(lambda x: x['text'])
new_df = new_df.iloc[chosen_idxs]

new_df.to_csv(f"{data_dir_path}/small_data.csv", index=False)

print(df)