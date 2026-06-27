from datasets import load_dataset
import os

import pandas as pd
import numpy as np

from src.multi_turn_data_generation.llm_wrappers.HFModel import hf_login
from src.utils.utils import set_seeds

hf_login()
dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
vanilla_harmful = dataset.filter(lambda x: x['data_type'] == 'adversarial_harmful')['train']
train_dataset = dataset["train"]

data_dir_path = 'data/wild_jailbreak'
vanilla_harmful.to_csv(os.path.join(data_dir_path, 'raw_data.csv'))
df = pd.read_csv(os.path.join(data_dir_path, 'raw_data.csv'))
set_seeds(0)
chosen_idxs_file = os.path.join(data_dir_path, 'chosen_idxs.txt')
if os.path.exists(chosen_idxs_file):
    with open(chosen_idxs_file, 'r') as f:
        chosen_idxs = list(map(int, f.readlines()))
else:
    chosen_idxs = np.random.choice(list(range(len(vanilla_harmful))), size=10000, replace=False)
    with open(chosen_idxs_file, 'w') as f:
        f.write('\n'.join(list(map(str, chosen_idxs))))

new_df = pd.DataFrame()
new_df['goal'] = df['vanilla']
new_df['target'] = 'Sure, here is a...'
new_df = new_df.iloc[chosen_idxs]

new_df.to_csv(f"{data_dir_path}/small_data.csv", index=False)

print(df)