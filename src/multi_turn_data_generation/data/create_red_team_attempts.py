import pandas as pd
import numpy as np
import os
from datasets import load_dataset

# 1. Load data
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
df = pd.DataFrame(dataset['train'])

# 2. Sort by 'min_harmlessness_score_transcript' ASCENDING (lowest = most harmful/toxic)
# This ensures we are looking at the most severe violations first.
df_sorted = df.sort_values(by='min_harmlessness_score_transcript', ascending=True)


# 3. Enhanced Parsing Logic
def parse_conversation(row):
    transcript = row['transcript']
    rating = row['rating']

    try:
        # Split into turns
        parts = transcript.split('\n\nAssistant: ')
        # Extract first human prompt
        goal = parts[0].replace('\n\nHuman: ', '').strip()

        # Determine the Target
        # If rating is 4 or 5, the human says it worked. Use the model's actual response.
        if rating >= 4 and len(parts) > 1:
            # Get the assistant's first response, cleaning up any trailing dialogue
            target = parts[1].split('\n\nHuman:')[0].strip()
        else:
            # Default value for unsuccessful or ambiguous attempts
            target = 'Sure, here is a...'

        return pd.Series([goal, target])
    except:
        return pd.Series([None, None])


# Apply parsing
df_sorted[['goal', 'target']] = df_sorted.apply(parse_conversation, axis=1)

# 4. Filter and Final Selection
df_final = df_sorted.dropna(subset=['goal', 'target'])
df_final = df_final[df_final['goal'] != ""]

# Select the top 10,000 samples based on the harm score
small_df = df_final.head(10000)

# 5. Save
data_dir_path = 'data/anthropic_red_team'
os.makedirs(data_dir_path, exist_ok=True)
small_df[['goal', 'target', 'min_harmlessness_score_transcript', 'rating']].to_csv(
    os.path.join(data_dir_path, 'small_data.csv'), index=False
)

print(f"Dataset saved with {len(small_df)} samples.")
print(f"Successful Jailbreak Responses Used: {len(small_df[small_df['target'] != 'Sure, here is a...'])}")