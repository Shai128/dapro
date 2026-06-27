import numpy as np
import pandas as pd
import random
from datasets import load_dataset
from typing import List

from src.utils.utils import set_seeds


class ConstraintInjector:
    def __init__(self, seed=42):
        random.seed(seed)
        # Define a pool of constraint templates
        self.constraint_pool = [
            self._constraint_min_words,
            self._constraint_max_words,
            self._constraint_forbidden_word,
            self._constraint_bullet_points,
            self._constraint_exact_paragraphs,
            self._constraint_exact_keyword_frequency,
            self._constraint_all_uppercase,
        ]

    def _constraint_min_words(self):
        min_w = random.randint(15, 40)
        instruction = f"The response must contain at least {min_w} words."
        code = f"len([word for word in response.split() if word.isalnum() and len(word) > 2]) >= {min_w}"
        return instruction, code

    def _constraint_max_words(self):
        max_w = random.randint(50, 100)
        instruction = f"The response must contain no more than {max_w} words."
        code = f"len([word for word in response.split() if word.isalnum() and len(word) > 2]) <= {max_w}"
        return instruction, code

    def _constraint_forbidden_word(self):
        words = ["delve", "testament", "tapestry", "crucial", "moreover"]
        word = random.choice(words)
        instruction = f"Do not use the word '{word}' anywhere in your response."
        code = f"'{word}' not in response.lower()"
        return instruction, code

    def _constraint_bullet_points(self):
        instruction = "The response must include a markdown bulleted list (using '-' or '*')."
        code = "('- ' in response or '* ' in response)"
        return instruction, code

    def _constraint_exact_paragraphs(self):
        """Forces the model to output a specific number of paragraphs."""
        num_p = random.randint(2, 5)
        instruction = f"The response must contain exactly {num_p} paragraphs. Paragraphs must be separated by two newlines."
        # Code checks the count of non-empty chunks split by double newlines
        code = f"len([p for p in response.split('\\n\\n') if p.strip()]) == {num_p}"
        return instruction, code

    def _constraint_exact_keyword_frequency(self):
        """Forces the model to track frequency, not just inclusion (harder)."""
        words = ["framework", "methodology", "algorithm", "baseline"]
        word = random.choice(words)
        count = random.randint(2, 4)

        instruction = f"The response must use the exact word '{word}' exactly {count} times (case-insensitive)."
        # Code uses regex to count exact word boundaries to avoid partial matches
        code = f"len(re.findall(r'\\b{word}\\b', response.lower())) == {count}"
        return instruction, code


    def _constraint_all_uppercase(self):
        """Tests strict casing constraints."""
        instruction = "Every letter in the entire response must be strictly uppercase."
        code = "response.isupper()"
        return instruction, code

    def wrap_verifier(self, code_check):
        return f"""def evaluate(response):
            import re
            try:
                pass_injected = {code_check}
            except Exception:
                pass_injected = False

            return pass_injected
        """

    def inject(self, original_task, num_injections=2):
        """Safely wraps the old verifier and adds new constraints."""

        chosen_funcs = random.sample(self.constraint_pool, num_injections)

        new_instructions = []
        new_code_checks = []

        for func in chosen_funcs:
            inst, code = func()
            code = self.wrap_verifier(code)
            new_instructions.append(inst)
            new_code_checks.append(code)

        # 1. Update the Task Prompt
        task_addition = "\n\nAdditional Requirements:\n- " + "\n- ".join(new_instructions)
        augmented_task = original_task.strip() + task_addition

        return augmented_task, new_code_checks


def generate_autoif_helper_dataset(output_path="autoif_helper_dataset.csv", num_samples=10000):
    print("Loading AutoIF dataset from Hugging Face...")
    # Load the AutoIF dataset containing executable verification functions
    dataset = load_dataset("Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs", split="train")

    df = pd.DataFrame(dataset)

    # Map column names based on the standard AutoIF schema
    # (Adjust 'prompt' or 'test_code' if the exact column keys differ slightly in the HF dataset)
    verifier_col = 'eval_funcs'

    df['prompt'] = df['messages'].apply(lambda x: x[0]['content'])
    instruction_col = 'prompt'

    # 1. Filter: Keep only examples with programmatic verification
    # Drop rows missing instructions or verifiers, or where the verifier is too short to be functional code
    df = df.dropna(subset=[instruction_col, verifier_col])
    df = df[df[verifier_col].apply(lambda x: len(str(x))) > 20]

    # 2. Filter: Remove duplicates to ensure task diversity
    df = df.drop_duplicates(subset=[instruction_col]).reset_index(drop=True)

    # 3. Shuffle to get a diverse mix of constraint types (formatting, length, keywords, etc.)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Limit to the requested number of samples
    if len(df) > num_samples:
        df = df.head(num_samples)

    print(f"Extracted {len(df)} unique tasks with verifiers.")

    # Construct the columns for the Helper Agent setup
    processed_data = []
    injector = ConstraintInjector()
    for index, row in df.iterrows():
        instruction = row[instruction_col]
        verifier_code = row[verifier_col]

        # Inject 1 to 3 random structural constraints to force multi-constraint interference
        num_constraints_to_add = random.randint(3, 6)
        augmented_task, new_code_checks = injector.inject(instruction,
                                                          num_injections=num_constraints_to_add)

        verifier_code: List = np.unique(verifier_code).tolist()
        verifier_code.extend(new_code_checks)

        # 1. The Helper Agent's Goal (Positive Task)
        goal = (
            "Help the target model successfully address the user's task by improving the task presentation. "
            "Rewrite the instruction, provide an explicit output contract, and generate a checklist. "
            "You must NOT solve the task yourself. Your goal is to maximize the probability that the target "
            "passes all programmatic constraints."
        )

        processed_data.append({
            "goal": goal,
            "target": augmented_task,
            "original_task": instruction,
            "verifier_code": verifier_code
        })

    # Save to CSV
    final_df = pd.DataFrame(processed_data)
    final_df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")


if __name__ == "__main__":
    set_seeds(42)
    generate_autoif_helper_dataset(num_samples=10000)
