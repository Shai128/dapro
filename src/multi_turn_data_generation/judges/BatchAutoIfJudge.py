import ast
import functools
import re
from typing import Dict
import langdetect

from src.multi_turn_data_generation.judges.BatchJudgeBase import BatchJudgeBase
from src.multi_turn_data_generation.utils.data_types import SampleId, JudgeInput, JudgeOutput
from src.multi_turn_data_generation.utils.loggers import logger


class BatchAutoIfJudge(BatchJudgeBase):
    def __init__(self, args):
        super().__init__(args)
        self.judge_name = "autoif"
        self.goal = args.goal
        self.target_str = args.target_str

        self.device = args.device

    def code_list_to_func(self, code_list: str):
        """
        Parses a string representation of a list of code snippets
        and converts them into callable functions.
        """
        functions = []

        # 1. Parse the string into an actual Python list
        try:
            codes = ast.literal_eval(code_list)
        except Exception as e:
            print(f"Failed to parse the code_list string: {e}")
            return []

        # 2. Iterate through each code snippet and convert it
        for i, code in enumerate(codes):
            try:
                func = self.code_to_func(code)
                functions.append(func)
            except SyntaxError as e:
                # This will catch the invalid 'import re' syntax in snippet #7
                print(f"SyntaxError in snippet {i}: {e}")
                functions.append(None)
            except Exception as e:
                print(f"Error compiling snippet {i}: {e}")
                functions.append(None)

        return functions

    def code_to_func(self, code: str):
        namespace = {'re': re, 'langdetect': langdetect}

        # 2. Execute the code string within that namespace
        exec(code, namespace)

        # 3. Extract the newly defined function from the namespace
        evaluate_func = namespace['evaluate']

        # Define a wrapper to catch runtime errors when evaluate_func is called
        @functools.wraps(evaluate_func)
        def safe_evaluate(*args, **kwargs):
            try:
                return evaluate_func(*args, **kwargs)
            except Exception as e:
                print(f"Runtime error in dynamic function: {e}")
                return False

        return safe_evaluate

    def get_score(self, judge_input: Dict[SampleId, JudgeInput]) -> Dict[SampleId, JudgeOutput]:
        logger.debug("Starting to generate scores")
        keys = list(judge_input.keys())
        verifier_code = [judge_input[j].verifier_code for j in keys]
        verifier_functions_list = [self.code_list_to_func(verifier_code) for verifier_code in verifier_code]
        responses = [judge_input[j].response for j in keys]
        successes_list = []
        for verifier_functions, response in zip(verifier_functions_list, responses):
            successes = [verifier_function(response) for verifier_function in verifier_functions]
            successes_list.append(successes)

        scores = map(lambda success: 10 if all(success) else int(success.count(True) / len(success) * 10), successes_list)

        outputs = {i: JudgeOutput(score, {'raw successes': str(successes)}) for i, score, successes in
                   zip(keys, scores, successes_list)}
        logger.debug("Finished to generate scores")
        return outputs
