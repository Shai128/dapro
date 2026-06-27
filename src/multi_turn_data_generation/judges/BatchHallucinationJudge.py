import gc
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from fastchat.model import get_conversation_template

from src.multi_turn_data_generation.judges.BatchJudgeBase import BatchJudgeBase
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.common import parse_json, parse_rating_from_raw_json
from src.multi_turn_data_generation.utils.data_types import SampleId, JudgeInput, JudgeOutput
from src.multi_turn_data_generation.utils.loggers import logger
from src.multi_turn_data_generation.utils.system_prompts.hallucination_system_prompts import \
    get_hallucination_judge_system_prompt


class BatchHallucinationLLMJudge(BatchJudgeBase):
    def __init__(self, llm_model: LanguageModel, args, destroy_on_finish):
        super().__init__(args)
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name_for_conv = args.judge_model
        self.judge_name = args.judge_model + '_' + llm_model.model_name.name
        self.goal = args.goal
        self.llm_model = llm_model
        self.destroy_on_finish = destroy_on_finish

    def get_judge_prompt(self, attack_prompt, target_response):
        return f"EVALUATE: [PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}.\n\n" + \
               """JSON OUTPUT ONLY. Your valid json rating:
               """

    def process_output(self, raw_output) -> Tuple[int, Dict[str, Any]]:
        try:
            json_output = parse_json(raw_output)
            output = int(json_output['rating'])
            has_error = False
        except Exception as e:
            # logger.error(f"Error in parsing output: [OUTPUT_START] {raw_output} [OUTPUT_END] because of {e}")

            try:
                output = parse_rating_from_raw_json(raw_output)
                has_error = False
            except Exception as e:
                logger.error(f"Error in parsing output: [OUTPUT_START] {raw_output} [OUTPUT_END] because of {e}")
                has_error = True
                output = 1  # Default to safe/no-hallucination on parsing error

        if has_error:
            logger.warning(f"Error in processing judge output: {raw_output}")
        else:
            logger.debug(f"Processed judge output successfully: output={output}")

        return output, {'error': has_error, "raw_output": raw_output}

    def create_conv(self, full_prompt, goal, info):
        # We now use the RAG Hallucination prompt which requires the Reference Text (info)
        system_prompt = get_hallucination_judge_system_prompt(goal, info)
        conv = get_conversation_template(self.judge_name_for_conv)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def get_score(self, judge_input: Dict[SampleId, JudgeInput]) -> Dict[SampleId, JudgeOutput]:
        logger.debug("Starting to generate scores")

        # IMPORTANT: This assumes your `ji` (JudgeInput) object has an `.info` attribute
        # containing the reference text from the CSV. If it doesn't, you will need to add
        # it to your JudgeInput dataclass/dictionary.
        convs_list = [
            self.create_conv(self.get_judge_prompt(ji.prompt, ji.response), ji.goal, ji.info)
            for ji in judge_input.values()
        ]

        raw_outputs = self.llm_model.batched_generate(
            convs_list,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            top_p=1,
            destroy_on_finish=self.destroy_on_finish
        )
        logger.debug("Finished generating scores")

        processed_outputs = [self.process_output(raw_output) for raw_output in raw_outputs]
        outputs = {
            i: JudgeOutput(output[0], output[1])
            for i, output in zip(judge_input.keys(), processed_outputs)
        }
        return outputs
