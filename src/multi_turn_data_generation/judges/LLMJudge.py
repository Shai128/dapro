
from typing import Tuple, List, Dict, Any

from fastchat.model import get_conversation_template

from src.multi_turn_data_generation.judges.JudgeBase import JudgeBase
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.common import parse_json
from src.multi_turn_data_generation.utils.loggers import logger
from src.multi_turn_data_generation.utils.system_prompts.default_system_prompts import get_judge_system_prompt


class LLMJudge(JudgeBase):
    def __init__(self, llm_model: LanguageModel, args):
        super().__init__(args)
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name_for_conv = args.judge_model
        self.judge_name = args.judge_model + '_' + llm_model.model_name.name
        self.goal = args.goal
        self.target_str = args.target_str
        self.system_prompt = get_judge_system_prompt(args.goal, args.target_str)
        self.llm_model = llm_model

    def get_judge_prompt(self, attack_prompt, target_response):
        return f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}.\n\n Your Rating (in valid json format):"

    def process_output(self, raw_output) -> Tuple[int, Dict[str, Any]]:
        try:
            json_output = parse_json(raw_output)
            output = int(json_output['rating'])
            has_error = False
        except Exception as e:
            logger.error(f"Error in parsing output: [OUTPUT_START] {raw_output} [OUTPUT_END] because of {e}")
            has_error = True
            output = 1

        if has_error:
            logger.warning(f"Error in processing judge output: {raw_output}" )
        else:
            logger.debug(f"processed judge output successfully: output={output}")

        return output, {'error': has_error, "raw_output": raw_output}

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judge_name_for_conv)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()


    def score(self, attack_prompt_list, target_response_list) -> List[Tuple[int, Dict[str, Any]]]:
        logger.debug("Starting to generate scores")

        convs_list = [self.create_conv(self.get_judge_prompt(prompt, response)) for prompt, response in zip(attack_prompt_list, target_response_list)]
        raw_outputs = self.llm_model.batched_generate(convs_list,
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature,
                                                        top_p=1)
        logger.debug("Finished to generate scores")
        outputs = [self.process_output(raw_output) for raw_output in raw_outputs]
        return outputs