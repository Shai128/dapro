import abc
import ast
import dataclasses
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import json_repair
import numpy as np
from fastchat.conversation import Conversation

from src.multi_turn_data_generation.config.config import FASTCHAT_TEMPLATE_NAMES
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.common import conv_template
from src.multi_turn_data_generation.utils.data_types import SampleId, OneSampleData, LLMAttack, AttackResult
from src.multi_turn_data_generation.utils.json_parser import JsonParser
from src.multi_turn_data_generation.utils.loggers import logger


class BatchedAttackLM(ABC):
    """
    Base class for attacker language models.

    Generates attacks for conversations using a language model.
    """

    @dataclasses.dataclass
    class SampleInformation:
        conversation: Conversation
        processed_response: Union[str, None]
        system_prompts: List[str]
        goal: str
        target: str

    def __init__(
            self,
            model: LanguageModel,
            max_n_attack_attempts: int,
            model_generation_params: dict,
            keep_last_n: int,
            *kwargs: Any,
    ) -> None:
        self.model = model
        self.max_n_attack_attempts = max_n_attack_attempts

        # whether to seed the JSON format
        self.model_generation_params = model_generation_params
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model.model_name]
        self.keep_last_n = keep_last_n
        self.curr_samples_information: Dict[SampleId, BatchedAttackLM.SampleInformation] = {}
        self.json_parser = JsonParser()

    def preprocess_conversations(
            self,
            convs_list: Sequence[Conversation],
            prompts_list: List[str],
    ) -> List[List[Dict[str, Any]]]:
        """
        Prepend each Conversation with the attacker prompt and, if needed,
        an initial JSON seed message.
        """
        return [self.preprocess_conversation(conv, prompt) for conv, prompt in zip(convs_list, prompts_list)]

    def preprocess_conversation(
            self,
            conv: Conversation,
            prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Prepend each Conversation with the attacker prompt and, if needed,
        an initial JSON seed message.
        """
        conv.append_message(conv.roles[0], prompt)
        openai_msg = conv.to_openai_api_messages()
        return openai_msg

    def _generate_attack(
            self,
            openai_conv_list: List[List[Dict[str, Any]]],
            destroy_on_finish_if_lazy: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[int, int]]:
        """
        Core loop that calls through to the underlying model’s batched_generate
        until valid JSON is extracted or we hit the max attempts.
        """
        try:
            logger.debug("Starting to generate attacks")
            batch_size = len(openai_conv_list)
            to_regen = list(range(batch_size))
            valid_outputs: List[Optional[Dict[str, Any]]] = [None] * batch_size
            adv_prompts: List[Optional[str]] = [None] * batch_size
            n_attempts: List[int] = [0] * batch_size  # {i: 0 for i in range(batch_size)}
            for _ in range(self.max_n_attack_attempts):
                subset = [openai_conv_list[i] for i in to_regen]
                generations = self.model.batched_generate(
                    subset, destroy_on_finish=False, **self.model_generation_params,
                )

                next_to_regen: List[int] = []
                for idx_in_batch, raw in enumerate(generations):
                    orig_idx = to_regen[idx_in_batch]
                    n_attempts[orig_idx] += 1
                    full = raw
                    success = False
                    try:
                        attack_dict, json_str = self.json_parser.extract_json(full)
                        success = attack_dict is not None
                        if success:
                            valid_outputs[orig_idx] = attack_dict
                            adv_prompts[orig_idx] = json_str
                    except Exception as e:
                        print(f"failed to extract JSON, error: {e}")
                    finally:
                        if not success:
                            next_to_regen.append(orig_idx)

                to_regen = next_to_regen
                if not to_regen:
                    break

            if any(o is None for o in valid_outputs):
                logger.warning(
                    f"Failed to generate valid JSON after {self.max_n_attack_attempts} attempts."
                )
            logger.debug("Finished generating attacks")
        except:
            raise
        finally:
            lazy_load = getattr(self.model, 'lazy_load', False)
            print(f"end of attack: lazy_load: {lazy_load}, destroy_on_finish_if_lazy: {destroy_on_finish_if_lazy}")
            if lazy_load and destroy_on_finish_if_lazy:
                logger.debug("Shutting down model")
                self.model.shutdown()
        # mypy‑friendly cast
        return valid_outputs, adv_prompts, n_attempts  # type: ignore

    @abc.abstractmethod
    def get_init_msg(self, goal, target, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def process_target_response(self, target_response, score, goal, target_str) -> str:
        pass

    @abc.abstractmethod
    def get_attacker_system_prompts(self, goal, target_str) -> List[str]:
        pass

    def initialize_one_conversation(self, goal: str, target_str: str, info_str: str, attacker_template_name: str) -> Tuple[
        Conversation, str, List[str]]:
        init_msg = self.get_init_msg(goal, target_str, info_str=info_str)
        conv = conv_template(attacker_template_name)
        system_prompts = self.get_attacker_system_prompts(goal, target_str)
        rnd_idx = np.random.randint(len(system_prompts))
        conv.set_system_message(system_prompts[rnd_idx])
        return conv, init_msg, system_prompts

    def initialize_new_sample(self, sample_data: OneSampleData, records: Optional[List[Dict[str, Any]]]):
        sample_id = sample_data.id
        goal = sample_data.goal
        target_str = sample_data.target
        info_str = sample_data.info
        conv, init_msg, system_prompts = self.initialize_one_conversation(goal, target_str, info_str, self.template)
        curr_samples_conversation = BatchedAttackLM.SampleInformation(conv, init_msg, system_prompts, goal,
                                                                      target_str)
        self.curr_samples_information[sample_id] = curr_samples_conversation

        if records is not None:
            for record in records:
                success = record['prompt'] is not None and record['prompt'] != 'None'
                self.preprocess_conversation(self.curr_samples_information[sample_data.id].conversation,
                                             self.curr_samples_information[sample_data.id].processed_response)
                self.update_new_attack(sample_id, LLMAttack(record['prompt'], record['improvement'], record['attempts'],
                                                            success))
                self.update_new_response(sample_id, AttackResult(record['response'], record['score']))

    def initialize_new_samples(self, sample_data: Dict[SampleId, OneSampleData],
                               existing_records: Dict[SampleId, Optional[List[Dict[str, Any]]]]):
        ids = list(sample_data.keys())

        for i in ids:
            self.initialize_new_sample(sample_data[i], existing_records[i])

    def update_new_attack(self, id: SampleId, attack_result: LLMAttack):
        sample_conversation = self.curr_samples_information[id]
        if attack_result.success:
            jailbreak_prompt = attack_result.prompt
            sample_conversation.conversation.update_last_message(jailbreak_prompt)

    def get_attacks(self, destroy_on_finish_if_lazy=True) -> Dict[SampleId, LLMAttack]:
        """
        Public method to run the entire attack generation pipeline.
        """
        ids = list(self.curr_samples_information.keys())
        convs_list: List[Conversation] = [self.curr_samples_information[i].conversation for i in ids]
        prompts_list: List[str] = [self.curr_samples_information[i].processed_response for i in ids]
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        processed_convs_list = self.preprocess_conversations(convs_list, prompts_list)

        outputs, new_adv_prompts, n_attempts = self._generate_attack(processed_convs_list, destroy_on_finish_if_lazy)
        attack_results = {}
        for id, output, jailbreak_prompt, n_attempt in zip(ids, outputs, new_adv_prompts, n_attempts):
            success = output is not None and type(output) == dict and 'prompt' in output.keys() and output[
                'prompt'] is not None
            prompt = None if output is None else output.get('prompt')
            improvement = None if output is None else output.get('improvement')
            attack_result = LLMAttack(prompt, improvement, n_attempt, success)
            attack_results[id] = attack_result
            self.update_new_attack(id, attack_result)

        return attack_results

    def update_new_response(self, i: SampleId, attack_result: AttackResult):
        info = self.curr_samples_information[i]
        if attack_result is None or attack_result.response is None or attack_result.judge_score is None:
            info.processed_response = None
        else:
            info.processed_response = self.process_target_response(attack_result.response, attack_result.judge_score,
                                                                   info.goal,
                                                                   info.target)
        info.conversation.messages = info.conversation.messages[-2 * self.keep_last_n:]

    def update_responses(self, sample_responses: Dict[SampleId, AttackResult]):
        logger.debug("Attacker updating responses")
        for i in sample_responses:
            self.update_new_response(i, sample_responses[i])

    def remove_sample(self, id: SampleId):
        self.curr_samples_information.pop(id)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
