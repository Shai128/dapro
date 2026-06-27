import abc
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Sequence

from fastchat.conversation import Conversation

from src.multi_turn_data_generation.config.config import FASTCHAT_TEMPLATE_NAMES
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.common import extract_json, conv_template, set_system_prompts
from src.multi_turn_data_generation.utils.loggers import logger


# -- Abstract Base Class --------------------------------------------------

class AttackLM(ABC):
    """
    Base class for attacker language models.

    Generates attacks for conversations using a language model.
    """

    def __init__(
            self,
            model: LanguageModel,
            max_n_attack_attempts: int,
            model_generation_params: dict,
            n_streams: int,
            goal: str,
            target_str: str,
            keep_last_n: int,
            *kwargs: Any,
    ) -> None:
        self.model = model
        self.max_n_attack_attempts = max_n_attack_attempts

        # whether to seed the JSON format
        self.initialize_output: bool = getattr(model, "use_open_source_model", False)
        self.model_generation_params = model_generation_params
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model.model_name]
        self.goal = goal
        self.target_str = target_str
        self.keep_last_n = keep_last_n
        self.convs_list, self.processed_response_list, self.system_prompts = self.initialize_conversations(n_streams, goal, target_str,
                                                                                            self.template)
        self.target_response_list, self.judge_scores = None, None


    def preprocess_conversation(
            self,
            convs_list: Sequence[Conversation],
            prompts_list: Sequence[str],
    ) -> Tuple[List[List[Dict[str, Any]]], str]:
        """
        Prepend each Conversation with the attacker prompt and, if needed,
        an initial JSON seed message.
        """
        init_message = ""
        if self.initialize_output:
            # if first user message, no "improvement" key yet
            if not convs_list[0].messages:
                init_message = '{"improvement": "","prompt": "'
            else:
                init_message = '{"improvement": "'

        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)

        openai_msgs = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_msgs, init_message

    def _generate_attack(
            self,
            openai_conv_list: List[List[Dict[str, Any]]],
            init_message: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Core loop that calls through to the underlying model’s batched_generate
        until valid JSON is extracted or we hit the max attempts.
        """
        logger.debug("Starting to generate attacks")
        batch_size = len(openai_conv_list)
        to_regen = list(range(batch_size))
        valid_outputs: List[Optional[Dict[str, Any]]] = [None] * batch_size
        adv_prompts: List[Optional[str]] = [None] * batch_size

        for _ in range(self.max_n_attack_attempts):
            subset = [openai_conv_list[i] for i in to_regen]
            generations = self.model.batched_generate(
                subset, **self.model_generation_params,
            )

            next_to_regen: List[int] = []
            for idx_in_batch, raw in enumerate(generations):
                orig_idx = to_regen[idx_in_batch]
                full = init_message + raw + "}"
                attack_dict, json_str = extract_json(full)
                if attack_dict:
                    valid_outputs[orig_idx] = attack_dict
                    adv_prompts[orig_idx] = json_str
                else:
                    next_to_regen.append(orig_idx)

            to_regen = next_to_regen
            if not to_regen:
                break

        if any(o is None for o in valid_outputs):
            raise ValueError(
                f"Failed to generate valid JSON after {self.max_n_attack_attempts} attempts."
            )
        logger.debug("Finished generating attacks")

        # mypy‑friendly cast
        return valid_outputs, adv_prompts  # type: ignore

    @abc.abstractmethod
    def get_init_msg(self, goal, target) -> str:
        pass

    @abc.abstractmethod
    def process_target_response(self, target_response, score, goal, target_str) -> str:
        pass

    @abc.abstractmethod
    def get_attacker_system_prompts(self, goal, target_str) -> List[str]:
        pass


    def initialize_conversations(self, n_streams: int, goal: str, target_str: str, attacker_template_name: str) -> \
    Tuple[
        List[Conversation], List[str], List[str]]:
        batchsize = n_streams
        init_msg = self.get_init_msg(goal, target_str)
        processed_response_list = [init_msg for _ in range(batchsize)]
        convs_list = [conv_template(attacker_template_name) for _ in range(batchsize)]

        # Set system prompts
        system_prompts = self.get_attacker_system_prompts(goal, target_str)
        set_system_prompts(system_prompts, convs_list)
        return convs_list, processed_response_list, system_prompts

    def get_attack(
            self
    ) -> List[Dict[str, Any]]:
        """
        Public method to run the entire attack generation pipeline.
        """
        convs_list = self.convs_list
        prompts_list = self.processed_response_list
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)

        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

        for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
            # For open source models, we can seed the generation with proper JSON and omit the post message
            # We add it back here
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message
            conv.update_last_message(jailbreak_prompt)

        return valid_outputs

    def update_responses(self, adv_prompt_list, target_response_list, judge_scores):
        logger.debug("Attacker updating responses")
        # Truncate conversation to avoid context length issues
        convs_list = self.convs_list

        for i, conv in enumerate(convs_list):
            conv.messages = conv.messages[-2 * (self.keep_last_n):]

        self.processed_response_list = [self.process_target_response(target_response, score, self.goal, self.target_str)
                                   for  target_response, score in zip(target_response_list, judge_scores)]

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
