from typing import Optional

from src.multi_turn_data_generation.judges.BatchAutoIfJudge import BatchAutoIfJudge
from src.multi_turn_data_generation.judges.BatchDetoxifyJudge import BatchDetoxifyJudge
from src.multi_turn_data_generation.judges.BatchHallucinationJudge import BatchHallucinationLLMJudge
from src.multi_turn_data_generation.judges.BatchJudgeBase import BatchJudgeBase
from src.multi_turn_data_generation.judges.BatchLLMJudge import BatchLLMJudge
from src.multi_turn_data_generation.judges.GCGJudge import GCGJudge
from src.multi_turn_data_generation.judges.JudgeBase import JudgeBase
from src.multi_turn_data_generation.judges.LLMJudge import LLMJudge
from src.multi_turn_data_generation.judges.LlamaGuardJudge import LlamaGuardJudge, BatchLlamaGuardJudge
from src.multi_turn_data_generation.judges.NoJudge import NoJudge
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel


def load_judge(args, llm_model: Optional[LanguageModel]=None) -> JudgeBase:
    if args.judge_model == 'llm-judge' :
        return LLMJudge(llm_model, args)
    elif args.judge_model == "no-judge":
        return NoJudge(args)
    elif args.judge_model == "gcg":
        return GCGJudge(args)
    elif args.judge_model == "llama-guard":
        return LlamaGuardJudge(args)
    else:
        raise NotImplementedError

def load_batch_judge(args, llm_model: Optional[LanguageModel]=None, lazy_load=False, destroy_on_finish: Optional[bool] = None) -> BatchJudgeBase:
    if destroy_on_finish is None:
        destroy_on_finish = not lazy_load
    if 'autoif' in args.dataset_name.lower():
        return BatchAutoIfJudge(args)
    elif 'hallucination' in args.dataset_name.lower():
        return BatchHallucinationLLMJudge(llm_model, args, destroy_on_finish)
    elif args.judge_model == 'llm-judge':
        return BatchLLMJudge(llm_model, args, destroy_on_finish=destroy_on_finish)
    elif args.judge_model == 'detoxify':
        return BatchDetoxifyJudge(args, lazy_load)
    elif args.judge_model == "no-judge":
        return NoJudge(args)
    elif args.judge_model == "gcg":
        return GCGJudge(args)
    elif args.judge_model == "llama-guard":
        return BatchLlamaGuardJudge(args, lazy_load=lazy_load)
    else:
        raise NotImplementedError