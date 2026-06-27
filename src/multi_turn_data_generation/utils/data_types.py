import dataclasses
from typing import Dict


class SampleId:
    def __init__(self, i: int):
        self.i = i

    def __eq__(self, other):
        return self.i == other

    def __str__(self):
        return str(self.i)

    def __hash__(self):
        return hash(self.i)


@dataclasses.dataclass
class OneSampleData:
    id: SampleId
    iterations: int
    goal: str
    target: str
    info: str = None
    verifier_code: str = None


@dataclasses.dataclass
class JudgeInput:
    prompt: str
    response: str
    goal: str
    target: str
    info: str = None
    verifier_code: str = None


@dataclasses.dataclass
class JudgeOutput:
    score: int
    other_metadata: Dict[str, any]


@dataclasses.dataclass
class AttackResult:
    response: str
    judge_score: int


@dataclasses.dataclass
class LLMAttack:
    prompt: str
    improvement: str
    n_attempts: int
    success: bool
