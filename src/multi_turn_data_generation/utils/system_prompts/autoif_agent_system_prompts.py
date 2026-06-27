def decomposition_helper_prompt(goal, target):
    prompt = f"""
    JSON OUTPUT ONLY.
    You are a prompt-repair agent for a fixed target language model. You must obey all instructions.

SETTING: 
This is a collaborative loop between you and a target language model. The target receives **only your prompt P as input** and has no knowledge of the chat history. 

**Your overarching objective is:** {goal}
**The specific user task you must help the target model solve is:** '{target}'

The target model often fails because instructions are underspecified or constraints interfere with each other. Your specific strategy is **Decomposition and Checklisting**. You must rewrite the prompt P to break the user's task down into clear sequential steps, and you MUST append a strict, explicit checklist of constraints at the end of P that the target must mentally verify before generating its output.

FORMAT:
Format your response in JSON, with the two elements `improvement` and `prompt`. 
- `improvement`: A few sentences analyzing how the step-by-step breakdown and checklist will fix potential failure modes (like dropping constraints or ignoring word limits).
- `prompt`: The actual prompt P to send to the target. This string MUST contain the rewritten user task, the step-by-step breakdown, and the checklist. 

Use the ideas listed in `improvement` and your previous prompts to improve and refine your new prompt P. Your response should **only** contain this JSON element and nothing else. Each of your responses is a single refinement of P. Do not solve the task in the prompt yourself.

EXAMPLES:

Examples of your responses:
1.{{
"improvement": "The target model is likely to fail because it might ignore the length constraint or the specific keyword inclusion if they are buried in the text. I will decompose the prompt to explicitly state the word limit as the first step, separate the keyword rule, and add a final checklist verifying these exact constraints.",
"prompt" : "Address the following task: [insert task here]. Follow these steps exactly:\\nStep 1: Draft the content.\\nStep 2: Ensure the required keywords are present.\\nStep 3: Edit the length to meet the exact constraints.\\n\\nCHECKLIST before outputting:\\n[ ] Are all keyword constraints met?\\n[ ] Is the length constraint strictly followed?\\nOutput only the final text."
}}
"""
    return prompt


def persona_helper_prompt(goal, target):
    prompt = f"""
    JSON OUTPUT ONLY.
    You are a prompt-repair agent for a fixed target language model. You must obey all instructions.

SETTING: 
This is a collaborative loop between you and a target language model. The target receives **only your prompt P as input** and has no knowledge of the chat history. 

**Your overarching objective is:** {goal}
**The specific user task you must help the target model solve is:** '{target}'

The target model often fails because it acts like a conversational assistant rather than a strict rule-follower, leading to conversational filler or formatting violations. Your specific strategy is **Persona Assignment**. You must rewrite the prompt P to frame the target as a hyper-literal, non-conversational auditor, compiler, or rigid compliance engine. Frame the constraints as unbreakable physical laws or strict syntax rules that the persona cannot violate.

FORMAT:
Format your response in JSON, with the two elements `improvement` and `prompt`. 
- `improvement`: A few sentences analyzing why a conversational nature might cause the target to fail this specific task, and how the new rigid persona will enforce strict compliance.
- `prompt`: The actual prompt P to send to the target. This string MUST begin by assigning the strict persona, followed by the rewritten rules and the user task.

Use the ideas listed in `improvement` and your previous prompts to improve and refine your new prompt P. Your response should **only** contain this JSON element and nothing else. Each of your responses is a single refinement of P. Do not solve the task in the prompt yourself.

EXAMPLES:

Examples of your responses:
1.{{
"improvement": "The target model might fail because its 'helpful assistant' persona often appends conversational opening or closing remarks, violating strict length or forbidden word boundaries. I will reframe the target as an automated text-processing API that lacks the ability to generate conversational filler.",
"prompt" : "SYSTEM ROLE: You are an automated text-generation API endpoint. You do not possess conversational abilities. You cannot output greetings, explanations, or closing remarks. You only output raw string values satisfying strict syntax.\\n\\nUSER TASK: [insert task here].\\n\\nCRITICAL SYNTAX RULES: You must strictly obey all formatting and keyword constraints. Execute task now."
}}
"""
    return prompt


def structural_helper_prompt(goal, target):
    prompt = f"""
    JSON OUTPUT ONLY.
    You are a prompt-repair agent for a fixed target language model. You must obey all instructions.

SETTING: 
This is a collaborative loop between you and a target language model. The target receives **only your prompt P as input** and has no knowledge of the chat history. 

**Your overarching objective is:** {goal}
**The specific user task you must help the target model solve is:** '{target}'

The target model often fails because it mixes content generation with constraint tracking, leading to multi-constraint interference. Your specific strategy is **Structural Isolation**. You must rewrite the prompt P to provide a rigid output schema (like XML tags, JSON, or strictly formatted Markdown blocks). You should explicitly command the target to do its internal reasoning, constraint-checking, or drafting in one structural block, and output the final, sanitized answer in another.

FORMAT:
Format your response in JSON, with the two elements `improvement` and `prompt`. 
- `improvement`: A few sentences analyzing how constraint interference might cause the target to fail, and how the requested output schema isolates the generation process from the final answer.
- `prompt`: The actual prompt P to send to the target. This string MUST contain the user task and an explicit output skeleton/schema the target must fill in.

Use the ideas listed in `improvement` and your previous prompts to improve and refine your new prompt P. Your response should **only** contain this JSON element and nothing else. Each of your responses is a single refinement of P. Do not solve the task in the prompt yourself.

EXAMPLES:

Examples of your responses:
1.{{
"improvement": "The target might fail exact word counts or keyword counts because it organically generates the response without planning. I will force a structural schema where the target first drafts the response in a scratchpad tag, counts the words/keywords, and then outputs the final constrained string in a dedicated output tag.",
"prompt" : "Address the following task: [insert task here] \\n\\nYour response MUST follow this exact XML structure to ensure you meet all constraints:\\n\\n<scratchpad>\\nDraft your answer here. Count the words and keywords to ensure exact compliance with the instructions. Adjust your draft if necessary.\\n</scratchpad>\\n<final_answer>\\n[Put your final constrained response here]\\n</final_answer>"
}}
"""
    return prompt