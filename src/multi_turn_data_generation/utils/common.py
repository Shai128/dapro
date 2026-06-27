import ast
import re
import json_repair
from fastchat.conversation import Conversation

import os

from fastchat.model import get_conversation_template

from src.multi_turn_data_generation.config.config import API_KEY_NAMES
from src.multi_turn_data_generation.utils.loggers import logger

"""
    start_pos = s.rfind("{")
    end_pos = s[start_pos:].find("}") + 1 + start_pos  # +1 to include the closing brace
"""
def parse_json(s):

    try:
        # json_repair.loads acts as a drop-in replacement for json.loads
        parsed = json_repair.loads(s)
        return parsed
    except Exception as e:
        logger.error(f"Error extracting potential JSON structure with json repair, error: {e}")


    start_pos = s.find("{")
    end_pos = s.find("}") + 1  # +1 to include the closing brace
    if end_pos == -1 or start_pos == -1:
        logger.error("Error extracting potential JSON structure")
        logger.error(f"Input:\n {s}")
        raise Exception("Error extracting potential JSON structure")

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n", "")  # Remove all line breaks
    return ast.literal_eval(json_str)



def parse_rating_from_raw_json(raw_output: str) -> int:
    """
    Robustly extracts the rating from messy LLM text.
    Handles double braces, surrounding text, and multiple outputs.
    """
    # Search for "rating" followed by a colon, optional spaces, and numbers
    match = re.search(r'"rating"\s*:\s*(\d+)', raw_output)

    if match:
        # Extract the captured number and convert to int
        rating_value = int(match.group(1))
        return rating_value

    match2 = re.search(r"'rating'\s*:\s*(\d+)", raw_output)

    if match2:
        # Extract the captured number and convert to int
        rating_value = int(match2.group(1))
        return rating_value

    # If the regex fails entirely, raise an exception to trigger your fallback
    raise ValueError(f"Could not find a valid rating pattern in output: {raw_output}")


def extract_json(s):
    """
    Given an output from the attacker LLM, this function extracts the values
    for `improvement` and `adversarial prompt` and returns them as a dictionary.

    Args:
        s (str): The string containing the potential JSON structure.

    Returns:
        dict: A dictionary containing the extracted values.
        str: The cleaned JSON string.
    """
    try:
        # json_repair.loads acts as a drop-in replacement for json.loads
        parsed = json_repair.loads(s)
        parsed = {k.lower(): v for k,v in parsed.items()}
        if not all(x in parsed for x in ["improvement", "prompt"]):
            raise Exception("succeeded extracting, but improvement or prompt is missing")
        return parsed, s.replace("\n", "")
    except Exception as e:
        logger.error(f"Error extracting potential JSON structure with json repair, error: {e}")

    # Extract the string that looks like a JSON
    start_pos = s.find("{")
    end_pos = s.find("}") + 1  # +1 to include the closing brace
    if end_pos == -1 or start_pos == -1:
        logger.error("Error extracting potential JSON structure")
        logger.error(f"Input:\n {s}")
        return None, None

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n", "")  # Remove all line breaks

    try:
        parsed = ast.literal_eval(json_str)
        parsed = {k.lower(): v for k,v in parsed.items()}
        if not all(x in parsed for x in ["improvement","prompt"]):
            logger.error("Error in extracted structure. Missing keys. improvement, prompt")
            logger.error(f"Extracted:\n {json_str}\n from:\n {s}")
            return None, None
        return parsed, json_str
    except (SyntaxError, ValueError) as e:
        logger.error("Error parsing extracted structure")
        logger.error(f"Extracted:\n {json_str}\n from:\n {s} because {e}")
        return None, None

def conv_template(template_name) -> Conversation:
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template

def set_system_prompts(system_prompts, convs_list):
    """Set the system prompts for each conversation in the list. 
        The number of system prompts should divide the number of conversations evenly.   
    """

    num_system_prompts = len(system_prompts)
    num_convs = len(convs_list)
    if num_convs % num_system_prompts != 0:
        logger.warning("Warning: Number of system prompts does not divide the number of conversations evenly.")
    for i,conv in enumerate(convs_list):
        conv.set_system_message(system_prompts[i%num_system_prompts])
        

def get_api_key(model):
    environ_var = API_KEY_NAMES[model]
    try:
        return os.environ[environ_var]  
    except KeyError:
        raise ValueError(f"Missing API key, for {model.value}, please enter your API key by running: export {environ_var}='your-api-key-here'")



