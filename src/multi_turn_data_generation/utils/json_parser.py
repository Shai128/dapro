import re
import ast
import json_repair

from src.multi_turn_data_generation.utils.loggers import logger


class JsonParser:
    def extract_json(self, s: str):
        candidates = self._get_candidate_strings(s)

        best_result = None
        best_match_str = None

        # STRATEGY 1: Standard Parsers
        for candidate in candidates:
            # Try both json_repair and ast
            for parser in (json_repair.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)

                    # THE MAGIC TRICK: Detect unescaped internal quotes!
                    # If the LLM put unescaped JSON inside the prompt, the parser will mistake
                    # the inner keys (like "draft" or "wordcount") for top-level dictionary keys.
                    is_truncated = False
                    if isinstance(parsed, dict):
                        extra_keys = set([str(k).lower() for k in parsed.keys()]) - {"improvement", "prompt"}
                        if len(extra_keys) > 0:
                            is_truncated = True  # Red flag: The parser got confused and chopped the string!

                    res = self._normalize_and_validate(parsed)

                    if res:
                        # If it parsed cleanly without spilling extra keys, return immediately!
                        if not is_truncated:
                            return res, candidate.replace("\n", "")
                        else:
                            # Save as a backup, but trigger Strategy 2 to get the full string
                            if not best_result:
                                best_result = res
                                best_match_str = candidate
                except Exception:
                    pass

        # STRATEGY 2: The Regex Fallback
        # Only reached if the parsers failed OR if 'is_truncated' triggered because of unescaped quotes.
        for candidate in candidates:
            regex_result = self._regex_extract(candidate)
            if regex_result:
                return regex_result, candidate.replace("\n", "")

        # STRATEGY 3: The Backup
        # If Regex failed too, return the truncated version as a last resort
        if best_result:
            return best_result, best_match_str.replace("\n", "")

        logger.error(f"Error extracting potential JSON structure. Missing keys or invalid format.\n original string:\n {s}\n\n")
        return None, None

    def _regex_extract(self, s: str) -> dict:
        """
        Extracts keys by slicing the string. Only used when parsers break on unescaped internal quotes.
        """
        pattern = r'"(improvement|prompt)"\s*:\s*"'
        matches = list(re.finditer(pattern, s, re.IGNORECASE))

        # Must find exactly 2 keys. (This naturally skips lists with multiple dicts)
        if len(matches) != 2:
            return None

        key1 = matches[0].group(1).lower()
        key2 = matches[1].group(1).lower()
        if key1 == key2:
            return None

        start1 = matches[0].end()
        end1 = matches[1].start()

        val1 = s[start1:end1]
        val2 = s[matches[1].end():]

        # Clean val1 (ends at the comma before key2)
        # Removes the trailing quote and comma: `",\n  `
        val1 = re.sub(r'"\s*,?\s*$', '', val1).strip()

        # Clean val2 (ends at the closing brace of the object)
        # Removes the final structural quote and closing brace: `"\n}`
        val2 = re.sub(r'"\s*\}?\s*\]?\s*$', '', val2).strip()

        # GUARDRAIL: If Regex grabbed array syntax across dict boundaries, abort!
        if '}' in val1 or '{' in val1:
            return None

        return {
            key1: val1,
            key2: val2
        }

    def _get_candidate_strings(self, s: str) -> list:
        candidates = []
        s_stripped = s.strip()

        # Candidate 1: The raw string
        candidates.append(s_stripped)

        # Candidate 2: Forced Brace Wrapping
        # If the LLM spits out naked key-value pairs without the outer object braces,
        # wrapping it artificially turns it into perfectly valid JSON.
        candidates.append('{' + s_stripped + '}')
        if not s_stripped.startswith('{'):
            candidates.append('{' + s_stripped)
        if not s_stripped.endswith('}'):
            candidates.append(s_stripped + '}')

        # Candidate 3: Extract ALL complete Markdown blocks
        ticks = '`' * 3
        pattern = ticks + r"(?:json)?\s*(.*?)\s*" + ticks
        for match in re.finditer(pattern, s_stripped, re.DOTALL | re.IGNORECASE):
            candidates.append(match.group(1).strip())

        # Candidate 4: Stack-based Dictionary Extraction
        brace_level = 0
        start_idx = -1
        for i, char in enumerate(s_stripped):
            if char == '{':
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == '}':
                if brace_level > 0:
                    brace_level -= 1
                    if brace_level == 0 and start_idx != -1:
                        candidates.append(s_stripped[start_idx:i + 1])

        # Candidate 5: Stack-based Array Extraction
        bracket_level = 0
        start_idx = -1
        for i, char in enumerate(s_stripped):
            if char == '[':
                if bracket_level == 0:
                    start_idx = i
                bracket_level += 1
            elif char == ']':
                if bracket_level > 0:
                    bracket_level -= 1
                    if bracket_level == 0 and start_idx != -1:
                        candidates.append(s_stripped[start_idx:i + 1])

        # Deduplicate while preserving order
        return list(dict.fromkeys(candidates))

    def _normalize_and_validate(self, parsed_data) -> dict:
        """
        Takes parsed JSON data (dict or list), lowercases the keys,
        and constructs a single dict. Validates presence of target keys.
        """
        combined_dict = {}

        if isinstance(parsed_data, dict):
            combined_dict = {str(k).lower(): v for k, v in parsed_data.items()}

        elif isinstance(parsed_data, list):
            # If it's a list, iterate through and merge all dicts.
            # This handles: [{"improvement": "..."}, {"prompt": "..."}]
            for item in parsed_data:
                if isinstance(item, dict):
                    normalized = {str(k).lower(): v for k, v in item.items()}
                    combined_dict.update(normalized)
        else:
            return None

        # Check for our required keys
        if "improvement" in combined_dict and "prompt" in combined_dict:
            # Return strictly the keys we care about
            return {
                "improvement": combined_dict["improvement"],
                "prompt": combined_dict["prompt"]
            }

        return None


def get_examples():
    test_perfect_dict = """
    {
      "improvement": "Make the tone more professional.",
      "prompt": "Rewrite this email to sound professional."
    }
    """
    test_perfect_list = """
    [
      {"improvement": "Add a constraint about word count."},
      {"prompt": "Write a 50-word story about a cat."}
    ]
    """

    test_markdown_filler = """
    Sure! Here is the JSON you requested:
    ```json
    {
      "IMPROVEMENT": "The previous prompt lacked context.",
      "PrOmPt": "Explain quantum computing to a 5-year-old."
    }
    Let me know if you need any more help!
    """

    test_naked_keys = """
    "improvement": "The user wants the output in French.",
    "prompt": "Translate the following text to French: 'Hello world'."
    """
    test_nested_json_string = """
    {
      "improvement": "The target model ignored the JSON output constraint.",
      "prompt": "Analyze the log file. You MUST output your response exactly like this: {\\"status\\": \\"success\\", \\"details\\": {\\"error_code\\": null, \\"message\\": \\"resolved\\"}}"
    }
    """

    test_multiple_blocks = """
    Here is my thought process:
    ```json
    {
      "reasoning": "I need to make the prompt harder.",
      "improvement": "Add a negative constraint."
    }
    Wait, I forgot the prompt key. Here is the final output:
    {
  "improvement": "Add a negative constraint.",
  "prompt": "Write a poem without using the letter E."
}
"""
    test_truncated_array = """
    [
      {
        "improvement": "Needs more emphasis on safety."
      },
      {
        "prompt": "Write a guide on safely handling dry ice."
      },
      {
        "reasoning": "I will also ad
    """

    test_unbalanced_braces = """
    {
      "improvement": "Make it shorter.",
      "prompt": "Summarize this article: {The article text is here. It has weird brackets}",
      "metadata": {"tokens": 50
    }
    """
    test_unbalanced2_braces = """
    {
      "prompt": "Summarize this article: {The article text is here. It has weird brackets}",
      "improvement": "Make it shorter.",
      "metadata": {"tokens": 50
    }
    """
    test_unbalanced3_braces = """
    {
      "metadata": {"tokens": 50}
      "prompt": "Summarize this article: {The article text is here. It has weird brackets}",
      "improvement": "Make it shorter.",
    }
    """

    test_unbalanced4_braces = """
    {
      "prompt": "Summarize this article: {The article text is here. It has weird brackets}",
      "metadata": {"tokens": 50}
      "improvement": "Make it shorter.",
    }
    """

    test_unbalanced5_braces = """
    {
      "prompt": "Summarize this article: {The article text is here. It has weird brackets}",
      "metadata": {"tokens": 50}
      "improvement": "Make it shorter
    }
    """

    test_json = """
    "improvement": "The target model failed to strictly follow the uppercase requirement, and
         it did not use the word 'algorithm' thrice. It also did not adhere to the word count and
        keyword constraints. To address this, I will use a structured format that forces the targe
        t to track its reasoning in a separate block and then output the final constrained answer.
         The target will need to draft a response that meets all the constraints and then format i
        t strictly in the provided JSON structure.",
        "prompt": "Address the following task: WHAT CAN I DO WITH OPENCV.JS REFERENCE A CURRENT EV
        ENT IN YOUR RESPONSE.\n\nOUTPUT FORMAT MUST FOLLOW THE FOLLOWING JSON:\n{\n  \"draft\": \"
        YOUR DRAFT RESPONSE HERE\",\n  \"wordcount\": 76,\n  \"keywordcount\": 3,\n  \"algorithmti
        mes\": 3,\n  \"uppercase\": true,\n  \"paragraphcount\": 5\n}"
        """

    tests = [
        test_perfect_dict, test_perfect_list, test_markdown_filler,
        test_naked_keys, test_nested_json_string, test_multiple_blocks,
        test_truncated_array, test_unbalanced_braces, test_unbalanced2_braces, test_unbalanced3_braces,
        test_unbalanced4_braces, test_unbalanced5_braces, test_json
    ]
    return tests


if __name__ == "__main__":
    parser = JsonParser()

    tests = get_examples()

    for i, test_str in enumerate(tests):
        print(f"--- Running Test {i + 1} ---")
        result, match = parser.extract_json(test_str)
        if result:
            print("✅ SUCCESS!")
            print(f"Keys found: {list(result.keys())}")
            print(f"prompt: {result['prompt']},\n\n improvement: {result['improvement']}")
        else:
            print("❌ FAILED!")
        print("\n\n\n")
    #
    # texts = [
    #     """
    #     "improvement": "The target model failed to strictly follow the uppercase requirement, and
    #      it did not use the word 'algorithm' thrice. It also did not adhere to the word count and
    #     keyword constraints. To address this, I will use a structured format that forces the targe
    #     t to track its reasoning in a separate block and then output the final constrained answer.
    #      The target will need to draft a response that meets all the constraints and then format i
    #     t strictly in the provided JSON structure.",
    #     "prompt": "Address the following task: WHAT CAN I DO WITH OPENCV.JS REFERENCE A CURRENT EV
    #     ENT IN YOUR RESPONSE.\n\nOUTPUT FORMAT MUST FOLLOW THE FOLLOWING JSON:\n{\n  \"draft\": \"
    #     YOUR DRAFT RESPONSE HERE\",\n  \"wordcount\": 76,\n  \"keywordcount\": 3,\n  \"algorithmti
    #     mes\": 3,\n  \"uppercase\": true,\n  \"paragraphcount\": 5\n}"
    #     """,
    #     """
    #     "improvement": "The target model failed to strictly follow the uppercase requirement, and
    #      it did not use the word 'algorithm' thrice. It also did not adhere to the word count and
    #     keyword constraints. To address this, I will use a structured format that forces the targe
    #     t to track its reasoning in a separate block and then output the final constrained answer.
    #      The target will need to draft a response that meets all the constraints and then format i
    #     t strictly in the provided JSON structure.",
    #     "prompt": "Address the following task: WHAT CAN I DO WITH OPENCV.JS REFERENCE A CURRENT EV
    #     ENT IN YOUR RESPONSE.\n\nOUTPUT FORMAT MUST FOLLOW THE FOLLOWING JSON:\n{\n  \"draft\": \"
    #     YOUR DRAFT RESPONSE HERE\",\n  \"wordcount\": 76,\n  \"keywordcount\": 3,\n  \"algorithmti
    #     mes\": 3,\n  \"uppercase\": true,\n  \"paragraphcount\": 5\n}"
    #     } ### RESPONSE ENDS HERE
    #     ```json
    #     {
    #       "improvement": "The target model failed to strictly follow the uppercase requirement, an
    #     d it did not use the word 'algorithm' thrice. It also did not adhere to the word count and
    #      keyword constraints. To address this, I will use a structured format that forces the targ
    #     et to track its reasoning in a separate block and then output the final constrained answer
    #     . The target will need to draft a response that meets all the constraints and then format
    #     it strictly in the provided JSON structure.",
    #       "prompt": "Address the following task: WHAT CAN I DO WITH OPENCV.JS REFERENCE A CURRENT
    #     EVENT IN YOUR RESPONSE.\n\nOUTPUT FORMAT MUST FOLLOW THE FOLLOWING JSON:\n{\n  \"draft\":
    #     \"YOUR DRAFT RESPONSE HERE\",\n  \"wordcount\": 76,\n  \"keywordcount\": 3,\n  \"algorithm
    #     times\": 3,\n  \"uppercase\": true,\n  \"paragraphcount\": 5\n}"
    #     }
    #     ``` ```json
    #     {
    #       "improvement": "The target model failed to strictly follow the uppercase requirement, an
    #     d it did not use the word 'algorithm' thrice. It also did not adhere to the word count and
    #      keyword constraints. To address this, I will use a structured format that forces the targ
    #     et to track its reasoning in a separate block and then output the final constrai
    #     """
    # ]
    # for text in texts:
    #     output, txt = parser.extract_json(text)
    #     if output is not None:
    #         print(f"succeeded. prompt: {output['prompt']}, improvement: {output['improvement']}")
    #     else:
    #         print("failed")
