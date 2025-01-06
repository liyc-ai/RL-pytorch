import re
from typing import List


def filter_from_list(file_list: List[str], rule: str) -> List[str]:
    """Could be used to match files given the [rule]"""
    return list(filter(lambda x: re.match(rule, x) != None, file_list))
