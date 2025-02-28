# editor/action_definitions.py

"""
편집 액션 정의. (Collision-free 전제)
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Action:
    action_type: str  # e.g. "Rotate", "Translate", "Replace", ...
    target_object: str
    params: Dict  # e.g. {"angle":90, "distance":1.2, "replace_to":"chair_2",...}

    def to_str(self) -> str:
        """
        For debugging/logging
        """
        return f"{self.action_type}({self.target_object}, params={self.params})"
