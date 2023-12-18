from ai2_kit.core.pydantic import BaseModel
from typing import Optional, List, Any


class CllWalkthroughUpdaterInputConfig(BaseModel):
    passing_rate_threshold: float = -1.0
    table: List[Any] = []

