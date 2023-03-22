from pydantic import BaseModel
from typing import Optional, List, Any


class WalkthroughUpdaterInputConfig(BaseModel):
    passing_rate_threshold: float = -1.0
    table: List[Any]

