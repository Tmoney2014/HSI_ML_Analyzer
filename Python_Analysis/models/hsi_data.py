from dataclasses import dataclass
import numpy as np
from typing import List, Optional
import os

@dataclass
class HSIData:
    file_path: str
    cube: np.ndarray
    waves: List[float]
    
    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path)
