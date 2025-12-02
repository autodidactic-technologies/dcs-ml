import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as _dt


class JSONSarsaLogger:
    """
    JSON-lines logger for SARSA-style transitions.
    Each line is a JSON object with fields:
      - s: list[float]           # state_t
      - a: {index:int, params:list[float]}
      - r: float
      - s_next: list[float]
      - a_next: {index:int, params:list[float]} | null
      - done: bool
      - info: {episode:int, t:int, global_step:int}
    A separate metadata JSON file is written alongside (path + '.meta.json').
    """

    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # open in append mode on each write to be robust to crashes
        self._meta_written = False
        self._meta_path = self.path.with_suffix(self.path.suffix + '.meta.json')

    def write_metadata(self, meta: Dict[str, Any]) -> None:
        """Write run metadata once (overwrite)."""
        meta = dict(meta)
        meta.setdefault('created_at', _dt.datetime.utcnow().isoformat() + 'Z')
        with self._meta_path.open('w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._meta_written = True

    def write_transition(self,
                         s: List[float],
                         a_index: int,
                         a_params: List[float],
                         r: float,
                         s_next: List[float],
                         a_next_index: Optional[int],
                         a_next_params: Optional[List[float]],
                         done: bool,
                         episode: int,
                         t_in_ep: int,
                         global_step: int) -> None:
        rec = {
            's': s,
            'a': {'index': int(a_index), 'params': list(map(float, a_params))},
            'r': float(r),
            's_next': s_next,
            'a_next': None if a_next_index is None else {
                'index': int(a_next_index),
                'params': [] if a_next_params is None else list(map(float, a_next_params)),
            },
            'done': bool(done),
            'info': {
                'episode': int(episode),
                't': int(t_in_ep),
                'global_step': int(global_step),
            }
        }
        # Write as one line JSON
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


