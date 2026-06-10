import os
import json

class ExperimentLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.json_dir = os.path.join(exp_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)

    def _convert(self, obj):
        import numpy as np

        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)

        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._convert(v) for v in obj]

        return obj

    def save_json(self, name, obj):
        path = os.path.join(self.json_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(self._convert(obj), f, indent=2)