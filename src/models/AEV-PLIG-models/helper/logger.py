import os
import json

class ExperimentLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.json_dir = os.path.join(exp_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)

    def save_json(self, name, obj):
        path = os.path.join(self.json_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)