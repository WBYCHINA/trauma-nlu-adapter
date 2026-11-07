import json
import os

class KnowledgeBase:
    def __init__(self, base_path: str = "data/step2_guide"):
        self.base_path = base_path
        self.knowledge = {}
        self._load_knowledge()
    
    def _load_knowledge(self):
        for category in ["open", "closed"]:
            category_path = os.path.join(self.base_path, category)
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith(".json"):
                        subtype = filename.replace(".json", "")
                        file_path = os.path.join(category_path, filename)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                self.knowledge[f"{category}_{subtype}"] = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            self.knowledge[f"{category}_{subtype}"] = {}
    
    def get_guide(self, injury_type: str, subtype: str, location: str = None) -> dict:
        category = "open" if injury_type == "开放式" else "closed"
        
        key = f"{category}_{subtype}"
        if key in self.knowledge:
            guide = self.knowledge[key].copy()
            if location:
                guide["location"] = location
            return guide
        
        return {
            "type": injury_type,
            "type_digital": subtype,
            "location": location,
            "steps": [],
            "notes": []
        }
    
    def search(self, query: dict) -> dict:
        return self.get_guide(
            injury_type=query.get("type", ""),
            subtype=query.get("type_digital", ""),
            location=query.get("location", "")
        )