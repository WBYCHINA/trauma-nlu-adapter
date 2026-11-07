import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class NLUService:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        print(f"[NLU] Loading model: {self.model_name}")
        print(f"[NLU] Target device: {self.device}")
        
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"[NLU] ✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[NLU] ❌ Failed to load model: {e}")
    
    def _generate(self, prompt: str) -> str:
        if not self.model or not self.tokenizer:
            return ""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                top_k=1,
                repetition_penalty=1.1,
                do_sample=False
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _extract_json(self, response: str) -> dict:
        try:
            if "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        return {}
    
    def detect_type(self, text: str) -> str:
        """步骤1：识别受伤大类型（开放式/闭合式）"""
        if self.model:
            prompt = f'<|im_start|>user\n识别受伤大类型：{text}<|im_end|>\n<|im_start|>assistant\n'
            response = self._generate(prompt)
            result = self._extract_json(response)
            return result.get("type", self._rule_detect_type(text))
        return self._rule_detect_type(text)
    
    def _rule_detect_type(self, text: str) -> str:
        open_keywords = ["流血", "割伤", "划伤", "扎伤", "烫伤", "烧伤", "咬伤", "刺伤", "破皮", "伤口"]
        return "开放式" if any(k in text for k in open_keywords) else "闭合式"
    
    def detect_subtype(self, text: str) -> str:
        """步骤2：识别受伤小类型（出血、骨折等）"""
        if self.model:
            prompt = f'<|im_start|>user\n识别受伤小类型：{text}<|im_end|>\n<|im_start|>assistant\n'
            response = self._generate(prompt)
            result = self._extract_json(response)
            return result.get("type_digital", self._rule_detect_subtype(text))
        return self._rule_detect_subtype(text)
    
    def _rule_detect_subtype(self, text: str) -> str:
        keyword_map = {
            "出血": ["流血", "出血", "喷血"],
            "割伤": ["割伤", "切伤"],
            "擦伤": ["擦伤", "划伤", "破皮"],
            "刺伤": ["刺伤", "扎伤"],
            "烧伤": ["烫伤", "烧伤"],
            "咬伤": ["咬伤", "蛰伤"],
            "骨折": ["骨折", "断了", "骨头"],
            "扭伤": ["扭伤", "崴了"],
            "脱位": ["脱臼"],
            "挫伤": ["挫伤", "撞到", "砸到"],
            "拉伤": ["拉伤"],
            "颅脑损伤": ["头", "脑", "头晕"]
        }
        for key, keywords in keyword_map.items():
            if any(k in text for k in keywords):
                return key
        return "未知"
    
    def detect_location(self, text: str) -> str:
        """步骤3：识别受伤部位（头部、手臂等）"""
        if self.model:
            prompt = f'<|im_start|>user\n识别受伤部位：{text}<|im_end|>\n<|im_start|>assistant\n'
            response = self._generate(prompt)
            result = self._extract_json(response)
            return result.get("location", self._rule_detect_location(text))
        return self._rule_detect_location(text)
    
    def _rule_detect_location(self, text: str) -> str:
        keyword_map = {
            "头部": ["头", "脑"],
            "颈部": ["脖子", "颈"],
            "胸部": ["胸", "胸口"],
            "腹部": ["腹", "肚子"],
            "背部": ["背", "腰"],
            "手臂": ["臂", "胳膊", "肩"],
            "腿部": ["腿", "膝"],
            "手部": ["手", "指", "腕"],
            "脚部": ["脚", "踝", "趾"]
        }
        for key, keywords in keyword_map.items():
            if any(k in text for k in keywords):
                return key
        return "未知"
    
    def classify(self, text: str) -> dict:
        """完整分类：大类型 → 小类型 → 部位"""
        injury_type = self.detect_type(text)
        subtype = self.detect_subtype(text)
        location = self.detect_location(text)
        
        return {
            "type": injury_type,
            "type_digital": subtype,
            "location": location
        }
    
    def chat(self, text: str) -> dict:
        """聊天接口"""
        intent = self.classify(text)
        
        return {
            "reply": f"已识别：大类型={intent['type']}，小类型={intent['type_digital']}，部位={intent['location']}",
            "intent": intent,
            "is_emergency": intent.get("type") == "开放式"
        }