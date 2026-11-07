import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class LLMInference:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", lora_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if lora_path and os.path.exists(lora_path):
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print(f"Loaded LoRA weights from {lora_path}")
            
            print(f"LLM loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("Will use keyword-based intent recognition instead")
    
    def predict_intent(self, input_text: str) -> dict:
        if self.model is None:
            return self._keyword_intent(input_text)
        
        prompt = f"""你是一个急救意图识别助手。请分析用户输入的伤情描述，提取出结构化的意图信息。
用户输入：{input_text}
意图输出："""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                top_k=1,
                repetition_penalty=1.1,
                do_sample=False
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            if "意图输出：" in response:
                json_str = response.split("意图输出：")[1].strip()
                if "{" in json_str and "}" in json_str:
                    start = json_str.index("{")
                    end = json_str.rindex("}") + 1
                    json_str = json_str[start:end]
                    return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return self._keyword_intent(input_text)
    
    def _keyword_intent(self, input_text: str) -> dict:
        injury_type_map = {
            "出血": ["出血", "流血", "喷血", "血"],
            "骨折": ["骨折", "断了", "骨头"],
            "烧伤": ["烧伤", "烫伤", "火"],
            "休克": ["休克", "晕倒", "昏迷"],
            "窒息": ["窒息", "卡住", "呼吸困难"],
            "心肺骤停": ["心跳", "呼吸", "骤停", "没气"],
            "颅脑损伤": ["头", "脑", "昏迷"],
            "胸腹部损伤": ["胸", "腹", "肚子"],
            "脊柱损伤": ["背", "脊柱", "脖子"],
            "溺水": ["溺水", "淹"],
            "电击伤": ["电", "触电"],
            "咬伤": ["咬"]
        }
        
        body_location_map = {
            "头部": ["头", "脑"],
            "颈部": ["脖子", "颈"],
            "胸部": ["胸", "心"],
            "腹部": ["腹", "肚子"],
            "背部": ["背"],
            "右上臂": ["右上臂", "右臂"],
            "左上臂": ["左上臂", "左臂"],
            "右前臂": ["右手", "右臂"],
            "左前臂": ["左手", "左臂"],
            "右大腿": ["右腿", "右大腿"],
            "左大腿": ["左腿", "左大腿"],
            "右小腿": ["右脚", "右小腿"],
            "左小腿": ["左脚", "左小腿"],
            "全身": ["全身", "到处"]
        }
        
        severity_map = {
            "重度": ["喷", "大量", "不停", "严重", "危险"],
            "中度": ["流", "有点", "一些"],
            "轻度": ["轻微", "一点点"]
        }
        
        injury_type = "未知"
        for key, keywords in injury_type_map.items():
            if any(k in input_text for k in keywords):
                injury_type = key
                break
        
        body_location = "未知"
        for key, keywords in body_location_map.items():
            if any(k in input_text for k in keywords):
                body_location = key
                break
        
        severity = "中度"
        for key, keywords in severity_map.items():
            if any(k in input_text for k in keywords):
                severity = key
                break
        
        return {
            "injury_type": injury_type,
            "body_location": body_location,
            "severity": severity,
            "confidence": 0.6
        }
    
    def close(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()
