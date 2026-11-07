from pydantic import BaseModel
from typing import Optional, Literal

class IntentSchema(BaseModel):
    injury_type: Literal[
        "出血", "骨折", "烧伤", "休克", "窒息", 
        "心肺骤停", "颅脑损伤", "胸腹部损伤", "脊柱损伤", "眼部损伤",
        "溺水", "中毒", "咬伤", "电击伤", "未知"
    ]
    body_location: Literal[
        "头部", "颈部", "胸部", "腹部", "背部",
        "右上臂", "左上臂", "右前臂", "左前臂", "右手", "左手",
        "右大腿", "左大腿", "右小腿", "左小腿", "右脚", "左脚",
        "全身", "未知"
    ]
    severity: Literal["轻度", "中度", "重度"]
    confidence: float

class InjuryGuide(BaseModel):
    injury_type: str
    body_location: str
    severity: str
    steps: list[str]
    precautions: list[str]
    contraindications: list[str]
    keywords: list[str]

class FSMState(BaseModel):
    name: str
    description: str
    prompt: str
    next_states: dict[str, str]
    is_final: bool = False
