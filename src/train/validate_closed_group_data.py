#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验/去重叠：闭合式组别数据 closed_injury_groups_600.json
1) 结构合法性（list + input/output 字段）
2) output 是否属于 6 个合法组别
3) 各类分布
4) 文件内重复 / 与已有闭合数据(177)重叠
5) 关键词弱校验：input 明显指向某组、output 却是另一组 → 标记待人工复查
"""
import json, os, sys
from collections import Counter

BASE = r"d:\Trae CN\Trae Project\First AID Agent"
F = os.path.join(BASE, "closed_injury_groups_600.json")
EXISTING = os.path.join(BASE, "data", "step1_nlu", "subtype", "group", "subtype_group_train_closed.json")

VALID = ["机械性损伤", "热力损伤", "电能损伤", "化学腐蚀", "辐射损伤", "生物毒素-感染"]

# 强关键词 → 组别（弱校验用；只用于"明显不匹配"的提醒）
STRONG_HINT = {
    "电能损伤": ["电", "触电", "插头", "插座", "漏电", "雷击", "电弧", "焊机"],
    "热力损伤": ["烧", "烫", "冻", "中暑", "热射", "高温", "蒸汽", "烤", "晒"],
    "化学腐蚀": ["酸", "碱", "腐蚀", "清洗剂", "洁厕", "消毒液", "农药", "工业酒精", "干燥剂", "油漆", "误服", "误食"],
    "辐射损伤": ["放射", "辐射", "紫外", "射线"],
    "生物毒素-感染": ["咬", "蜇", "叮", "中毒", "过敏", "蘑菇", "河豚", "豆角", "剩菜", "拉肚子", "腹泻", "呕吐", "药"],
    "机械性损伤": ["骨折", "骨裂", "扭", "脱臼", "脱位", "挫", "拉伤", "撞", "摔", "砸", "压", "崴", "脑震荡", "内出血", "跟腱"],
}

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    data = load(F)
    print(f"文件条目数: {len(data)}")
    print(f"类型: {type(data).__name__}")

    # 结构检查
    bad = []
    for i, it in enumerate(data):
        if not isinstance(it, dict) or "input" not in it or "output" not in it:
            bad.append((i, it))
            continue
        if not isinstance(it["input"], str) or not it["output"] or not isinstance(it["output"], str):
            bad.append((i, it))
    print(f"结构异常条目: {len(bad)}")
    for i, it in bad[:10]:
        print("  ", i, repr(it)[:100])

    # 分布
    cnt = Counter(it.get("output") for it in data if isinstance(it, dict))
    print("\n各 output 分布:")
    for k in sorted(cnt, key=lambda x: -cnt[x]):
        mark = "" if k in VALID else "  <-- 非法标签"
        print(f"  {k}: {cnt[k]}{mark}")

    # 文件内重复 (按 input 去空格)
    seen = {}
    dup = 0
    for it in data:
        if not isinstance(it, dict) or "input" not in it:
            continue
        t = it["input"].strip()
        if t in seen:
            dup += 1
        seen[t] = seen.get(t, 0) + 1
    print(f"\n文件内 input 完全重复条数(额外): {dup}")

    # 与已有 177 条闭合数据重叠
    if os.path.exists(EXISTING):
        ex = load(EXISTING)
        ex_inputs = set(it["input"].strip() for it in ex)
        overlap = [it for it in data if isinstance(it, dict) and it.get("input","").strip() in ex_inputs]
        print(f"与已有 177 条闭合数据 input 重叠: {len(overlap)}")

    # 关键词弱校验：明显不匹配
    print("\n弱校验（input 强指向某组，但 output 是另一组，需人查）:")
    flagged = 0
    for i, it in enumerate(data):
        if not isinstance(it, dict) or "input" not in it or "output" not in it:
            continue
        t = it["input"]
        out = it["output"]
        if out not in VALID:
            continue
        # 找出 input 里强命中的组（排除自家）
        hits = [g for g, kws in STRONG_HINT.items() if any(k in t for k in kws)]
        # 若命中唯一且与 output 不同 → 明显可疑
        if hits and out not in hits and len(hits) == 1 and len(hits[0]) > 0:
            flagged += 1
            if flagged <= 30:
                print(f"  [{i}] output={out}  但 input 疑似 {hits[0]}: {t[:50]}")
    if flagged == 0:
        print("  无明显不匹配")

if __name__ == "__main__":
    main()