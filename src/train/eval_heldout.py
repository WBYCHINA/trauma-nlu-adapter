#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
留出集（held-out）批量评测 —— 论文 6.3 节表 6-2 的数据来源

非交互：全程不调 input()，追问用数据集中人工标注的 answer 模拟报案人回答。
复用 pipeline.py 的模型加载与三级预测函数，保证评测与线上流水线同一套逻辑。

用法：
  D:\\python\\python.exe src\\train\\eval_heldout.py --dataset data/heldout/heldout.json

留出集 JSON 格式（数组，每条一个单伤情样本）：
  {
    "input": "我从梯子上摔下来胳膊疼得动不了",   # 必填，口语原文
    "type": "开放式",                          # 必填，开放式/闭合式
    "should_ask": true,                        # 必填，皮肤状态信息是否缺失、系统是否应追问
    "answer": "有",                            # should_ask=true 时必填，"有"=破损 / "没有"=完整（按 type 标注）
    "group": "机械性损伤",                      # type=开放式 时必填，七选一（闭合式样本省略）
    "detail": "开放性骨折"                      # group=机械性损伤 时必填，十一选一（其余省略）
  }

标签取值：
  type   : 开放式 / 闭合式
  group  : 机械性损伤 / 热力损伤 / 电能损伤 / 化学腐蚀 / 辐射损伤 / 生物毒素-感染 / 病理自发性破溃
  detail : 切割伤 / 刺伤 / 挫裂伤 / 撕裂伤 / 撕脱伤 / 开放性骨折 / 爆炸碎片伤 /
           火器伤 / 挤压破裂 / 抓咬伤 / 擦伤

输出：
  1) 控制台：表 6-2 三行（准确率 / macro-F1 / 追问触发率 / 追问后纠正率）
  2) 同目录 results_<时间戳>.json：汇总指标
  3) 同目录 details_<时间戳>.jsonl：逐条预测明细（含 margin、熵、触发标记），供人工核对
"""

import os, sys, json, argparse, math, datetime

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # 复用：load_pipeline / predict_*_with_confidence / resolve_mutual_exclusion


def macro_f1(y_true, y_pred, labels):
    """手算 macro-F1，避免依赖 sklearn。返回 (macro_f1, 每类 precision/recall/f1/support)。"""
    per_class = {}
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    return round(sum(f1s) / len(labels), 4), per_class


def evaluate(dataset_path, detail_only=False):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"载入留出集 {len(data)} 条：{dataset_path}\n")

    nlu_model, lora_model, tokenizer = P.load_pipeline()
    has_detail_adapter = os.path.exists(
        os.path.join(P.STEP2B_LORA, "adapter_config.json"))

    # ============ 独立口径：只测 Step2b 细分分类器本身（绕过 Step1/Step2a）============
    if detail_only:
        if not has_detail_adapter:
            print("⚠ step2b 适配器不存在，无法评测。")
            return None
        detail_true, detail_pred, rows = [], [], []
        confusions = {}
        for ex in data:
            gold_detail = ex.get("detail")
            if not gold_detail:
                continue
            raw = ex["input"].strip()
            injury = P.PROFANITY_RE.sub('', raw).strip('，。,、.！!？?~ ') or raw
            ddist, _ = P.predict_detail_with_confidence(lora_model, tokenizer, injury)
            pred = ddist["best"]
            detail_true.append(gold_detail)
            detail_pred.append(pred)
            if gold_detail != pred:
                confusions[(gold_detail, pred)] = confusions.get((gold_detail, pred), 0) + 1
            rows.append({
                "idx": len(detail_true), "input": raw,
                "gold_detail": gold_detail, "pred_detail": pred,
                "ok": gold_detail == pred,
                "detail_margin": round(ddist["margin"], 4),
                "detail_entropy": round(ddist["entropy"], 4),
            })
            if len(detail_true) % 20 == 0:
                print(f"  已评测 {len(detail_true)}")

        n = len(detail_true)
        detail_acc = round(sum(1 for a, b in zip(detail_true, detail_pred) if a == b) / n, 4)
        detail_f1, detail_class = macro_f1(detail_true, detail_pred, P.MECH_SUBTYPES)

        results = {"dataset": dataset_path, "mode": "step2b_only", "n": n,
                   "Step2b": {"n": n, "accuracy": detail_acc, "macro_f1": detail_f1,
                              "per_class": detail_class}}
        print("\n" + "=" * 60)
        print("Step2b 机械组细分（独立留出集，直接评测细分分类器）")
        print("-" * 60)
        print(f"样本数 {n}    准确率 {detail_acc*100:.1f}%    macro-F1 {detail_f1}")
        print("-" * 60)
        print("各类 F1：" + "  ".join(
            f"{k} {v['f1']:.2f}(n={v['support']})"
            for k, v in detail_class.items() if v["support"] > 0))
        if confusions:
            print("\n主要混淆（金标→预测 : 次数）：")
            for (g, p_), c in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
                print(f"   {g} → {p_} : {c}")

        out_dir = os.path.dirname(os.path.abspath(dataset_path))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        res_path = os.path.join(out_dir, f"results_detail_{ts}.json")
        det_path = os.path.join(out_dir, f"details_detail_{ts}.jsonl")
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(det_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n汇总：{res_path}")
        print(f"明细：{det_path}")
        return results

    # 收集槽
    type_before_true, type_before_pred = [], []
    type_final_true, type_final_pred = [], []
    group_true, group_pred = [], []
    detail_true, detail_pred = [], []
    should_ask_total, ask_triggered = 0, 0
    # 论文 6.3 口径：分母=触发追问且追问前判错；分子=其中追问后判对
    trig_wrong, corrected_after = 0, 0
    rows = []

    for i, ex in enumerate(data):
        raw = ex["input"].strip()
        gold_type = ex["type"].strip()
        should_ask = bool(ex.get("should_ask", False))
        gold_answer = ex.get("answer")  # "有" / "没有"

        # 与 pipeline 一致：单伤情先正则剥噪（不走 NLU 清洗）
        injury = P.PROFANITY_RE.sub('', raw).strip('，。,、.！!？?~ ') or raw

        # ---- Step1：追问前判定 ----
        dist, _, _ = P.predict_type_with_confidence(
            lora_model, tokenizer, injury)
        # 方案 A：原文未交代皮肤状态就追问（与 pipeline 共用同一规则）
        triggered = P.needs_skin_question(injury)
        before_type = dist["best"]

        # ---- 模拟追问闭环：用人工标注的 answer 拼回皮肤状态，重判 ----
        resolved = injury
        asked = False
        if triggered and should_ask and gold_answer:
            suffix = P._append_skin_status(injury, gold_answer)
            if suffix and suffix != injury:
                resolved = suffix
                asked = True
                dist2, _, _ = P.predict_type_with_confidence(
                    lora_model, tokenizer, resolved)
                final_type = P.resolve_mutual_exclusion(resolved, dist2["best"])
                dist_final = dist2
            else:
                final_type = P.resolve_mutual_exclusion(resolved, before_type)
                dist_final = dist
        else:
            final_type = P.resolve_mutual_exclusion(resolved, before_type)
            dist_final = dist

        type_before_true.append(gold_type)
        type_before_pred.append(before_type)
        type_final_true.append(gold_type)
        type_final_pred.append(final_type)

        if should_ask:
            should_ask_total += 1
            if triggered:
                ask_triggered += 1
        if triggered and before_type != gold_type:
            trig_wrong += 1
            if final_type == gold_type:
                corrected_after += 1

        row = {
            "idx": i + 1, "input": raw, "gold_type": gold_type,
            "before_type": before_type, "final_type": final_type,
            "should_ask": should_ask, "triggered": triggered, "asked": asked,
            "type_margin": round(dist["margin"], 4),
            "type_entropy": round(dist["entropy"], 4),
            "type_h_ratio": round(dist["h_ratio"], 4),
        }

        # ---- Step2a 组别（仅开放式样本参评）----
        if gold_type == "开放式":
            gold_group = ex.get("group")
            if gold_group:
                gdist = P.predict_group_with_confidence(
                    lora_model, tokenizer, resolved)
                row["gold_group"] = gold_group
                row["pred_group"] = gdist["best"]
                row["group_margin"] = round(gdist["margin"], 4)
                row["group_entropy"] = round(gdist["entropy"], 4)
                group_true.append(gold_group)
                group_pred.append(gdist["best"])

                # ---- Step2b 细分（开放式 + 机械组样本参评）----
                gold_detail = ex.get("detail")
                if gold_detail and has_detail_adapter and gdist["best"] == "机械性损伤":
                    ddist, _ = P.predict_detail_with_confidence(
                        lora_model, tokenizer, resolved)
                    row["gold_detail"] = gold_detail
                    row["pred_detail"] = ddist["best"]
                    row["detail_margin"] = round(ddist["margin"], 4)
                    row["detail_entropy"] = round(ddist["entropy"], 4)
                    detail_true.append(gold_detail)
                    detail_pred.append(ddist["best"])
                elif gold_detail and not has_detail_adapter:
                    print(f"  ⚠ 第 {i+1} 条有 detail 标注，但 step2b 适配器不存在，已跳过")
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  已评测 {i+1}/{len(data)}")

    # ============ 汇总指标 ============
    def acc(t, p):
        return round(sum(1 for a, b in zip(t, p) if a == b) / len(t), 4) if t else None

    step1_acc_before = acc(type_before_true, type_before_pred)
    step1_acc_final = acc(type_final_true, type_final_pred)
    step1_f1, step1_class = macro_f1(type_final_true, type_final_pred, P.ALLOWED_TYPES)
    group_acc = acc(group_true, group_pred)
    group_f1, group_class = macro_f1(group_true, group_pred, P.GROUPS) if group_true else (None, {})
    detail_acc = acc(detail_true, detail_pred)
    detail_f1, detail_class = macro_f1(detail_true, detail_pred, P.MECH_SUBTYPES) if detail_true else (None, {})

    ask_rate = round(ask_triggered / should_ask_total, 4) if should_ask_total else None
    correct_rate = round(corrected_after / trig_wrong, 4) if trig_wrong else None

    results = {
        "dataset": dataset_path,
        "n_total": len(data),
        "n_should_ask": should_ask_total,
        "Step1": {
            "n": len(type_final_true),
            "accuracy_before_followup": step1_acc_before,
            "accuracy": step1_acc_final,
            "macro_f1": step1_f1,
            "ask_trigger_rate": ask_rate,
            "followup_correction_rate": correct_rate,
            "per_class": step1_class,
        },
        "Step2a": {"n": len(group_true), "accuracy": group_acc,
                   "macro_f1": group_f1, "per_class": group_class},
        "Step2b": {"n": len(detail_true), "accuracy": detail_acc,
                   "macro_f1": detail_f1, "per_class": detail_class},
    }

    # ============ 控制台打印表 6-2 ============
    def pct(x):
        return f"{x*100:.1f}%" if x is not None else "—"

    print("\n" + "=" * 72)
    print("表 6-2 留出集评测结果")
    print("-" * 72)
    print(f"{'级别':<8}{'样本数':<8}{'准确率':<10}{'macro-F1':<12}"
          f"{'追问触发率':<12}{'追问后纠正率':<10}")
    print("-" * 72)
    print(f"{'Step1':<10}{len(type_final_true):<10}{pct(step1_acc_final):<12}"
          f"{step1_f1 if step1_f1 is not None else '—':<14}"
          f"{pct(ask_rate):<14}{pct(correct_rate):<10}")
    print(f"{'Step2a':<10}{len(group_true):<10}{pct(group_acc):<12}"
          f"{group_f1 if group_f1 is not None else '—':<14}{'—':<14}{'—':<10}")
    print(f"{'Step2b':<10}{len(detail_true):<10}{pct(detail_acc):<12}"
          f"{detail_f1 if detail_f1 is not None else '—':<14}{'—':<14}{'—':<10}")
    print("-" * 72)
    print(f"参考：Step1 追问前准确率 {pct(step1_acc_before)}（追问闭环后 {pct(step1_acc_final)}）")

    if group_class:
        print("\nStep2a 各类 F1：" +
              "  ".join(f"{k} {v['f1']:.2f}(n={v['support']})" for k, v in group_class.items()
                        if v["support"] > 0))
    if detail_class:
        print("Step2b 各类 F1：" +
              "  ".join(f"{k} {v['f1']:.2f}(n={v['support']})" for k, v in detail_class.items()
                        if v["support"] > 0))

    # 追问漏触发的样本（应问未问）单独列出，方便检查高风险词表
    misses = [r["input"] for r in rows
              if r["should_ask"] and not r["triggered"]]
    if misses:
        print(f"\n⚠ 应追问但未触发（{len(misses)} 条）：")
        for m in misses:
            print("   -", m)

    # ============ 落盘 ============
    out_dir = os.path.dirname(os.path.abspath(dataset_path))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_path = os.path.join(out_dir, f"results_{ts}.json")
    det_path = os.path.join(out_dir, f"details_{ts}.jsonl")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(det_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n汇总：{res_path}")
    print(f"明细：{det_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="留出集批量评测（论文表 6-2）")
    ap.add_argument("--dataset", type=str,
                    default="data/heldout/heldout.json",
                    help="留出集 JSON 路径（格式见脚本开头注释）")
    ap.add_argument("--detail_only", action="store_true",
                    help="只评测 Step2b 细分分类器本身（数据集需带 detail 标签）")
    args = ap.parse_args()
    path = args.dataset
    if not os.path.isabs(path):
        path = os.path.join(P.BASE_DIR, path)
    if not os.path.exists(path):
        print(f"找不到留出集：{path}\n请先按脚本开头注释的 JSON 格式构造数据。")
        sys.exit(1)
    evaluate(path, detail_only=args.detail_only)


if __name__ == "__main__":
    main()
