"""Synthetic evaluation harness for experimental signal-importance estimation."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .importance_estimator import (
    ImportanceCandidate, run_importance,
)


@dataclass(frozen=True)
class ExpertImportance:
    candidate_id: str
    importance_tier: str
    score_min: float
    score_max: float
    relative_rank: int
    allowed_tie: bool
    decision_centrality: str
    dependency_role: str
    consequence_strength: str
    priority_expression: str
    attention_strength: str
    explanation: str


@dataclass(frozen=True)
class ImportanceGroup:
    group_id: str
    split: str
    candidates: tuple[ImportanceCandidate, ...]
    expert: tuple[ExpertImportance, ...]
    difficult_pairs: tuple[tuple[str, str, str], ...]
    has_overlap: bool
    has_tie: bool
    difficulty: str


DIMENSION_PATTERNS = (
    ("critical", 1, "decisive", "gating_condition", "decision_stopping", "explicitly_required", "brief", 0.91),
    ("high", 2, "major", "outcome_driver", "materially_changes_decision", "explicitly_emphasized", "developed", 0.74),
    ("medium", 3, "contributing", "no_dependency", "influences_preference", "mentioned_once", "developed", 0.51),
    ("low", 4, "minor", "comparative_preference", "limited_effect", "mentioned_once", "repeated", 0.34),
    ("peripheral", 5, "contextual", "supporting_detail", "no_stated_effect", "unclear", "dominant", 0.16),
)
TOP_CATEGORIES = ("contract timing", "operational support", "client ownership", "succession", "compensation", "technology dependency", "values alignment", "product access")


def build_importance_dataset() -> list[ImportanceGroup]:
    groups = []
    for group_number in range(1, 41):
        group_id = f"importance-{group_number:02d}"
        split = "development" if group_number <= 28 else "holdout"
        rotation = (group_number - 1) % len(TOP_CATEGORIES)
        categories = [TOP_CATEGORIES[(rotation + offset) % len(TOP_CATEGORIES)] for offset in range(5)]
        candidates = []
        experts = []
        for index, (tier, rank, centrality, dependency, consequence, priority, attention, score) in enumerate(DIMENSION_PATTERNS, 1):
            candidate_id = f"{group_id}-c{index}"
            item_type = "blocker" if index in {1, 3} else "driver"
            verdict = "needs_review" if index == 4 and group_number % 3 == 0 else "pass"
            if index == 1:
                evidence = f"We cannot proceed unless the {categories[index-1]} issue is resolved."
                rationale = "The advisor states a brief explicit gating condition."
                mentions = 1
            elif index == 2:
                evidence = f"The {categories[index-1]} outcome is a major reason I am considering this move."
                rationale = "The advisor emphasizes a major positive decision factor."
                mentions = 2
            elif index == 3:
                evidence = f"The {categories[index-1]} concern matters, along with several other considerations."
                rationale = "The advisor identifies a meaningful but non-gating consideration."
                mentions = 2
            elif index == 4:
                evidence = f"I prefer the {categories[index-1]} option, although it is not required."
                rationale = "The advisor expresses a secondary comparative preference."
                mentions = 4
            else:
                evidence = f"We discussed the {categories[index-1]} details at length, but they do not affect my decision."
                rationale = "The lengthy discussion is peripheral to the decision."
                mentions = 6
            transcript_order = ((index * 3 + group_number) % 5) + 1
            business_score = round(3.5 + ((group_number * 2 + index * 3) % 15) / 10, 2)
            support_score = round(0.76 + ((group_number + index * 2) % 20) / 100, 2)
            candidates.append(ImportanceCandidate(
                candidate_id, group_id, item_type, categories[index-1], evidence, rationale,
                verdict, "explicit" if index != 4 else "tightly_implied", transcript_order,
                decision_direction="affects_timing" if index == 1 else "supports_move" if item_type == "driver" else "opposes_move",
                bounded_advisor_context=evidence, business_score=business_score,
                support_score=support_score, mention_count=mentions,
                representative_context_present=index == 5 and group_number % 4 == 0,
            ))
            experts.append(ExpertImportance(candidate_id, tier, max(0, score - 0.08), min(1, score + 0.08), rank, False, centrality, dependency, consequence, priority, attention, f"Expert rank {rank}: {tier} transcript-relative importance."))
        # Ineligible procedural remnant proves filtering without entering comparison.
        candidates.append(ImportanceCandidate(
            f"{group_id}-c6", group_id, "driver", "follow-up scheduling",
            "I will call next week.", "Procedural follow-up only.", "reject", "explicit", 6,
            business_score=4.9, support_score=0.2, mention_count=3,
        ))
        pairs = tuple((experts[a].candidate_id, experts[b].candidate_id, "a_more_important") for a, b in ((0, 1), (0, 3), (1, 2), (3, 4)))
        groups.append(ImportanceGroup(group_id, split, tuple(candidates), tuple(experts), pairs, group_number % 4 == 0, False, "hard" if group_number % 3 == 0 else "medium"))
    return groups


def validate_dataset(groups: list[ImportanceGroup]) -> dict[str, Any]:
    ids = [group.group_id for group in groups]
    candidate_ids = [candidate.candidate_id for group in groups for candidate in group.candidates]
    assert len(groups) == 40 and len(ids) == len(set(ids))
    assert len(candidate_ids) == len(set(candidate_ids))
    assert sum(group.split == "development" for group in groups) == 28
    assert sum(group.split == "holdout" for group in groups) == 12
    assert sum(len(group.candidates) for group in groups) >= 200
    assert all(len(group.expert) == 5 for group in groups)
    return {"groups": 40, "candidates": len(candidate_ids), "eligible": sum(len(group.expert) for group in groups), "development": 28, "holdout": 12}


def _ranks_from_order(order: list[str]) -> dict[str, int]:
    return {candidate_id: rank for rank, candidate_id in enumerate(order, 1)}


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values); position = 0
    while position < len(ordered):
        end = position
        while end + 1 < len(ordered) and values[ordered[end + 1]] == values[ordered[position]]: end += 1
        rank = (position + end + 2) / 2
        for index in ordered[position:end + 1]: ranks[index] = rank
        position = end + 1
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2: return 0.0
    lm, rm = mean(left), mean(right)
    numerator = sum((a-lm)*(b-rm) for a,b in zip(left,right,strict=True))
    denominator = math.sqrt(sum((a-lm)**2 for a in left) * sum((b-rm)**2 for b in right))
    return numerator / denominator if denominator else 0.0


def spearman(expert: list[int], predicted: list[int]) -> float:
    return _correlation(_rankdata([float(x) for x in expert]), _rankdata([float(x) for x in predicted]))


def kendall(expert: list[int], predicted: list[int]) -> float:
    concordant = discordant = 0
    for i in range(len(expert)):
        for j in range(i + 1, len(expert)):
            product = (expert[i] - expert[j]) * (predicted[i] - predicted[j])
            concordant += product > 0; discordant += product < 0
    return (concordant - discordant) / (concordant + discordant) if concordant + discordant else 0.0


def ndcg(expert: list[int], predicted_order: list[int]) -> float:
    relevance = {rank: len(expert) - rank + 1 for rank in expert}
    dcg = sum((2 ** relevance[expert[index]] - 1) / math.log2(position + 2) for position, index in enumerate(predicted_order))
    ideal = sum((2 ** score - 1) / math.log2(position + 2) for position, score in enumerate(sorted(relevance.values(), reverse=True)))
    return dcg / ideal if ideal else 0.0


def ranking_metrics(groups: list[ImportanceGroup], rankings: dict[str, list[str]]) -> dict[str, float]:
    values = Counter(); count = 0
    for group in groups:
        predicted = rankings.get(group.group_id)
        if not predicted: continue
        expert_by_id = {item.candidate_id: item.relative_rank for item in group.expert}
        predicted = [candidate_id for candidate_id in predicted if candidate_id in expert_by_id]
        if len(predicted) != len(expert_by_id): continue
        ids = list(expert_by_id); expert = [expert_by_id[candidate_id] for candidate_id in ids]
        predicted_ranks = _ranks_from_order(predicted); guessed = [predicted_ranks[candidate_id] for candidate_id in ids]
        values["spearman"] += spearman(expert, guessed); values["kendall"] += kendall(expert, guessed)
        predicted_indices = [ids.index(candidate_id) for candidate_id in predicted]
        values["ndcg"] += ndcg(expert, predicted_indices)
        top = min(expert_by_id, key=expert_by_id.get); position = predicted.index(top) + 1
        values["mrr"] += 1 / position; values["top1"] += position == 1; values["top3_recall"] += position <= 3
        count += 1
    return {key: value / count for key, value in values.items()} | {"group_count": count}


def baseline_rankings(groups: list[ImportanceGroup], seed: int = 17) -> dict[str, dict[str, list[str]]]:
    rng = random.Random(seed); output: dict[str, dict[str, list[str]]] = {name: {} for name in ("business_score", "transcript_order", "evidence_length", "mention_frequency", "validation_support", "structured_rule", "random")}
    for group in groups:
        eligible = [candidate for candidate in group.candidates if candidate.validation_verdict in {"pass", "needs_review"}]
        output["business_score"][group.group_id] = [item.candidate_id for item in sorted(eligible, key=lambda x: (-(x.business_score or 0), x.candidate_id))]
        output["transcript_order"][group.group_id] = [item.candidate_id for item in sorted(eligible, key=lambda x: x.transcript_order)]
        output["evidence_length"][group.group_id] = [item.candidate_id for item in sorted(eligible, key=lambda x: (-len(x.advisor_evidence), x.candidate_id))]
        output["mention_frequency"][group.group_id] = [item.candidate_id for item in sorted(eligible, key=lambda x: (-x.mention_count, x.candidate_id))]
        output["validation_support"][group.group_id] = [item.candidate_id for item in sorted(eligible, key=lambda x: (-(x.support_score or 0), x.candidate_id))]
        expert_items = {item.candidate_id: item for item in group.expert}
        output["structured_rule"][group.group_id] = [item.candidate_id for item in sorted(group.expert, key=lambda x: (-((x.score_min+x.score_max)/2), x.candidate_id))]
        random_items = [item.candidate_id for item in eligible]; rng.shuffle(random_items); output["random"][group.group_id] = random_items
    return output


def run_baselines(groups: list[ImportanceGroup]) -> dict[str, Any]:
    rankings = baseline_rankings(groups)
    return {name: ranking_metrics(groups, values) for name, values in rankings.items()}


def run_live(groups: list[ImportanceGroup], split: str, mode: str, output_path: Path, order: str = "original", group_limit: int | None = None) -> dict[str, Any]:
    if split not in {"development", "holdout"}: raise ValueError("invalid split")
    selected = [group for group in groups if group.split == split]
    if group_limit is not None: selected = selected[:group_limit]
    rows = []
    for group in selected:
        candidates = list(group.candidates)
        if order == "reverse": candidates.reverse()
        elif order == "shuffle": random.Random(group.group_id).shuffle(candidates)
        result = run_importance(candidates, mode)
        rows.append({"group_id": group.group_id, "order": order, "result": {"mode": result.mode, "transcript_id": result.transcript_id, "results": [asdict(item) for item in result.results], "pairwise": [item.model_dump() for item in result.pairwise], "pairwise_cycles": result.pairwise_cycles, "warnings": result.warnings, "used_fallback": result.used_fallback, "malformed_output": result.malformed_output}})
    document = {"dataset": validate_dataset(groups), "split": split, "mode": mode, "order": order, "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def analyze_live(groups: list[ImportanceGroup], document: dict[str, Any]) -> dict[str, Any]:
    selected = {group.group_id: group for group in groups if group.split == document["split"]}
    rankings = {}; omitted = invented = malformed = cycles = disagreement = 0; tier_correct = tier_total = 0; score_errors=[]; top_errors=[]; pair_correct=pair_total=pair_tie_correct=pair_tie_total=0; tier_pairs=[]; direct_differences=[]
    for row in document["rows"]:
        group = selected[row["group_id"]]; result = row["result"]
        malformed += result["malformed_output"]; cycles += len(result["pairwise_cycles"]); disagreement += "listwise_pairwise_disagreement" in result["warnings"]
        expected_ids = {item.candidate_id for item in group.expert}; predicted_ids = {item["candidate_id"] for item in result["results"]}
        omitted += len(expected_ids-predicted_ids); invented += len(predicted_ids-expected_ids)
        rankings[group.group_id] = [item["candidate_id"] for item in sorted(result["results"], key=lambda x:x["relative_rank"])]
        expert = {item.candidate_id:item for item in group.expert}
        for item in result["results"]:
            if item["candidate_id"] not in expert: continue
            target=expert[item["candidate_id"]]; tier_correct += item["importance_tier"] == target.importance_tier; tier_total += 1; tier_pairs.append((target.importance_tier,item["importance_tier"])); direct_differences.append(item["score_difference"])
            midpoint=(target.score_min+target.score_max)/2; score_errors.append(item["structured_score"]-midpoint)
        for comparison in result["pairwise"]:
            a=expert.get(comparison["candidate_a_id"]); b=expert.get(comparison["candidate_b_id"])
            if not a or not b: continue
            expected="approximately_equal" if a.relative_rank==b.relative_rank else "a_more_important" if a.relative_rank<b.relative_rank else "b_more_important"
            pair_total+=1; pair_correct+=comparison["relative_importance"]==expected
            if expected=="approximately_equal": pair_tie_total+=1; pair_tie_correct+=comparison["relative_importance"]==expected
        if rankings[group.group_id]:
            expected_top=min(group.expert,key=lambda x:x.relative_rank).candidate_id; predicted_top=rankings[group.group_id][0]
            if expected_top != predicted_top: top_errors.append({"group_id":group.group_id,"expected_top":expected_top,"predicted_top":predicted_top,"cause":"model_relative_weight_error"})
    ranking=ranking_metrics(list(selected.values()),rankings)
    tier_labels=sorted({value for pair in tier_pairs for value in pair}); per_tier={}
    for label in tier_labels:
        tp=sum(a==p==label for a,p in tier_pairs); fp=sum(a!=label and p==label for a,p in tier_pairs); fn=sum(a==label and p!=label for a,p in tier_pairs); precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; per_tier[label]={"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0}
    return {"ranking":ranking,"tier_accuracy":tier_correct/tier_total if tier_total else 0,"tier_macro_precision":mean(x["precision"] for x in per_tier.values()) if per_tier else 0,"tier_macro_recall":mean(x["recall"] for x in per_tier.values()) if per_tier else 0,"tier_macro_f1":mean(x["f1"] for x in per_tier.values()) if per_tier else 0,"per_tier":per_tier,"score_mae":mean(abs(x) for x in score_errors) if score_errors else 0,"score_rmse":math.sqrt(mean(x*x for x in score_errors)) if score_errors else 0,"direct_structured_mean_difference":mean(direct_differences) if direct_differences else 0,"pairwise_accuracy":pair_correct/pair_total if pair_total else 0,"pairwise_count":pair_total,"tie_accuracy":pair_tie_correct/pair_tie_total if pair_tie_total else None,"omitted":omitted,"invented":invented,"malformed":malformed,"cycles":cycles,"listwise_pairwise_disagreement_groups":disagreement,"top_errors":top_errors}


def write_report(result: dict[str, Any], path: Path) -> None:
    lines=["# Signal Importance Evaluation","",f"Mode: {result.get('mode','analysis')}",""]
    if "baselines" in result:
        lines += ["## Baselines","","| Baseline | Spearman | Kendall | NDCG | Top-1 |","|---|---:|---:|---:|---:|"]
        for name,item in result["baselines"].items(): lines.append(f"| {name} | {item.get('spearman',0):.3f} | {item.get('kendall',0):.3f} | {item.get('ndcg',0):.3f} | {item.get('top1',0):.3f} |")
    if "analysis" in result:
        item=result["analysis"]; lines += ["## Model","",f"- Spearman: {item['ranking'].get('spearman',0):.3f}",f"- Kendall: {item['ranking'].get('kendall',0):.3f}",f"- NDCG: {item['ranking'].get('ndcg',0):.3f}",f"- Top-1: {item['ranking'].get('top1',0):.3f}",f"- Tier accuracy: {item['tier_accuracy']:.3f}"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("validate","baselines","development-listwise","development-pairwise","development-hybrid","holdout-listwise","holdout-pairwise","holdout-hybrid","order-sensitivity","report")); parser.add_argument("--output-dir",type=Path,default=Path("data/outputs/importance")); parser.add_argument("--input",type=Path); args=parser.parse_args(); groups=build_importance_dataset(); validate_dataset(groups); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=="validate": result=validate_dataset(groups)
    elif args.mode=="baselines": result={"mode":"baselines","baselines":run_baselines(groups)}; write_report(result,args.output_dir/"baselines.md")
    elif args.mode=="report":
        if not args.input: raise SystemExit("--input required")
        document=json.loads(args.input.read_text(encoding="utf-8")); result={"mode":document["mode"],"analysis":analyze_live(groups,document)}; write_report(result,args.output_dir/"analysis.md"); (args.output_dir/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    elif args.mode=="order-sensitivity":
        docs=[run_live(groups,"development","importance_shadow_listwise",args.output_dir/f"order-{order}.json",order,8) for order in ("original","reverse","shuffle")]; result={"mode":"order-sensitivity","group_count":8,"documents":[str(args.output_dir/f"order-{order}.json") for order in ("original","reverse","shuffle")]}
    else:
        split="development" if args.mode.startswith("development") else "holdout"; method=args.mode.split("-",1)[1]; mode=f"importance_shadow_{method}"; result=run_live(groups,split,mode,args.output_dir/f"{args.mode}.json")
    print(json.dumps(result,indent=2,default=lambda value:dict(value)))


if __name__=="__main__": main()
