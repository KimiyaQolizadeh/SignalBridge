"""Authorized synthetic labels and calibration helpers for dedup experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledPair:
    pair_id: str
    item_type_a: str
    category_a: str
    evidence_a: str
    rationale_a: str
    item_type_b: str
    category_b: str
    evidence_b: str
    rationale_b: str
    duplicate: bool
    label_rationale: str
    difficulty: str


def _pair(
    pair_id: str,
    left: tuple[str, str],
    right: tuple[str, str],
    duplicate: bool,
    difficulty: str,
    label_rationale: str,
    *,
    item_types: tuple[str, str] = ("blocker", "blocker"),
    rationales: tuple[str, str] = ("Grounded rationale A.", "Grounded rationale B."),
) -> LabeledPair:
    return LabeledPair(
        pair_id, item_types[0], left[0], left[1], rationales[0],
        item_types[1], right[0], right[1], rationales[1], duplicate,
        label_rationale, difficulty,
    )


DUPLICATE_PAIRS = (
    _pair("d01", ("Partner Approval", "My partner must approve before we proceed."), ("Decision Maker Dependency", "We cannot proceed until my partner agrees."), True, "lexical mismatch", "Same approval source and response."),
    _pair("d02", ("Third-Party Product Restriction", "I refuse to use third-party products."), ("Product Policy Requirement", "I will only use proprietary products."), True, "lexical mismatch", "Same product-policy restriction."),
    _pair("d03", ("Client Time", "I want more time with clients."), ("Administrative Relief", "I need less administration so I can focus on clients."), True, "broad/narrow restatement", "Same desired client-time outcome.", item_types=("driver", "driver")),
    _pair("d04", ("Salesforce Dependency", "We cannot move unless Salesforce integrates."), ("CRM Integration", "Salesforce integration is required before we proceed."), True, "easy", "Same named technology dependency."),
    _pair("d05", ("Book Portability", "I cannot leave my client book behind."), ("Client Transfer", "The book must transfer for me to move."), True, "lexical mismatch", "Same portability condition."),
    _pair("d06", ("Payout Requirement", "The payout has to work before I move."), ("Economics Condition", "I would only proceed if the economics work."), True, "broad/narrow restatement", "Same economic condition."),
    _pair("d07", ("Decision Commitment", "We're moving forward."), ("Intent To Proceed", "I am ready to make the move."), True, "lexical mismatch", "Same explicit commitment.", item_types=("driver", "driver")),
    _pair("d08", ("Compliance Access", "All compliance records must remain accessible."), ("Document Retention", "I cannot move unless I retain access to every compliance file."), True, "broad/narrow restatement", "Same compliance-record condition."),
    _pair("d09", ("Succession Timing", "I do not want to work another eight years."), ("Exit Horizon", "My transition needs to happen before an eight-year horizon."), True, "lexical mismatch", "Same timing constraint."),
    _pair("d10", ("Fee Responsibility", "I need to know whether I pay the fee."), ("Cost Ownership", "Who bears this fee is material to my decision."), True, "different category", "Same unresolved fee owner."),
    _pair("d11", ("Workload Reduction", "I need a firm that reduces my workload."), ("Operational Support", "I am looking for more support so I stop working late."), True, "lexical mismatch", "Same support-driven workload outcome.", item_types=("driver", "driver")),
    _pair("d12", ("Dealer Participation", "My dealer has to participate before I proceed."), ("Dealer Dependency", "I cannot move until the dealer is on board."), True, "easy", "Same dealer dependency."),
    _pair("d13", ("Office Cost", "I need clarity on whether I pay office costs."), ("Expense Responsibility", "Office expense responsibility must be resolved."), True, "different category", "Same office-cost question."),
    _pair("d14", ("Values Alignment", "I want a firm aligned with my values."), ("Cultural Fit", "Shared values are why I am considering this firm."), True, "lexical mismatch", "Same values-based motivation.", item_types=("driver", "driver")),
    _pair("d15", ("Client Offering", "I am excited to bring clients an institutional offering."), ("Institutional Access", "Institutional products would let me deliver what clients need."), True, "broad/narrow restatement", "Same client-offering value.", item_types=("driver", "driver")),
    _pair("d16", ("Technology Value", "This platform is far ahead of ours."), ("Platform Improvement", "The technology is a major improvement over my current system."), True, "lexical mismatch", "Same platform-value motivation.", item_types=("driver", "driver")),
    _pair("d17", ("No Delegation", "We do not delegate portfolio decisions."), ("Investment Control", "I must retain every investment decision."), True, "different category", "Same control requirement."),
    _pair("d18", ("Compliance Review", "Compliance must approve before I proceed."), ("Regulatory Approval", "I cannot move without compliance approval."), True, "easy", "Same compliance approval dependency."),
    _pair("d19", ("Risk Profile", "Risk profiling must remain part of my process."), ("Suitability Process", "I require risk assessment before portfolio selection."), True, "broad/narrow restatement", "Same suitability requirement."),
    _pair("d20", ("Compensation Clarity", "I need the payout calculation resolved."), ("Payout Uncertainty", "I cannot decide until I understand how payout is calculated."), True, "lexical mismatch", "Same payout-calculation uncertainty."),
    _pair("d21", ("Same Category", "My partner must approve."), ("Same Category", "Partner agreement is required."), True, "same wording", "Same factor despite generic category."),
    _pair("d22", ("Rationale Noise", "We cannot proceed until my partner approves."), ("Rationale Noise", "We cannot proceed until my partner approves."), True, "generated-rationale noise", "Identical evidence; generated rationales differ.", rationales=("Approval delays timing.", "A speculative unrelated generated explanation.")),
    _pair("d23", ("Client No-Show", "A referred provider failed to meet my client."), ("Referral Reliability", "I need confidence referred specialists will show up."), True, "broad/narrow restatement", "Same referral reliability concern."),
    _pair("d24", ("Product Access", "I need access to corporate-class funds."), ("Product Gap", "The absence of corporate-class funds is a problem for my market."), True, "lexical mismatch", "Same product availability factor."),
    _pair("d25", ("Transition Commitment", "This is the direction I want to take."), ("Decision Commitment", "I want to proceed with the transition."), True, "easy", "Same explicit intent.", item_types=("driver", "driver")),
)


DISTINCT_PAIRS = (
    _pair("n01", ("Third-Party Products", "We will not use third-party products."), ("Delegation Authority", "We do not delegate portfolio decisions."), False, "same topic but distinct", "Product policy and decision authority require different responses."),
    _pair("n02", ("Technology Integration", "Salesforce must integrate."), ("Compliance Access", "Compliance files must remain accessible."), False, "same topic but distinct", "Different technology requirements."),
    _pair("n03", ("Partner Approval", "My partner must approve."), ("Book Portability", "My client book must transfer."), False, "easy", "Different dependencies."),
    _pair("n04", ("Client Time", "I want more client time."), ("Compensation", "I want higher compensation."), False, "easy", "Different motivations.", item_types=("driver", "driver")),
    _pair("n05", ("Commitment", "I would move if the payout works."), ("Payout Condition", "The payout must work before I move."), False, "conditional", "Positive and negative sides require different treatment.", item_types=("driver", "blocker")),
    _pair("n06", ("Partner Discussion", "I will discuss it with my partner."), ("Partner Approval", "My partner must approve before I proceed."), False, "same topic but distinct", "Procedure is not an approval dependency."),
    _pair("n07", ("Digital Preference", "I prefer digital documents."), ("Digital Requirement", "I cannot proceed without digital documents."), False, "same topic but distinct", "Preference and mandatory condition differ."),
    _pair("n08", ("Technology Question", "Do you integrate with Salesforce?"), ("Technology Dependency", "We cannot move without Salesforce integration."), False, "same topic but distinct", "Information request and material dependency differ."),
    _pair("n09", ("Current Pain", "My compliance system is terrible."), ("Future Condition", "Your system must preserve compliance records."), False, "same topic but distinct", "Current dissatisfaction is not a future requirement."),
    _pair("n10", ("Commitment", "We're moving forward."), ("Procedural Review", "I will review the materials."), False, "easy", "Commitment and diligence differ.", item_types=("driver", "driver")),
    _pair("n11", ("Operating Model", "I refuse third-party products."), ("Operating Model", "I must own the client relationship."), False, "same category", "Same category but distinct control domains."),
    _pair("n12", ("Approval", "Compliance must approve."), ("Approval", "My partner must approve."), False, "same category", "Different approval sources require different actions."),
    _pair("n13", ("Cost", "Office expenses must be resolved."), ("Cost", "The payout percentage must be resolved."), False, "same category", "Different economic consequences."),
    _pair("n14", ("Timing", "I need to move before year-end."), ("Timing", "My partner must approve before proceeding."), False, "same category", "Deadline and approval dependency differ."),
    _pair("n15", ("Client Service", "I want more client-facing time."), ("Client Service", "Referred specialists must reliably meet clients."), False, "same category", "Desired time and referral risk differ."),
    _pair("n16", ("Product Policy", "I only use proprietary products."), ("Product Access", "I need corporate-class funds."), False, "same topic but distinct", "Policy restriction and product availability differ."),
    _pair("n17", ("Control", "I retain investment discretion."), ("Control", "I own the client relationship."), False, "same category", "Investment and relationship authority differ."),
    _pair("n18", ("Compliance", "Compliance approval is required."), ("Compliance", "I need complete document access."), False, "same topic but distinct", "Approval and record access require different responses."),
    _pair("n19", ("Economics", "The payout percentage is too low."), ("Economics", "I need clarity on office expenses."), False, "same topic but distinct", "Revenue and expense concerns differ."),
    _pair("n20", ("Technology", "Salesforce integration is required."), ("Technology", "The planning platform must support corporate accounts."), False, "same topic but distinct", "Different systems and requirements."),
    _pair("n21", ("Intent", "I am ready to move."), ("Interest", "The platform sounds interesting."), False, "similar evidence different intent", "Commitment and generic interest differ.", item_types=("driver", "driver")),
    _pair("n22", ("Current Workload", "I am exhausted from working late."), ("Support Motivation", "I am seeking support so I can stop working late."), False, "similar evidence different intent", "Pain alone and recruiting motivation differ.", item_types=("driver", "driver")),
    _pair("n23", ("Dealer", "My dealer must participate."), ("Dealer", "My dealer owns the client records."), False, "same category", "Participation dependency and ownership constraint differ."),
    _pair("n24", ("Succession", "I want to retire within five years."), ("Succession", "The succession payout calculation is unclear."), False, "same category", "Timing and economics differ."),
    _pair("n25", ("Ambiguous", "That could work for me."), ("Commitment", "I will proceed."), False, "ambiguous", "Ambiguous acceptance is not explicit commitment.", item_types=("driver", "driver")),
)

LABELED_PAIRS = DUPLICATE_PAIRS + DISTINCT_PAIRS


def classification_metrics(labels: list[bool], scores: list[float], threshold: float) -> dict[str, float | int]:
    predictions = [score >= threshold for score in scores]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels, strict=True))
    fp = sum(prediction and not label for prediction, label in zip(predictions, labels, strict=True))
    fn = sum(not prediction and label for prediction, label in zip(predictions, labels, strict=True))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = 0.25
    f05 = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "f0.5": f05, "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn}


def calibrate_merge_threshold(labels: list[bool], scores: list[float], minimum_precision: float = 0.95) -> dict[str, float | int]:
    thresholds = sorted({0.0, 1.0, *scores})
    results = [classification_metrics(labels, scores, threshold) for threshold in thresholds]
    eligible = [result for result in results if result["precision"] >= minimum_precision]
    pool = eligible or results
    return max(pool, key=lambda result: (result["recall"], result["f0.5"], result["threshold"]))
