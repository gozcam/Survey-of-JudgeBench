"""Generate figures for the JudgeBench survey report.

Loads all full-run JSONL outputs from outputs/, computes derived metrics,
and saves one PDF + one PNG per figure to figures/ at the repo root.

Figures produced:
  figure1_failure_modes          - stacked bar: correct / incorrect / inconsistent per model
  figure2_category_accuracy      - grouped bar: accuracy by category and model
  figure3_inconsistency_heatmap  - heatmap: inconsistency rate by model and category
  figure4_agreement_matrix       - heatmap: pairwise agreement between prompted judges
  figure5_consensus              - horizontal stacked bar: consensus distribution across prompted models
  figure6_score_margin           - grouped bar: reward model score margin by category
  figure7_incorrect_heatmap      - heatmap: incorrect judgment rate by model and category

When rerun all figures are regenerated
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "figures"

# all five buckets used throughout metric computation and plotting
CATEGORIES = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", "Overall"]

# the four categories that have distinct task-domain semantics; Overall is excluded
# from category-level plots so it doesn't distort the per-domain view
PLOT_CATEGORIES = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench"]

# livebench sources match exactly; mmlu-pro has subcategories like "mmlu-pro-math"
# so it needs prefix matching in top_level_category rather than a direct dict lookup
CATEGORY_MAP = {
    "livebench-math": "livebench-math",
    "livebench-reasoning": "livebench-reasoning",
    "livecodebench": "livecodebench",
}

# human-readable axis labels for each category; include the source name in parens
# so readers can cross-reference with the raw output files
CATEGORY_LABELS = {
    "mmlu-pro": "Knowledge\n(mmlu-pro)",
    "livebench-reasoning": "Reasoning\n(livebench)",
    "livebench-math": "Math\n(livebench)",
    "livecodebench": "Coding\n(livecodebench)",
    "Overall": "Overall",
}

# fixed display order and colors across all multi-model figures, sorted by overall accuracy;
# keeping this centralized ensures every figure uses the same legend ordering and palette
MODEL_ORDER = [
    ("gpt-4.1-mini", "GPT-4.1-mini",   "#1f77b4"),
    ("gemini",       "Gemini 2.5 FL",  "#2ca02c"),
    ("reward",       "Skywork-Reward", "#9467bd"),
    ("critic",       "Skywork-Critic", "#ff7f0e"),
    ("4o-mini",      "GPT-4o-mini",    "#d62728"),
    ("llama",        "Llama-3.1-8B",   "#7f7f7f"),
]

# short display names for the four prompted models, matched by the same fragment logic
# as MODEL_ORDER; used in agreement matrix and consensus figures
PROMPTED_LABELS = [
    ("gpt-4.1-mini", "GPT-4.1-mini"),
    ("gemini",       "Gemini 2.5 FL"),
    ("4o-mini",      "GPT-4o-mini"),
    ("llama",        "Llama-3.1-8B"),
]


# ---------------------------------------------------------------------------
# Shared data loading — mirrors analyze_outputs.py
# ---------------------------------------------------------------------------

def top_level_category(source: str) -> str:
    if source in CATEGORY_MAP:
        return CATEGORY_MAP[source]
    # mmlu-pro subcategories (e.g. "mmlu-pro-biology") all roll up to "mmlu-pro"
    if source.startswith("mmlu-pro"):
        return "mmlu-pro"
    return "other"


def flip(decision: str) -> str:
    # in the second pass A and B are swapped, so un-flip the decision
    # before comparing it against the original label
    if decision == "A>B":
        return "B>A"
    if decision == "B>A":
        return "A>B"
    return decision


def score_pair(pair: dict) -> str:
    """Return 'correct', 'incorrect', or 'inconsistent' for a judged pair."""
    judgments = pair["judgments"]
    j1 = judgments[0] if judgments else None
    j2 = judgments[1] if len(judgments) > 1 else None

    if j1 is None:
        return "incorrect"
    # single-pass judge (reward model): one decision, no ordering swap to check
    if j2 is None:
        d = j1["decision"]
        return "correct" if d == pair["label"] else "incorrect"

    # two-pass scoring: tally votes after un-flipping the swapped pass
    decision1 = j1["decision"]
    decision2 = flip(j2["decision"])
    label = pair["label"]

    # +1 when a pass agrees with the label, -1 when it disagrees;
    # counter == 0 means the judge changed its mind when the order changed
    counter = 0
    for d in [decision1, decision2]:
        if d == label:
            counter += 1
        elif d == flip(label):
            counter -= 1

    if counter > 0:
        return "correct"
    if counter < 0:
        return "incorrect"
    return "inconsistent"


def load_output_file(path: Path) -> list[dict]:
    """Load a JSONL output file and return its records as a list of dicts."""
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(pairs: list[dict]) -> dict:
    """Return per-category accuracy and outcome counts for a list of judged pairs."""
    stats = {cat: {"correct": 0, "incorrect": 0, "inconsistent": 0} for cat in CATEGORIES}
    for pair in pairs:
        cat = top_level_category(pair["source"])
        outcome = score_pair(pair)
        if cat in stats:
            stats[cat][outcome] += 1
        stats["Overall"][outcome] += 1

    results = {}
    for cat, counts in stats.items():
        total = counts["correct"] + counts["incorrect"] + counts["inconsistent"]
        # guard against empty categories — small pilots may skip some domains entirely
        if total == 0:
            results[cat] = {"accuracy": None, **counts, "total": 0}
        else:
            results[cat] = {"accuracy": 100.0 * counts["correct"] / total, **counts, "total": total}
    return results


def model_short_name(path: Path) -> str:
    # output filenames follow "dataset=...,judge_model=<name>"; extract just the model name
    stem = path.stem
    if "judge_model=" in stem:
        return stem.split("judge_model=")[1]
    return stem


def load_all_models() -> tuple[dict, dict]:
    """Return (model_metrics, model_pairs) for all full-run output files."""
    output_files = sorted(
        p for p in OUTPUTS_DIR.glob("dataset=judgebench,*.jsonl")
        if "pilot" not in p.name
    )
    if not output_files:
        print("No full-run output files found in outputs/.", file=sys.stderr)
        sys.exit(1)

    model_metrics: dict[str, dict] = {}
    model_pairs: dict[str, list] = {}
    for path in output_files:
        key = model_short_name(path)
        pairs = load_output_file(path)
        model_metrics[key] = compute_metrics(pairs)
        model_pairs[key] = pairs
    return model_metrics, model_pairs


# ---------------------------------------------------------------------------
# Shared helpers for cross-model and reward-model analysis
# ---------------------------------------------------------------------------

def _final_verdict(pair: dict) -> str:
    """Collapse a (possibly two-pass) pair into a single verdict string."""
    judgments = pair["judgments"]
    j1 = judgments[0] if judgments else None
    j2 = judgments[1] if len(judgments) > 1 else None
    if j1 is None:
        return "unknown"
    if j2 is None:
        return j1["decision"]
    d1 = j1["decision"]
    d2 = flip(j2["decision"])
    # if the two passes disagree after un-flipping, treat it as a tie
    return d1 if d1 == d2 else "tie"


def _pairwise_agreement(k1: str, k2: str, model_pairs: dict) -> float:
    """Return the % of shared pairs where k1 and k2 give the same final verdict."""
    by_id1 = {p["pair_id"]: p for p in model_pairs[k1]}
    by_id2 = {p["pair_id"]: p for p in model_pairs[k2]}
    shared = set(by_id1) & set(by_id2)
    if not shared:
        return 0.0
    agree = sum(1 for pid in shared if _final_verdict(by_id1[pid]) == _final_verdict(by_id2[pid]))
    return 100.0 * agree / len(shared)


def _prompted_keys(model_pairs: dict) -> list[str]:
    """Return model keys whose judge_name is arena_hard (prompted paradigm)."""
    # judge_name is written by the runner and is the only reliable way to
    # distinguish prompted from fine-tuned judges for models of the same family
    return [k for k, pairs in model_pairs.items()
            if pairs and pairs[0].get("judge_name") == "arena_hard"]


def _prompted_display_name(key: str) -> str:
    """Return the short display name for a prompted model key."""
    k = key.lower()
    for fragment, label in PROMPTED_LABELS:
        if fragment in k:
            return label
    return key[:14]


def _reward_scores(pair: dict) -> tuple[float, float] | None:
    """Extract (score_A, score_B) from a reward-model pair, or None if unavailable."""
    try:
        # the reward runner stores raw scalar scores in judgments[0]["judgment"]["scores"]
        # rather than a text decision like chat-style judges do
        scores = pair["judgments"][0]["judgment"]["scores"]
        if scores and len(scores) == 2:
            return float(scores[0]), float(scores[1])
    except (IndexError, KeyError, TypeError):
        pass
    return None


def _reward_key(model_pairs: dict) -> str | None:
    """Return the model key for the Skywork Reward model, or None if not loaded."""
    for key, pairs in model_pairs.items():
        if pairs and pairs[0].get("judge_name") == "reward_model" and "Skywork" in key:
            return key
    return None


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    """Save a figure as both PDF (for LaTeX) and PNG (for preview)."""
    for ext in ("pdf", "png"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"    -> {out.name}")
    plt.close(fig)


def _ordered_models(model_metrics: dict) -> list[tuple[str, str, str]]:
    """Return (key, label, color) tuples in fixed display order for loaded models."""
    result = []
    for fragment, label, color in MODEL_ORDER:
        for key in model_metrics:
            if fragment in key.lower() and (key, label, color) not in result:
                result.append((key, label, color))
                break
    return result


# ---------------------------------------------------------------------------
# Figure 1 — overall outcome breakdown (stacked bar, sorted by accuracy)
# ---------------------------------------------------------------------------

def make_figure_failure_modes(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Stacked bar: proportion of Correct / Incorrect / Inconsistent per model."""
    ordered = _ordered_models(model_metrics)
    labels = [label for _, label, _ in ordered]

    correct_pct, incorrect_pct, inconsistent_pct = [], [], []
    for key, _, _ in ordered:
        ov = model_metrics[key]["Overall"]
        t = ov["total"]
        correct_pct.append(100.0 * ov["correct"] / t)
        incorrect_pct.append(100.0 * ov["incorrect"] / t)
        inconsistent_pct.append(100.0 * ov["inconsistent"] / t)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x, correct_pct, color="#4CAF50", label="Correct")
    ax.bar(x, incorrect_pct, bottom=correct_pct, color="#F44336", label="Incorrect")
    bottoms_ins = [c + i for c, i in zip(correct_pct, incorrect_pct)]
    ax.bar(x, inconsistent_pct, bottom=bottoms_ins, color="#FF9800", label="Inconsistent")

    # annotate each segment with its percentage; skip segments too narrow to read
    for i, (c, inc, ins) in enumerate(zip(correct_pct, incorrect_pct, inconsistent_pct)):
        for val, base, color in (
            (c,   0,       "white"),
            (inc, c,       "white"),
            (ins, c + inc, "white"),
        ):
            if val >= 6:
                ax.text(i, base + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Percentage of pairs (%)", fontsize=11)
    ax.set_title("Outcome breakdown per judge (350 pairs, sorted by accuracy)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save(fig, figures_dir, "figure1_failure_modes")


# ---------------------------------------------------------------------------
# Figure 2 — category-level accuracy grouped bar chart (section 4.2)
# ---------------------------------------------------------------------------

def make_figure_category_accuracy(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Grouped bar chart: accuracy per model for each of the four task categories."""
    ordered = _ordered_models(model_metrics)
    n_models = len(ordered)
    n_cats = len(PLOT_CATEGORIES)

    bar_w = 0.13
    group_gap = 0.05
    group_w = n_models * bar_w + group_gap
    x_centers = np.arange(n_cats) * group_w

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (key, label, color) in enumerate(ordered):
        offsets = x_centers + (i - n_models / 2 + 0.5) * bar_w
        accs = [model_metrics[key][cat]["accuracy"] or 0 for cat in PLOT_CATEGORIES]
        bars = ax.bar(offsets, accs, width=bar_w, color=color, label=label,
                      edgecolor="white", linewidth=0.5)

        # label bars that are tall enough to accommodate text without overlap
        for bar, val in zip(bars, accs):
            if val >= 12:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333333",
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in PLOT_CATEGORIES],
        fontsize=11,
    )
    ax.set_ylabel("Pairwise accuracy (%)", fontsize=11)
    ax.set_title("Category-level accuracy by judge (350 pairs)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    fig.tight_layout()
    _save(fig, figures_dir, "figure2_category_accuracy")


# ---------------------------------------------------------------------------
# Figure 3 — inconsistency rate heatmap by model and category (section 4.4)
# ---------------------------------------------------------------------------

def make_figure_inconsistency_heatmap(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Heatmap of order-swap inconsistency rate (%) by judge and task category."""
    ordered = _ordered_models(model_metrics)
    heatmap_cats = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", "Overall"]
    cat_labels = [
        "Knowledge (mmlu-pro)",
        "Reasoning (livebench)",
        "Math (livebench)",
        "Coding (livecodebench)",
        "Overall",
    ]
    model_labels = [label for _, label, _ in ordered]

    data = np.array([
        [
            100.0 * model_metrics[key][cat]["inconsistent"] / model_metrics[key][cat]["total"]
            if model_metrics[key][cat]["total"] > 0 else 0.0
            for key, _, _ in ordered
        ]
        for cat in heatmap_cats
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=42, aspect="auto")

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_yticks(range(len(cat_labels)))
    ax.set_yticklabels(cat_labels, fontsize=10)

    # separator line above the Overall row to visually distinguish it from per-category rows
    ax.axhline(len(heatmap_cats) - 1.5, color="white", linewidth=2)

    for i in range(len(heatmap_cats)):
        for j in range(len(ordered)):
            val = data[i, j]
            # switch to white text on dark cells so all annotations remain readable
            text_color = "white" if val > 26 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Inconsistency rate (%)", fontsize=9)

    ax.set_title("Order-swap inconsistency rate by judge and category", fontsize=12)
    fig.tight_layout()
    _save(fig, figures_dir, "figure3_inconsistency_heatmap")


# ---------------------------------------------------------------------------
# Figure 4 — prompted model pairwise agreement matrix (appendix)
# ---------------------------------------------------------------------------

def make_figure_agreement_matrix(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Heatmap of pairwise verdict agreement (%) between the four prompted judges."""
    prompted = _prompted_keys(model_pairs)
    # fall back gracefully if fewer than two prompted models were loaded
    if len(prompted) < 2:
        print("    [skipped: fewer than 2 prompted models found]")
        return

    # order rows/columns to match MODEL_ORDER so the layout is consistent with other figures
    order_map = {fragment: i for i, (fragment, _, _) in enumerate(MODEL_ORDER)}
    prompted = sorted(prompted, key=lambda k: order_map.get(
        next((f for f, _, _ in MODEL_ORDER if f in k.lower()), ""), 99
    ))
    labels = [_prompted_display_name(k) for k in prompted]
    n = len(prompted)

    # build the n x n agreement matrix; diagonal is always 100%
    data = np.zeros((n, n))
    for i, k1 in enumerate(prompted):
        for j, k2 in enumerate(prompted):
            data[i, j] = 100.0 if k1 == k2 else _pairwise_agreement(k1, k2, model_pairs)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="Blues", vmin=30, vmax=100, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            # use white text on the darker cells (high agreement) for readability
            text_color = "white" if val > 75 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement rate (%)", fontsize=9)

    ax.set_title("Pairwise verdict agreement between prompted judges", fontsize=12)
    fig.tight_layout()
    _save(fig, figures_dir, "figure4_agreement_matrix")


# ---------------------------------------------------------------------------
# Figure 5 — prompted consensus distribution (appendix)
# ---------------------------------------------------------------------------

def make_figure_consensus(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Horizontal stacked bar: how often all / most / few prompted judges are correct."""
    prompted = _prompted_keys(model_pairs)
    if len(prompted) < 2:
        print("    [skipped: fewer than 2 prompted models found]")
        return

    # find pairs that every prompted model actually judged so the consensus is fair
    by_id = {k: {p["pair_id"]: p for p in model_pairs[k]} for k in prompted}
    shared_ids = set.intersection(*(set(d.keys()) for d in by_id.values()))
    total = len(shared_ids)
    nm = len(prompted)

    # count correct votes per pair and bucket into five difficulty tiers
    buckets = {"All correct": 0, "Majority correct": 0, "Even split": 0,
               "Majority wrong": 0, "All wrong": 0}
    for pid in shared_ids:
        correct_count = sum(1 for k in prompted if score_pair(by_id[k][pid]) == "correct")
        if correct_count == nm:
            buckets["All correct"] += 1
        elif correct_count > nm / 2:
            buckets["Majority correct"] += 1
        elif correct_count == nm / 2:
            buckets["Even split"] += 1
        elif correct_count > 0:
            buckets["Majority wrong"] += 1
        else:
            buckets["All wrong"] += 1

    labels = list(buckets.keys())
    counts = list(buckets.values())
    pcts = [100.0 * c / total for c in counts]

    # green-to-red palette reflects the correctness gradient across the five buckets
    colors = ["#4CAF50", "#8BC34A", "#9E9E9E", "#FF7043", "#F44336"]

    fig, ax = plt.subplots(figsize=(9, 3))
    left = 0.0
    for label, pct, count, color in zip(labels, pcts, counts, colors):
        ax.barh(0, pct, left=left, color=color, edgecolor="white", linewidth=0.8)
        if pct >= 5:
            ax.text(left + pct / 2, 0, f"{label}\n{count} ({pct:.1f}%)",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of 350 pairs (%)", fontsize=11)
    ax.set_title(
        f"Consensus distribution across {nm} prompted judges (N={total} pairs)", fontsize=12
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, figures_dir, "figure5_consensus")


# ---------------------------------------------------------------------------
# Figure 6 — reward model score margin by category (appendix)
# ---------------------------------------------------------------------------

def make_figure_score_margin(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Grouped bar: mean reward model score margin for correct vs. incorrect pairs, by category."""
    rk = _reward_key(model_pairs)
    if rk is None:
        print("    [skipped: reward model output not found]")
        return

    # accumulate margins separately for correct and incorrect pairs per category;
    # margin = |score_A - score_B|, larger means the model was more confident
    margins: dict[str, dict[str, list[float]]] = {
        cat: {"correct": [], "incorrect": []} for cat in PLOT_CATEGORIES
    }
    for pair in model_pairs[rk]:
        scores = _reward_scores(pair)
        if scores is None:
            continue
        margin = abs(scores[0] - scores[1])
        outcome = score_pair(pair)
        cat = top_level_category(pair["source"])
        if cat in margins:
            bucket = "correct" if outcome == "correct" else "incorrect"
            margins[cat][bucket].append(margin)

    cats = PLOT_CATEGORIES
    correct_means = [
        float(np.mean(margins[c]["correct"])) if margins[c]["correct"] else 0.0
        for c in cats
    ]
    incorrect_means = [
        float(np.mean(margins[c]["incorrect"])) if margins[c]["incorrect"] else 0.0
        for c in cats
    ]

    x = np.arange(len(cats))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_c = ax.bar(x - bar_w / 2, correct_means, width=bar_w,
                    color="#4CAF50", label="Correct pairs")
    bars_i = ax.bar(x + bar_w / 2, incorrect_means, width=bar_w,
                    color="#F44336", label="Incorrect pairs")

    # annotate each bar with its mean value so the convergence pattern is immediately visible
    for bar, val in zip(list(bars_c) + list(bars_i), correct_means + incorrect_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=11)
    ax.set_ylabel("Mean score margin", fontsize=11)
    ax.set_title(
        "Skywork-Reward score margin: correct vs. incorrect pairs by category", fontsize=12
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, figures_dir, "figure6_score_margin")


# ---------------------------------------------------------------------------
# Figure 7 — incorrect rate heatmap by model and category (appendix)
# ---------------------------------------------------------------------------

def make_figure_incorrect_heatmap(
    model_metrics: dict,
    model_pairs: dict,
    figures_dir: Path,
) -> None:
    """Heatmap of incorrect rate (%) by judge and task category."""
    ordered = _ordered_models(model_metrics)
    heatmap_cats = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", "Overall"]
    cat_labels = [
        "Knowledge (mmlu-pro)",
        "Reasoning (livebench)",
        "Math (livebench)",
        "Coding (livecodebench)",
        "Overall",
    ]
    model_labels = [label for _, label, _ in ordered]

    data = np.array([
        [
            100.0 * model_metrics[key][cat]["incorrect"] / model_metrics[key][cat]["total"]
            if model_metrics[key][cat]["total"] > 0 else 0.0
            for key, _, _ in ordered
        ]
        for cat in heatmap_cats
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    # vmax=50 covers the worst observed incorrect rate (reward model on coding);
    # using the same YlOrRd palette as figure3 keeps the two heatmaps visually consistent
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=50, aspect="auto")

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_yticks(range(len(cat_labels)))
    ax.set_yticklabels(cat_labels, fontsize=10)

    # separator line above the Overall row, matching figure3 layout
    ax.axhline(len(heatmap_cats) - 1.5, color="white", linewidth=2)

    for i in range(len(heatmap_cats)):
        for j in range(len(ordered)):
            val = data[i, j]
            # threshold chosen so text stays readable against the YlOrRd gradient
            text_color = "white" if val > 32 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Incorrect rate (%)", fontsize=9)

    ax.set_title("Incorrect judgment rate by judge and category", fontsize=12)
    fig.tight_layout()
    _save(fig, figures_dir, "figure7_incorrect_heatmap")


# ---------------------------------------------------------------------------
# Figure registry — add new figures here
# ---------------------------------------------------------------------------

FIGURES: dict[str, callable] = {
    "failure_modes":        make_figure_failure_modes,
    "category_accuracy":    make_figure_category_accuracy,
    "inconsistency_heatmap": make_figure_inconsistency_heatmap,
    "agreement_matrix":     make_figure_agreement_matrix,
    "consensus":            make_figure_consensus,
    "score_margin":         make_figure_score_margin,
    "incorrect_heatmap":    make_figure_incorrect_heatmap,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading model outputs...")
    model_metrics, model_pairs = load_all_models()
    print(f"Loaded {len(model_metrics)} model(s).")
    print(f"Generating {len(FIGURES)} figure(s) -> {FIGURES_DIR}/\n")

    for name, fn in FIGURES.items():
        print(f"  [{name}]")
        fn(model_metrics, model_pairs, FIGURES_DIR)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
