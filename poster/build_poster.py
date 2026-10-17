"""
=============================================================================
SCRIPT: build_poster.py
PROJECT: AIPI 590 - Project 1: Why Was I Rejected?
AUTHOR: Tonantzin Real Rojas
DESCRIPTION:
    Builds the 36" x 24" (landscape) print poster as HTML from the saved
    results in models/, so every number on the poster comes from the data.
    Render to PDF with headless Chrome (see poster/README note in the main
    README, or run this script with --pdf).
=============================================================================
"""

import json
import math
import os
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POSTER_DIR = os.path.join(BASE_DIR, "poster")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Chart palette (reference data-viz palette, light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def load(name):
    with open(os.path.join(MODELS_DIR, name), "r") as f:
        return json.load(f)


audit = load("audit_disparity_results.json")
shap = load("shap_actionability_results.json")
meta = load("model_metadata.json")

overall = audit["overall"]
subgroups = audit["subgroups"]
comps = audit["comparisons"]
n_rejected = audit["metadata"]["model_rejected_sample_size"]
n_test = audit["metadata"]["test_sample_size"]
shares = shap["share_of_top_k_reasons_pct"]
not_actionable_pct = round(100.0 - shares["actionable"], 1)


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return 100 * (center - half), 100 * (center + half)


# ---------------------------------------------------------------------------
# Chart 1: what SHAP's top-3 reasons are made of (100% stacked bar)
# ---------------------------------------------------------------------------
def shap_bar_svg(width=900):
    segments = [
        ("Can change today", shares["actionable"], BLUE, "#ffffff"),
        ("Past payment history", shares["historical"], "#8f8e88", "#ffffff"),
        ("Bank-set credit limit", shares["institutional"], "#bdbcb6", INK),
        ("Demographics", shares["immutable"], "#dcdbd5", INK),
    ]
    bar_y, bar_h, gap = 64, 86, 3
    total = sum(s[1] for s in segments)
    x = 0.0
    parts = []
    labels = []
    for i, (name, pct, color, text_color) in enumerate(segments):
        w = width * pct / total
        rx = 6 if i in (0, len(segments) - 1) else 0
        parts.append(
            f'<rect x="{x + (gap if i else 0):.1f}" y="{bar_y}" width="{max(w - (gap if i else 0), 1):.1f}" '
            f'height="{bar_h}" fill="{color}" rx="{rx}"/>'
        )
        cx = x + w / 2
        if pct >= 25:
            parts.append(
                f'<text x="{cx:.1f}" y="{bar_y + bar_h / 2 + 13}" text-anchor="middle" font-size="38" '
                f'font-weight="700" fill="{text_color}">{pct:.0f}%</text>'
            )
        labels.append((name, pct, x, w, cx))
        x += w
    # Labels: big segments below the bar, small segments above (right-aligned)
    out = []
    for name, pct, x0, w, cx in labels:
        if pct >= 25:
            out.append(f'<text x="{x0 + 4:.1f}" y="{bar_y + bar_h + 40}" font-size="28" fill="{INK}">{name}</text>')
        elif name == "Bank-set credit limit":
            out.append(f'<text x="{x0 + w:.1f}" y="{bar_y - 16}" text-anchor="end" font-size="26" fill="{INK_2}">'
                       f'Credit limit {pct:.1f}% · demographics {shares["immutable"]:.0f}%</text>')
    return (
        f'<svg viewBox="0 0 {width} {bar_y + bar_h + 56}" width="100%" role="img" '
        f'aria-label="Share of SHAP top-3 reasons by whether the applicant can change them">'
        + "".join(parts) + "".join(out) + "</svg>"
    )


# ---------------------------------------------------------------------------
# Chart 2: access (plan rate) vs. effort, per demographic dimension (forest plot)
# ---------------------------------------------------------------------------
DIMENSIONS = [
    ("gender", "Sex"),
    ("age", "Age"),
    ("education", "Education"),
    ("marriage", "Marital status"),
]


def plan_rate_bars_svg(width=960):
    """Horizontal bars: share of each group that gets a feasible plan."""
    label_w = 330
    x0, x1 = label_w, 940
    v_max = 60.0
    head_h, row_h, bar_h, dim_gap = 48, 48, 38, 12
    top = 6

    def x(v):
        return x0 + v / v_max * (x1 - x0)

    avg = overall["feasibility_rate_pct"]
    els = []
    avg_segments = []
    y = top
    for key, dim_name in DIMENSIONS:
        comp = comps[key]
        sig = comp["is_feasibility_statistically_significant"]
        a, b = subgroups[comp["group_a"]], subgroups[comp["group_b"]]
        gap = abs(a["feasibility_rate_pct"] - b["feasibility_rate_pct"])
        badge_color = ORANGE if sig else INK_2
        badge = f"★ Significant gap: {gap:.1f} points" if sig else "No significant difference"
        els.append(f'<text x="0" y="{y + 30}" font-size="31" font-weight="600" fill="{INK}">{dim_name}</text>')
        els.append(f'<text x="{x0}" y="{y + 30}" font-size="27" font-weight="600" fill="{badge_color}">{badge}</text>')
        y += head_h
        bar_rows_top = y
        low_rate = min(a["feasibility_rate_pct"], b["feasibility_rate_pct"])
        for g in (a, b):
            rate = g["feasibility_rate_pct"]
            if sig:
                color = ORANGE if rate == low_rate else BLUE
            else:
                color = "#c4c3bd"
            cy = y + row_h / 2
            weight = "600" if sig and rate == low_rate else "400"
            els.append(f'<text x="22" y="{cy + 10}" font-size="29" font-weight="{weight}" fill="{INK}">{g["group"]}</text>')
            els.append(f'<rect x="{x0}" y="{cy - bar_h / 2:.1f}" width="{x(rate) - x0:.1f}" height="{bar_h}" rx="5" fill="{color}"/>')
            label_color = "#ffffff" if sig else INK
            els.append(f'<text x="{x(rate) - 12:.1f}" y="{cy + 10}" text-anchor="end" font-size="29" font-weight="600" fill="{label_color}">{rate:.1f}%</text>')
            y += row_h
        avg_segments.append((bar_rows_top, y))
        y += dim_gap
    bottom = y - dim_gap
    # Reference line: all rejected applicants
    height = bottom + 10
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Share of rejected applicants in each group who get a feasible plan">'
        + "".join(els) + "</svg>"
    )


# ---------------------------------------------------------------------------
# Worked example: the app's high-risk preset applicant
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "LIMIT_BAL": "Bank-set credit limit",
    "SEX": "Sex", "EDUCATION": "Education", "MARRIAGE": "Marital status", "AGE": "Age",
    "PAY_1": "Payment status, this month",
    **{f"PAY_{i}": f"Payment status, {i} months ago" for i in range(2, 7)},
    "BILL_AMT1": "Card balance, this month",
    **{f"BILL_AMT{i}": f"Card balance, {i} months ago" for i in range(2, 7)},
    "PAY_AMT1": "Amount paid, this month",
    **{f"PAY_AMT{i}": f"Amount paid, {i} months ago" for i in range(2, 7)},
}


def payment_status_label(code):
    code = int(round(code))
    if code == -2:
        return "no balance"
    if code == -1:
        return "paid in full on time"
    if code == 0:
        return "minimum payment made"
    return f"{code}-month delay"


def example_applicant():
    """Runs the real model, SHAP, and DiCE on the app's 'High-risk applicant' preset."""
    import joblib
    import pandas as pd
    sys.path.insert(0, BASE_DIR)
    from src.xai_shap import CreditSHAPExplainer, feature_category
    from src.xai_recourse import CreditRecourseEngine

    bill, paid, pay1 = 45000, 500, 2
    vector = {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_1": pay1, "PAY_2": pay1 - 1, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": bill, **{f"BILL_AMT{i}": int(bill * (1 - 0.05 * (i - 1))) for i in range(2, 7)},
        "PAY_AMT1": paid, "PAY_AMT2": int(paid * 0.9), **{f"PAY_AMT{i}": int(paid * 0.8) for i in range(3, 7)},
    }
    df = pd.DataFrame([vector])
    model = joblib.load(os.path.join(MODELS_DIR, "credit_model.joblib"))
    risk = float(model.predict_proba(df)[0, 1])

    _, _, shap_df = CreditSHAPExplainer().explain_instance(df)
    top = shap_df[shap_df["SHAP_Attribution"] > 0].sort_values("SHAP_Attribution", ascending=False).head(3)
    reasons = [(FEATURE_LABELS.get(f, f), feature_category(f) == "actionable") for f in top["Feature"]]

    # Same settings as the app (sample_size=1500), so the plan matches what the app shows
    cfs, _, _ = CreditRecourseEngine(sample_size=1500).generate_recourse_plans(df, total_cfs=2)
    plan = cfs.iloc[0]
    steps = []
    if plan["PAY_1"] < pay1:
        steps.append(("Catch up on payments", f"{payment_status_label(pay1)} ➔ {payment_status_label(plan['PAY_1'])}"))
    if plan["BILL_AMT1"] < bill - 0.5:
        steps.append(("Lower your card balance", f"${bill:,.0f} ➔ ${plan['BILL_AMT1']:,.0f}"))
    if plan["PAY_AMT1"] > paid + 0.5:
        steps.append(("Pay more this month", f"${paid:,.0f} ➔ ${plan['PAY_AMT1']:,.0f}"))
    new_risk = float(model.predict_proba(cfs.iloc[[0]][df.columns])[0, 1])
    return {"age": vector["AGE"], "pay1": pay1, "bill": bill, "paid": paid, "risk": risk,
            "reasons": reasons, "steps": steps, "new_risk": new_risk}


ex = example_applicant()
reasons_html = "".join(
    f'<li>{name} <span class="{"can" if can else "cant"}">{"✓ can change" if can else "✗ can\'t change"}</span></li>'
    for name, can in ex["reasons"]
)
steps_html = "".join(f'<li><b>{title}:</b><br>{detail}</li>' for title, detail in ex["steps"])


# ---------------------------------------------------------------------------
# Text derived from results
# ---------------------------------------------------------------------------
def disadvantaged_groups():
    out = []
    for key, _ in DIMENSIONS:
        c = comps[key]
        if c["is_feasibility_statistically_significant"]:
            a, b = subgroups[c["group_a"]], subgroups[c["group_b"]]
            low, high = (a, b) if a["feasibility_rate_pct"] < b["feasibility_rate_pct"] else (b, a)
            out.append((low["group"], low["feasibility_rate_pct"], high["group"], high["feasibility_rate_pct"]))
    return out


gaps = disadvantaged_groups()
gap_text = " and ".join(f"<b>{lo.lower()}</b> applicants ({lr:.1f}% vs. {hr:.1f}%)" for lo, lr, hi, hr in gaps)
low_groups = " or ".join(f"<b>{lo.lower()}</b>" for lo, lr, hi, hr in gaps)
any_effort_gap = any(c["is_burden_statistically_significant"] for c in comps.values())
min_effort_p_adj = min(c["welch_t_pvalue_holm_adjusted"] for c in comps.values())
ratios = [c["disparity_ratio"] for c in comps.values()]
effort_range = f"group ratios {min(ratios):.2f}x to {max(ratios):.2f}x"
if gaps and not any_effort_gap:
    rq3_headline = f'Effort is the same for everyone, but applicants who are {low_groups} get a plan <span class="hlw">less often</span>.'
elif gaps:
    rq3_headline = f'Effort differs by group, and applicants who are {low_groups} get a plan <span class="hlw">less often</span>.'
else:
    rq3_headline = "No significant differences between groups in effort or in getting a plan."
effort_note = (
    "No difference is significant."
    if not any_effort_gap else "Some effort differences are significant (corrected p &lt; 0.05)."
)
auc = meta["metrics"]["roc_auc"]
threshold_pct = int(round(audit["metadata"]["threshold"] * 100))

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Why Was I Rejected? Poster</title>
<style>
  @page {{ size: 36in 24in; margin: 0; }}
  :root {{
    --ink: {INK};
    --ink-2: {INK_2};
    --navy: #0b2a5b;
    --accent: {BLUE};
    --accent-soft: #eaf2fc;
    --warm: {ORANGE};
    --warm-soft: #fdf0ea;
    --rule: #dcdbd5;
    --panel: #f6f6f3;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    width: 36in; height: 24in; overflow: hidden;
    background: #ffffff; color: var(--ink);
    font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    font-size: 24pt; line-height: 1.32;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }}
  .poster {{ width: 36in; height: 24in; display: flex; flex-direction: column; }}

  /* Header */
  header {{
    background: var(--navy); color: #ffffff;
    padding: 0.55in 0.7in 0.5in 0.7in;
    display: flex; align-items: center; justify-content: space-between; gap: 0.6in;
  }}
  header h1 {{ font-size: 104pt; line-height: 1.0; margin: 0; font-weight: 800; letter-spacing: -1pt; }}
  header .sub {{ font-size: 36pt; margin: 0.16in 0 0.14in 0; font-weight: 500; color: #dbe6f7; }}
  header .byline {{ font-size: 25pt; color: #b9c9e4; font-weight: 500; }}
  .qr {{ display: flex; align-items: center; gap: 0.3in; }}
  .qr .qrbox {{ background: #ffffff; padding: 0.27in; border-radius: 0.1in; }}  /* >= 4-module quiet zone */
  .qr img {{ width: 2.0in; height: 2.0in; display: block; }}
  .qr .qrtext {{ font-size: 22pt; line-height: 1.3; color: #dbe6f7; max-width: 6in; text-align: right; }}
  .qr .qrtext b {{ color: #ffffff; font-size: 26pt; }}

  /* Columns */
  main {{
    flex: 1; display: grid;
    grid-template-columns: 7.55fr 8.35fr 9.9fr 7.75fr;
    gap: 0.8in; padding: 0.45in 0.8in 0.4in 0.8in;
    min-height: 0;
  }}
  .col {{ display: flex; flex-direction: column; gap: 0.3in; min-height: 0; }}
  .col.fill {{ justify-content: space-between; }}
  h2 {{
    font-size: 44pt; line-height: 1.05; margin: 0; color: var(--navy); font-weight: 800;
    padding-bottom: 0.1in; border-bottom: 5pt solid var(--navy);
  }}
  h2 .num {{ color: var(--accent); margin-right: 0.12in; }}
  h3 {{ font-size: 29pt; margin: 0 0 0.06in 0; color: var(--navy); font-weight: 700; line-height: 1.15; }}
  p {{ margin: 0; }}
  b, strong {{ font-weight: 600; }}
  .charttitle {{ font-size: 23pt; font-weight: 600; color: var(--ink); margin-top: 0.04in; }}
  ul, ol {{ margin: 0; padding-left: 0.36in; }}
  li {{ margin: 0 0 0.07in 0; }}
  .muted {{ color: var(--ink-2); }}
  .small {{ font-size: 20pt; line-height: 1.3; }}
  .src {{ font-size: 17pt; color: var(--ink-2); line-height: 1.25; }}

  /* Stat tiles */
  .stats {{ display: flex; flex-direction: column; gap: 0.16in; }}
  .stat {{ display: grid; grid-template-columns: 2.35in 1fr; align-items: center; gap: 0.2in;
           background: var(--panel); border-radius: 0.12in; padding: 0.16in 0.22in; }}
  .stat .big {{ font-size: 72pt; font-weight: 700; color: var(--navy); line-height: 1; }}
  .stat .what {{ font-size: 22.5pt; line-height: 1.27; }}

  .callout {{ background: var(--accent-soft); border-left: 8pt solid var(--accent); border-radius: 0.08in;
              padding: 0.18in 0.24in; }}
  .callout.warm {{ background: var(--warm-soft); border-left-color: var(--warm); }}
  .rq {{ display: grid; grid-template-columns: 0.75in 1fr; gap: 0.08in 0.12in; align-items: baseline; }}
  .rq .tag {{ font-weight: 700; color: var(--accent); font-size: 26pt; }}

  /* Pipeline */
  .pipeline {{ display: flex; flex-direction: column; gap: 0.14in; }}
  .step {{ display: grid; grid-template-columns: 0.62in 1fr; gap: 0.16in; align-items: start; }}
  .step .n {{ width: 0.62in; height: 0.62in; border-radius: 50%; background: var(--navy); color: #fff;
              font-weight: 700; font-size: 26pt; display: flex; align-items: center; justify-content: center; }}
  .step .t {{ font-size: 22pt; line-height: 1.28; }}
  .step .t b {{ color: var(--navy); }}

  .rulebox {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.18in; font-size: 20pt; line-height: 1.32; }}
  .rb {{ border-radius: 0.1in; padding: 0.14in 0.18in; }}
  .rb.go {{ background: var(--accent-soft); }}
  .rb.lock {{ background: var(--panel); }}
  .rbt {{ font-size: 23pt; font-weight: 700; margin-bottom: 0.06in; }}
  .rb.go .rbt {{ color: var(--accent); }}
  .rb.lock .rbt {{ color: var(--ink-2); }}
  table.rules {{ width: 100%; border-collapse: collapse; font-size: 20pt; line-height: 1.25; }}
  table.rules th {{ text-align: left; background: var(--navy); color: #fff; padding: 0.07in 0.12in; font-weight: 600; }}
  table.rules td {{ padding: 0.07in 0.12in; border-bottom: 1.5pt solid var(--rule); vertical-align: top; }}
  table.rules tr.go td:first-child {{ color: var(--accent); font-weight: 600; }}
  table.rules tr.lock td:first-child {{ color: var(--ink-2); font-weight: 600; }}

  figure {{ margin: 0; }}
  figure img.shot {{ width: 100%; display: block; border-radius: 0.08in; border: 2pt solid var(--rule); }}
  figcaption {{ font-size: 19pt; color: var(--ink-2); margin-top: 0.06in; line-height: 1.25; }}

  /* Findings */
  .finding {{ display: flex; flex-direction: column; gap: 0.1in; }}
  .headline {{ font-size: 27pt; font-weight: 400; line-height: 1.18; color: var(--ink); }}
  .headline .hl {{ color: var(--accent); font-weight: 700; }}
  .headline .hlw {{ color: var(--warm); font-weight: 700; }}
  .minis {{ display: flex; gap: 0.2in; }}
  .mini {{ flex: 1; background: var(--panel); border-radius: 0.1in; padding: 0.12in 0.18in; }}
  .mini .big {{ font-size: 40pt; font-weight: 700; color: var(--navy); line-height: 1.05; }}
  .mini .what {{ font-size: 19pt; line-height: 1.22; }}
  .plan {{ display: grid; grid-template-columns: auto 1fr; gap: 0.06in 0.2in; align-items: baseline;
           background: var(--accent-soft); border-radius: 0.1in; padding: 0.14in 0.22in; font-size: 21pt; }}
  .plan .k {{ color: var(--navy); font-weight: 700; }}

  .example {{ background: var(--panel); border-radius: 0.12in; padding: 0.18in 0.22in; display: flex; flex-direction: column; gap: 0.08in; }}
  .example .profile {{ font-size: 21pt; color: var(--ink-2); }}
  .example .verdict {{ font-size: 24pt; font-weight: 600; color: #b42318; }}
  .example .twocol {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.16in; }}
  .example .box {{ background: #ffffff; border-radius: 0.1in; padding: 0.12in 0.16in; border: 2pt solid var(--rule); }}
  .example .planbox {{ border-color: var(--accent); }}
  .example .lbl {{ font-size: 22pt; font-weight: 700; color: var(--navy); margin-bottom: 0.04in; }}
  .example ol {{ padding-left: 0.32in; font-size: 19pt; line-height: 1.25; }}
  .example li {{ margin-bottom: 0.06in; }}
  .example .can {{ color: var(--accent); font-weight: 600; white-space: nowrap; }}
  .example .cant {{ color: var(--ink-2); font-weight: 600; white-space: nowrap; }}
  .example .approved {{ font-size: 19pt; font-weight: 600; color: #067647; }}
  .effortline {{ background: var(--panel); border-radius: 0.1in; padding: 0.12in 0.2in; font-size: 21pt; line-height: 1.28; }}
  .takeaways ol {{ padding-left: 0.42in; }}
  .takeaways li {{ font-size: 23pt; margin-bottom: 0.09in; }}
  .tight li {{ font-size: 22.5pt; line-height: 1.3; margin-bottom: 0.06in; }}
  .refs {{ font-size: 15pt; line-height: 1.25; color: var(--ink-2); }}
  .refs p {{ margin: 0 0 0.03in 0; text-indent: -0.2in; padding-left: 0.2in; }}
</style>
</head>
<body>
<div class="poster">
<header>
  <div>
    <h1>Why Was I Rejected?</h1>
    <p class="sub">Turning credit rejections into actionable plans, and auditing who gets one</p>
    <div class="byline">Tonantzin Real Rojas &nbsp;·&nbsp; AIPI 590: Explainable AI &nbsp;·&nbsp; Duke University</div>
  </div>
  <div class="qr">
    <div class="qrtext"><b>Try the app &amp; see the code</b><br>github.com/tonantzin-real/<br>AIPI_590_Project_1<br><span style="font-size:18pt">Live demo GIF, notebook, and full results inside</span></div>
    <div class="qrbox"><img src="assets/qr_repo.svg" alt="QR code linking to the GitHub repository"></div>
  </div>
</header>

<main>
  <!-- ===================== COLUMN 1: MOTIVATION ===================== -->
  <section class="col fill">
    <h2><span class="num">1</span>Why this matters</h2>
    <p>In Mexico, millions of people are outside the formal financial system, and few know what lenders look for.</p>
    <div class="stats">
      <div class="stat"><div class="big">53%</div><div class="what">of Mexican adults have a formal financial account (vs. 73.5% in Latin America &amp; Caribbean)<br><span class="src">World Bank Global Findex, 2024 data</span></div></div>
      <div class="stat"><div class="big">37%</div><div class="what">of adults (18–70) hold any formal credit<br><span class="src">INEGI &amp; CNBV, ENIF 2024</span></div></div>
      <div class="stat"><div class="big">61%</div><div class="what">score below the minimum desirable level of financial knowledge<br><span class="src">BBVA Research, 2019 (OECD/INFE method)</span></div></div>
    </div>
    <div class="callout">
      <h3>The problem</h3>
      <p>When an algorithm rejects someone, today's explanations (e.g. <b>SHAP</b>) say <i>why</i>: "your payment history raised your risk." That helps the lender, but not the applicant, who needs to know <b>what to do next</b>.</p>
    </div>
    <p><b>Audience:</b> people applying for credit, especially first-time and underserved borrowers, not the lender.</p>
    <div>
      <h3>Research questions</h3>
      <div class="rq">
        <div class="tag">RQ1</div><div>How actionable are SHAP's explanations for rejected applicants?</div>
        <div class="tag">RQ2</div><div>Can we offer realistic plans using only changes a person can make <b>today</b>?</div>
        <div class="tag">RQ3</div><div>Is that path to approval <b>equally available</b> across sex, age, education, and marital status?</div>
      </div>
    </div>
  </section>

  <!-- ===================== COLUMN 2: WHAT I BUILT ===================== -->
  <section class="col fill">
    <h2><span class="num">2</span>What I built</h2>
    <div class="pipeline">
      <div class="step"><div class="n">1</div><div class="t"><b>Credit model.</b> LightGBM on 30,000 credit card clients (UCI, Taiwan 2005), ROC-AUC {auc:.2f}. Rejected if predicted default risk ≥ {threshold_pct}%.</div></div>
      <div class="step"><div class="n">2</div><div class="t"><b>Why? (SHAP).</b> Which of each person's top reasons could they actually change?</div></div>
      <div class="step"><div class="n">3</div><div class="t"><b>What to do? (DiCE).</b> Plans that only change what a person controls, in plain language.</div></div>
      <div class="step"><div class="n">4</div><div class="t"><b>Fairness audit.</b> All {n_rejected:,} rejected test applicants, across 4 demographic dimensions.</div></div>
      <div class="step"><div class="n">5</div><div class="t"><b>Web app</b> where anyone can try a profile.</div></div>
    </div>
    <div>
      <h3>Key design decision: what can a plan change?</h3>
      <div class="rulebox">
        <div class="rb go"><div class="rbt">✓ Can change</div>Payment status: only improve<br>Balance: only go down<br>Payment: only go up, up to the bill</div>
        <div class="rb lock"><div class="rbt">🔒 Locked</div>Who you are: sex, age, education, marital status<br>The past: previous months<br>The lender's: credit limit</div>
      </div>
    </div>
    <p class="small"><b>Effort</b> = how big a plan's changes are (normalized L1 distance). Group gaps tested with Fisher's exact and Welch's t-tests, corrected for multiple comparisons.</p>
    <div class="example">
      <h3>Example: one rejected applicant</h3>
      <p class="profile">{ex['age']} years old · {payment_status_label(ex['pay1'])} · ${ex['bill']:,} balance · paid ${ex['paid']:,} last month</p>
      <p class="verdict">✗ Rejected: {ex['risk'] * 100:.1f}% predicted risk of default</p>
      <div class="twocol">
        <div class="box shapbox"><div class="lbl">SHAP: <i>why?</i></div><ol>{reasons_html}</ol></div>
        <div class="box planbox"><div class="lbl">Our plan: <i>what to do</i></div><ol>{steps_html}</ol>
          <p class="approved">✓ Approved: risk drops to {ex['new_risk'] * 100:.1f}%</p></div>
      </div>
    </div>
  </section>

  <!-- ===================== COLUMN 3: FINDINGS ===================== -->
  <section class="col">
    <h2><span class="num">3</span>What I found</h2>

    <div class="finding">
      <h3>RQ1 · SHAP explains the past, not the next step</h3>
      <div class="headline"><span class="hl">{not_actionable_pct:.0f}%</span> of SHAP's top reasons are things the applicant <b>can't change today</b>.</div>
      {shap_bar_svg()}
      <div class="minis">
        <div class="mini"><div class="big">{shap['applicants_whose_top_reason_is_not_actionable_pct']:.1f}%</div><div class="what">of rejected applicants: the #1 reason is not actionable</div></div>
        <div class="mini"><div class="big">{shap['applicants_with_no_actionable_reason_in_top_k_pct']:.1f}%</div><div class="what">get no actionable reason at all in their top 3</div></div>
      </div>
    </div>

    <div class="finding">
      <h3>RQ2 · Realistic plans are possible for about half</h3>
      <div class="headline"><span class="hl">{overall['feasibility_rate_pct']:.1f}%</span> of rejected applicants ({overall['successful_total']:,} of {overall['evaluated_total']:,}) get a realistic plan using only 3 things they control.</div>
    </div>

    <div class="finding">
      <h3>RQ3 · Equal effort, unequal access</h3>
      <div class="headline">{rq3_headline}</div>
      <p class="charttitle">Out of the rejected applicants in each group, how many got a plan?</p>
      {plan_rate_bars_svg(width=960)}
      <div class="effortline"><b>Effort, once a plan exists:</b> about the same for every group ({effort_range}). {effort_note}</div>
    </div>
  </section>

  <!-- ===================== COLUMN 4: TAKEAWAYS, LIMITATIONS, FUTURE ===================== -->
  <section class="col fill">
    <h2><span class="num">4</span>Takeaways</h2>
    <div class="takeaways callout">
      <ol>
        <li><b>Explanations should be actionable.</b> Attribution alone leaves most rejected applicants without a path forward.</li>
        <li><b>Audit access, not only effort.</b> Here the unfairness is <i>who gets a plan at all</i>.</li>
        <li><b>Realistic plans are possible</b> for about half of rejected applicants.</li>
      </ol>
    </div>

    <div>
      <h3>Limitations</h3>
      <ul class="tight">
        <li><b>Taiwanese data (2005)</b>, not Mexican; amounts are not in pesos.</li>
        <li><b>Rejections are simulated</b> from predicted default risk.</li>
        <li><b>Only 3 things can change</b>; no income or savings data.</li>
        <li><b>Effort is statistical</b>, not the real cost for a given person.</li>
        <li><b>Age and marital status overlap</b>; not tested jointly.</li>
      </ul>
    </div>

    <div>
      <h3>Risks &amp; responsible deployment</h3>
      <ul class="tight">
        <li><b>Unaffordable plans</b> could push people to informal loans.</li>
        <li><b>Plans are guidance</b>, not guarantees.</li>
        <li><b>Explanations need audits</b> to avoid fairwashing.</li>
      </ul>
    </div>

    <div class="callout warm">
      <h3>Future work: from v1 to Mexico</h3>
      <ul class="tight">
        <li>Retrain with a <b>Mexican institution's data</b>.</li>
        <li><b>Spanish, mobile-first</b> app.</li>
        <li>Affordable, multi-month plans.</li>
        <li>User study with unbanked users.</li>
      </ul>
    </div>

    <div class="refs">
      <p>Wachter, Mittelstadt &amp; Russell (2017). Counterfactual explanations without opening the black box. <i>Harv. JOLT</i>.</p>
      <p>Ustun, Spangher &amp; Liu (2019). Actionable recourse in linear classification. <i>FAT*</i>.</p>
      <p>Mothilal, Sharma &amp; Tan (2020). Diverse counterfactual explanations (DiCE). <i>FAT*</i>.</p>
      <p>Lundberg &amp; Lee (2017). A unified approach to interpreting model predictions. <i>NeurIPS</i>.</p>
      <p>Sharma et al. (2020). CERTIFAI. <i>AIES</i>. · Karimi et al. (2022). Survey of algorithmic recourse. <i>ACM CSUR</i>.</p>
      <p>Yeh &amp; Lien (2009). <i>Expert Syst. Appl.</i> · INEGI &amp; CNBV (2025) ENIF 2024 · World Bank (2025) Global Findex · BBVA Research (2019). Full list in the repo.</p>
    </div>
  </section>
</main>
</div>
</body>
</html>
"""

html_path = os.path.join(POSTER_DIR, "poster.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"[poster] HTML written to {html_path}")

if "--pdf" in sys.argv or "--png" in sys.argv:
    url = "file://" + html_path
    if "--pdf" in sys.argv:
        pdf_path = os.path.join(POSTER_DIR, "poster.pdf")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", url], check=True, capture_output=True)
        print(f"[poster] PDF written to {pdf_path}")
    if "--png" in sys.argv:
        png_path = os.path.join(POSTER_DIR, "poster_preview.png")
        # 36in x 24in at 96 CSS px per inch
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--window-size=3456,2304", f"--screenshot={png_path}", url], check=True, capture_output=True)
        print(f"[poster] Preview written to {png_path}")
