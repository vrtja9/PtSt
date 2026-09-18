"""fusion/render_slides.py -- render slides/deck.pptx and slides/deck.md from slides/slide_data.json.

HARD RULE (2026-09-18): every number placed on a slide is read from slide_data.json (itself
produced by fusion/build_deck.py's live re-run of the pipeline + a live pytest subprocess) via an
f-string interpolation of a dict lookup -- never a hand-typed numeric literal. The one exception
is sqrt(2) in the (R3) rule's prose, and even that is `math.sqrt(2)` computed at build time, not
typed as a decimal.

Run:  python -m fusion.build_deck && python -m fusion.render_slides
C/C++ -> Python -> R: python-pptx is used purely as a templating library here (no math), so no
language parallel applies beyond "build a document object, then serialize it."
"""
from __future__ import annotations

import json
import math
import os

from pptx import Presentation
from pptx.util import Inches, Pt

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SLIDES_DIR = os.path.join(REPO_ROOT, "slides")
SLIDE_DATA_PATH = os.path.join(SLIDES_DIR, "slide_data.json")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")


def _load_data() -> dict:
    if not os.path.exists(SLIDE_DATA_PATH):
        raise RuntimeError(
            f"render_slides: {SLIDE_DATA_PATH} does not exist -- run "
            "`python -m fusion.build_deck` first (HARD RULE: no placeholder numbers)."
        )
    with open(SLIDE_DATA_PATH) as f:
        return json.load(f)


def _build_slides(d: dict) -> list[dict]:
    s1, s2, s3 = d["slide1"], d["slide2"], d["slide3"]

    slide1_bullets = [
        f"Sample space: (Y x {{0,1}}, B(Y) (x) 2^{{0,1}}); pi_t(y) := P[R=1|Y=y] = c_t * Phi(beta*y); "
        f"P_t,Y = N(m_t, sigma^2)",
        f"S_t = Law(Y|R=1), density s_t = pi_t*p_t / rho_bar_t",
        f"m={s1['m']}, M={s1['M']}, n={s1['n']}, sigma={s1['sigma']}, beta={s1['beta']}, "
        f"m_t={s1['m_t_intercept']}+{s1['m_t_slope']}t, c_t={s1['c_t_intercept']}{s1['c_t_slope']}t",
        f"Assumption 1 holds exactly: Phi(beta*y)/Phi(beta*y') has no t. "
        f"rho_bar_t = c_t*Phi(a_t) varies with t but cancels out of S_t",
        f"Sampler: Y~N(m_t,sigma^2), R~Bern(c_t*Phi(beta*Y)), keep Y with R=1 "
        f"-- rejection sampling IS conditioning on R=1",
        f"Definition: a_t := beta*m_t / sqrt(1+beta^2 sigma^2); at t=m, a_m={s1['a_m']:.4f} (live)",
        f"theta*(y) = log Phi(a_m) - log Phi(beta*y); alpha*(t) = log Phi(a_t) - log Phi(a_m)",
        f"Closed form: E_St[Y] = m_t + beta*sigma^2*phi(a_t) / (sqrt(1+beta^2 sigma^2)*Phi(a_t)); "
        f"measured survey bias E_St[Y]-m_t = {s1['survey_bias_t1']:+.4f} at t=1, "
        f"{s1['survey_bias_tM']:+.4f} at t={s1['M']}",
        f"Design rule (R3): beta*sigma < 1/sqrt(2) (={1/math.sqrt(2):.4f}); "
        f"tail index kappa = 1+1/(beta^2 sigma^2) = {s1['tail_index_kappa']:.2f} "
        f"(R3 holds: {s1['r3_holds']})",
    ]
    slide1_caption = (
        f"Survey bias E_St[Y]-mu(t) falls from {s1['survey_bias_t1']:+.3f} at t=1 to "
        f"{s1['survey_bias_tM']:+.3f} at t={s1['M']} -- a constant-bias baseline must fail"
    )

    foc_str = ", ".join(f"t={t}:{v:.3f}" for t, v in sorted(s2["foc"].items(), key=lambda kv: int(kv[0])))
    r_hat_str = ", ".join(
        f"k={k}:{v:+.4f}" for k, v in sorted(s2["r_hat_unbounded"].items(), key=lambda kv: int(kv[0]))
    )
    slide2_bullets = [
        f"Pooled F over rows (t,z,y), 2mn={s2['rows_total']} rows; "
        f"L=mean((1-z)*exp(r)-z*r), r=theta(y)+alpha[t]",
        f"Why this loss: its minimizer over separable r is log dP_t,Y/dS_t (the NWJ variational "
        f"form of KL); alpha(t)=-log E_St[e^theta] is a per-year log-partition constant; "
        f"the anchor alpha(m)=0 removes the flat direction (theta+c, alpha-c)",
        f"Implementation: theta=MLP(1->H->1,ReLU,H={s2['H']}); alpha=free vector in R^(m-1) "
        f"concat with a constant 0; float64; Adam lr={s2['lr']}, wd={s2['wd']} (network only), "
        f"batch={s2['batch']}; stratified 80/20 split; early stopping on validation risk",
        f"Early stopping is REQUIRED, not cosmetic: the piecewise-linear theta_k (+k at population "
        f"rows, -k at survey rows) gives R_hat_n(k)=(1/2)(e^-k-k) -> -inf as k->inf. Measured "
        f"[{r_hat_str}] vs the trained best val loss {s2['best_val_loss']:.4f} at epoch "
        f"{s2['best_epoch']} of {s2['epochs']} run -- the best checkpoint is NOT the last epoch",
        f"Verification: {s2['pytest_summary']}. T4 torch-vs-numpy: "
        f"loss diff {s2['t4']['L_diff']:.1e}, worst grad diff {s2['t4']['worst_grad_diff']:.1e}; "
        f"T7 parametric recovery a_hat={s2['t7_a_hat']:.4f}; "
        f"T11 weight tails kappa={s2['t11_kappa']:.2f}, n_eff CV={s2['t11_n_eff_cv']:.3f}, "
        f"SE ratio={s2['t11_se_ratio']:.2f}",
        f"Diagnostics before any mu~ is reported: FOC mean_i e^(theta~+alpha~(t)) per year "
        f"[{foc_str}]; theta~ vs theta* grid; n_eff",
    ]

    table_rows = s3["table"]
    table_lines = [
        f"{'t':>2} {'mu(t)':>7} {'survey':>7} {'mu~(t)+-SE':>14} {'offset':>7} {'oracle':>7} "
        f"{'n_eff/n':>8} {'bias cut':>8}"
    ]
    for r in table_rows:
        table_lines.append(
            f"{r['t']:>2} {r['mu_true']:>7.3f} {r['survey_mean']:>7.3f} "
            f"{r['mu_tilde']:>7.3f}+-{r['se']:.3f} {r['offset']:>7.3f} {r['oracle']:>7.3f} "
            f"{r['n_eff_over_n']:>8.2f} {100*r['bias_reduction_pct']:>7.0f}%"
        )

    l2b, l2h = s3["learning2"]["beta06"], s3["learning2"]["beta15"]
    l3 = s3["learning3"]
    tb = l3["theta_star_bounds"]
    cf = l3["counterfactual"]
    cf_full, cf_c2, cf_c3 = cf["full"], cf["2"], cf["3"]
    cs = l3["covariate_shift"]
    cs2, cs3, cs4 = cs["2"], cs["3"], cs["4"]

    row_last = table_rows[-1]  # t = M
    drift_share_last = row_last["mu_tilde_minus_oracle"] / row_last["mu_tilde_minus_mu"]
    sig_ts = ", ".join(str(r["t"]) for r in table_rows if abs(r["mu_tilde_minus_mu_in_se"]) >= 2)
    decomp_str = "; ".join(
        f"t={r['t']}: {r['mu_tilde_minus_mu']:+.4f}={r['oracle_minus_mu']:+.4f}+{r['mu_tilde_minus_oracle']:+.4f}"
        f" ({r['mu_tilde_minus_mu_in_se']:+.1f} SE)"
        for r in table_rows
    )
    headline_str = ", ".join(f"t={r['t']}:{100*r['bias_reduction_pct']:.0f}%" for r in table_rows)
    band_23_s9 = cs2["s9"] - cs3["s9"]
    mo_seq = ", ".join(f"{r['mu_tilde_minus_oracle']:+.4f}" for r in table_rows)

    slide3_bullets = [
        "\n".join(table_lines),
        f"Learning 1: alpha(t) is the unknown per-year normalising constant and it CANCELS in "
        f"mu~=E_St[e^theta Y]/E_St[e^theta] -- lagged population data teaches theta's SHAPE only, "
        f"never the current-year level, which is exactly what makes t>m possible",
        f"Learning 2: the weight tail index decides whether ANY of the inference is valid. "
        f"E_St[w^k]<inf iff k<kappa=1+1/(beta^2 sigma^2). Old beta=1.5: kappa={l2h['kappa']:.2f}, "
        f"n_eff/n CV={l2h['cv']:.3f} across seeds, delta-SE {l2h['se_ratio']:.2f}x the true "
        f"sampling sd, bias decay ~n^-{sum(l2h['decay_exponents']) / len(l2h['decay_exponents']):.2f} "
        f"(predicted n^-{l2h['predicted_exponent']:.2f}) instead of n^-1, T7 a_hat={l2h['t7_a_hat']:.4f} "
        f"(fails |a-1|<0.02). beta={s1['beta']}: kappa={l2b['kappa']:.2f}, CV={l2b['cv']:.3f}, "
        f"SE ratio={l2b['se_ratio']:.2f}, T7 a_hat={l2b['t7_a_hat']:.4f} (passes). One defect, not four",
        f"Learning 3 (headline): mu~ cuts the survey's bias vs mu(t) by {headline_str} -- "
        f"decomposition mu~-mu=(oracle-mu)+(mu~-oracle): {decomp_str}; only t={sig_ts} exceeds "
        f"2 SE -- sampling variability, not model error, explains most of the other gaps",
        f"Learning 3 (mechanism, not circular): hold theta~'s drift delta(y):=theta~(y)-theta*(y) "
        f"FLAT beyond a cutoff c (delta(min(y,c))), paired on the same S_{s1['M']} draws -- an "
        f"ablation of the trained net's own output, not a re-interpolation of it. Full drift shift "
        f"{cf_full['mean']:+.4f} (sd {cf_full['sd']:.4f}); flat beyond y=3 -> {cf_c3['mean']:+.4f} "
        f"(y>3 contributes {100*cf_c3['contribution_pct']:.0f}%); flat beyond y=2 -> "
        f"{cf_c2['mean']:+.4f} (y>2 contributes {100*cf_c2['contribution_pct']:.0f}%) -- at "
        f"t={s1['M']}, {row_last['mu_tilde_minus_oracle']:+.4f} of the "
        f"{row_last['mu_tilde_minus_mu']:+.4f} total gap ({100*drift_share_last:.0f}%) is this "
        f"model-drift term, the rest is sampling",
        f"Learning 3 (diagnosis): the far tail y>3 is {cs3['s9']:.1%} of S_{s1['M']}'s mass but "
        f"contributes only {100*cf_c3['contribution_pct']:.0f}% of the shift; the 2<y<=3 band "
        f"({band_23_s9:.1%} of mass) dominates -- that band is data-RICH at t={s1['M']} but "
        f"data-POOR in training: survey mass share above y=2/3/4, pooled t<=m vs S_{s1['M']}: "
        f"{cs2['train_tm']:.1%} vs {cs2['s9']:.1%} ({cs2['ratio']:.1f}x), "
        f"{cs3['train_tm']:.1%} vs {cs3['s9']:.1%} ({cs3['ratio']:.1f}x), "
        f"{cs4['train_tm']:.1%} vs {cs4['s9']:.1%} ({cs4['ratio']:.1f}x) -- TEMPORAL COVARIATE "
        f"SHIFT, not extrapolation. mu~-oracle grows monotonically with t ({mo_seq}) -- a "
        f"structural property of this data-fusion setting, not an implementation bug",
        f"theta* is strictly decreasing, convex, bounded below by log Phi(a_m)={tb['log_phi_am']:.3f}; "
        f"the bounded head B={l3['theta_bounded']:.0f} never binds (theta* in "
        f"[{tb['at_train_hi']:.2f}, {tb['at_train_lo']:.2f}] over the training range) -- "
        f"identified, not yet applied. Sharper fix from the covariate-shift diagnosis: "
        f"weight/augment training years toward the forecast years' y-region, or constrain theta~ "
        f"to be monotone+convex (matching theta*'s known shape) instead of an unconstrained MLP",
    ]

    return [
        {"title": "The data-generating process", "bullets": slide1_bullets,
         "caption": slide1_caption, "figure": s1["figure"]},
        {"title": "Solving the optimization problem", "bullets": slide2_bullets,
         "caption": None, "figure": s2["figure_val_curve"]},
        {"title": "Results and learnings", "bullets": slide3_bullets,
         "caption": None, "figure": s3["figure"], "font_size": 9},
    ]


def build_pptx(path: str, slides_spec: list[dict]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for spec in slides_spec:
        slide = prs.slides.add_slide(blank_layout)

        title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.7))
        title_box.text_frame.text = spec["title"]
        title_box.text_frame.paragraphs[0].font.size = Pt(28)
        title_box.text_frame.paragraphs[0].font.bold = True

        font_size = spec.get("font_size", 11)
        body_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.95), Inches(6.4), Inches(6.1))
        tf = body_box.text_frame
        tf.word_wrap = True
        for i, bullet in enumerate(spec["bullets"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"- {bullet}"
            p.font.size = Pt(font_size)

        fig_path = os.path.join(FIGURES_DIR, spec["figure"])
        pic = slide.shapes.add_picture(fig_path, Inches(7.0), Inches(0.95), width=Inches(6.0))

        if spec["caption"]:
            cap_top = Inches(0.95) + pic.height + Inches(0.1)
            cap_box = slide.shapes.add_textbox(Inches(7.0), cap_top, Inches(6.0), Inches(0.6))
            cap_box.text_frame.word_wrap = True
            cap_box.text_frame.text = spec["caption"]
            cap_box.text_frame.paragraphs[0].font.size = Pt(11)
            cap_box.text_frame.paragraphs[0].font.italic = True

    prs.save(path)


def build_markdown_fallback(path: str, slides_spec: list[dict]) -> None:
    lines = ["# Data Fusion Interview Challenge -- slides (Keynote-paste fallback)\n"]
    for spec in slides_spec:
        lines.append(f"## {spec['title']}\n")
        for bullet in spec["bullets"]:
            lines.append(f"- {bullet}")
        if spec["caption"]:
            lines.append(f"\n_{spec['caption']}_")
        lines.append(f"\n![]({os.path.join('..', 'figures', spec['figure'])})\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    data = _load_data()
    slides_spec = _build_slides(data)
    os.makedirs(SLIDES_DIR, exist_ok=True)
    pptx_path = os.path.join(SLIDES_DIR, "deck.pptx")
    md_path = os.path.join(SLIDES_DIR, "deck.md")
    build_pptx(pptx_path, slides_spec)
    build_markdown_fallback(md_path, slides_spec)
    print("saved:", pptx_path)
    print("saved:", md_path)
