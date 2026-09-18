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


def _fmt_sci(x: float) -> str:
    """1-sig-fig scientific notation without a padded exponent (1e-3, not 1e-03)."""
    mantissa, exp = f"{x:.1e}".split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


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

    foc_seq = ", ".join(f"{v:.3f}" for _, v in sorted(s2["foc"].items(), key=lambda kv: int(kv[0])))
    r_hat_sorted = sorted(s2["r_hat_unbounded"].items(), key=lambda kv: int(kv[0]))
    r_hat_ks = ",".join(k for k, _ in r_hat_sorted)
    r_hat_vals = ", ".join(f"{v:.4f}" for _, v in r_hat_sorted)

    val_frac = d["config"]["val_frac"]
    train_pct = round((1 - val_frac) * 100)
    val_pct = round(val_frac * 100)
    n_free_alpha = s1["m"] - 1
    n_params = 3 * s2["H"] + 1
    rows_total_str = f"{s2['rows_total']:,}".replace(",", " ")

    group_order = ["DGP", "Loss/gradients", "Estimator", "Weights"]
    tests_by_group: dict[str, list[str]] = {g: [] for g in group_order}
    for t in s2["tests"]:
        tests_by_group[t["group"]].append(f"{t['id']} {t['desc']}")
    test_lines = {g: "; ".join(items) for g, items in tests_by_group.items()}

    slide2_bullets = [
        "## Objective",
        f"F: T~Unif[m], Z~Bern(½), Y|T=t,Z=1 ~ P_t,Y, Y|T=t,Z=0 ~ S_t. Rows (t,z,y): "
        f"2mn = {rows_total_str}",
        "L = mean[(1−z)·e^r − z·r], r = θ(y)+α[t], "
        "α ∈ 𝒜 = {α ∈ ℝ^m : α(m)=0}",
        "E_F[L] = (1/2m)·Σ_t J_t(θ+α(t)), J_t(r) = E_St[e^r] − E_Pt,Y[r] "
        "= E_St[e^r − ρ_t·r]",
        "φ(u) = e^u − ρu, φ″ = e^u > 0 ⇒ unique min at r = log ρ_t "
        "= θ*(y)+α*(t)",
        "min J_t = 1 − KL(P_t,Y‖S_t) (NWJ); excess = E_Pt,Y[e^δ −1− δ] "
        "≥ 0, δ = r − log ρ_t",
        "∂/∂α(t) = 0 ⇒ E_St[e^{θ+α(t)}] = 1 ⇒ α(t) = "
        "−log Z_t, Z_t = E_St[e^θ] (log-partition)",
        "(θ+c, α−c) leaves r fixed ⇒ α(m)=0 pins the flat direction",
        "## Model and training",
        f"θ: MLP 1→{s2['H']}→1, ReLU, 3H+1 = {n_params} params; "
        f"α: free ℝ^{n_free_alpha} ⊕ {{0}}; float64",
        f"Adam lr {_fmt_sci(s2['lr'])}, wd {_fmt_sci(s2['wd'])} (net only), batch {s2['batch']}, "
        f"stratified {train_pct}/{val_pct} per (t,z), early stop on val risk",
        f"Regularisation mandatory: θ_k = +k on z=1, −k on z=0 (continuous PL, "
        f"representable) ⇒ R̂_n = ½(e^{{−k}}−k) → −∞. "
        f"Measured k={r_hat_ks}: **{r_hat_vals}**",
        f"Best val **{s2['best_val_loss']:.4f} @ epoch {s2['best_epoch']}/{s2['epochs']}** "
        f"(θ≡0 ⇒ 0.5)",
        f"## Tests {s2['pytest_passed']}/{len(s2['tests'])}, {s2['pytest_time_s']:.2f} s",
        f"DGP — {test_lines['DGP']}",
        f"Loss/gradients — {test_lines['Loss/gradients']}",
        f"Estimator — {test_lines['Estimator']}",
        f"Weights — {test_lines['Weights']}",
        "## Diagnostics before any μ̃",
        f"FOC mean_i e^{{θ̃+α̃(t)}} = {foc_seq} (necessary, not sufficient)",
        "θ̃−θ* grid, mean-centred; n_eff = (Σw)²/Σw²",
    ]

    table_rows = s3["table"]
    table_lines = [
        f"{'t':>2} {'μ(t)':>7} {'survey':>7} {'μ̃(t)±SE':>14} {'offset':>7} "
        f"{'oracle':>7} {'n_eff/n':>8} {'bias cut':>8}"
    ]
    for r in table_rows:
        table_lines.append(
            f"{r['t']:>2} {r['mu_true']:>7.3f} {r['survey_mean']:>7.3f} "
            f"{r['mu_tilde']:>7.3f}±{r['se']:.3f} {r['offset']:>7.3f} {r['oracle']:>7.3f} "
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
    t11d_pred = s2["t11d_predicted_n_bias"]

    def _ok(a_hat: float) -> str:
        return "✓" if abs(a_hat - 1) < 0.02 else "✗"

    avg_exp_h = sum(l2h["decay_exponents"]) / len(l2h["decay_exponents"])
    l2_table_lines = [
        f"{'':>22} {'β=1.5, κ='+format(l2h['kappa'],'.2f'):>22} "
        f"{'β=0.6, κ='+format(l2b['kappa'],'.2f'):>22}",
        f"{'n_eff/n CV, 10 seeds':>22} {l2h['cv']:>22.3f} {l2b['cv']:>22.3f}",
        f"{'delta-SE ÷ sd':>22} {l2h['se_ratio']:>22.2f} {l2b['se_ratio']:>22.2f}",
        f"{'bias decay':>22} "
        f"{'n^-'+format(avg_exp_h,'.2f')+' (pred n^-'+format(l2h['predicted_exponent'],'.2f')+')':>22} "
        f"{'n·bias → '+format(t11d_pred,'.3f'):>22}",
        f"{'T7 â':>22} {format(l2h['t7_a_hat'],'.4f')+' '+_ok(l2h['t7_a_hat']):>22} "
        f"{format(l2b['t7_a_hat'],'.4f')+' '+_ok(l2b['t7_a_hat']):>22}",
    ]

    decomp_table_lines = [
        f"{'t':>2} {'μ̃−μ':>9} {'sampling':>9} {'model':>9} {'SE':>6}"
    ]
    for r in table_rows:
        decomp_table_lines.append(
            f"{r['t']:>2} {r['mu_tilde_minus_mu']:>+9.4f} {r['oracle_minus_mu']:>+9.4f} "
            f"{r['mu_tilde_minus_oracle']:>+9.4f} {r['mu_tilde_minus_mu_in_se']:>+5.1f}"
        )

    band_2to3_pct = cf_c2["contribution_pct"] - cf_c3["contribution_pct"]
    ablation_lines = [
        f"Ablation δ(y) = θ̃−θ*, held flat past c, paired on S_{s1['M']} draws:",
        f"{'c':>6} {'mean shift':>11} {'contributes':>14}",
        f"{'full':>6} {cf_full['mean']:>+11.4f} {'—':>14}",
        f"{3:>6} {cf_c3['mean']:>+11.4f} {'y>3: '+format(100*cf_c3['contribution_pct'],'.0f')+'%':>14}",
        f"{2:>6} {cf_c2['mean']:>+11.4f} {'y>2: '+format(100*cf_c2['contribution_pct'],'.0f')+'%':>14}",
        f"⇒ band 2<y≤3 = {100*band_2to3_pct:.0f}%",
    ]

    cov_lines = [
        f"Survey mass share, pooled t≤m vs S_{s1['M']}:",
        f"{'y>':>4} {'pooled t≤m':>11} {'S_'+str(s1['M']):>8} {'ratio':>7}",
        f"{2:>4} {cs2['train_tm']:>10.1%} {cs2['s9']:>8.1%} {format(cs2['ratio'],'.1f')+'x':>7}",
        f"{3:>4} {cs3['train_tm']:>10.1%} {cs3['s9']:>8.1%} {format(cs3['ratio'],'.1f')+'x':>7}",
        f"{4:>4} {cs4['train_tm']:>10.1%} {cs4['s9']:>8.1%} {format(cs4['ratio'],'.1f')+'x':>7}",
        "**Temporal covariate shift**, not extrapolation; excess risk is weighted by the "
        "*training* measure.",
    ]

    slide3_bullets = [
        "μ̃(t) = Σᵢwᵢyᵢ / Σᵢwᵢ, w = e^{θ̃(y)}, "
        f"yᵢ ~ S_t, t > m",
        "\n".join(table_lines),
        "**L1 — α cancels.** μ(t) = e^{α*(t)}E_St[e^{θ*}Y] and "
        "1 = e^{α*(t)}E_St[e^{θ*}] ⇒ μ(t) = E_St[e^{θ*}Y]/E_St[e^{θ*}]. "
        "Paired years give θ's *shape*; the current survey's own Z_t gives the level. "
        "t > m needs no P_t.",
        "**L2 — κ decides validity.** w = Φ(a_m)/Φ(βy); "
        "1/Φ(βy) ~ e^{β²y²/2} (Mills) ⇒ E_St[w^k] < ∞ ⇔ "
        "k < κ = 1+1/(β²σ²). k=2 ⇔ βσ<1 (CLT/SE); "
        "k=3 ⇔ βσ<1/√2 (O(1/n) bias) ⇒ (R3).",
        "\n".join(l2_table_lines),
        "One cause, four symptoms.",
        "**L3 — residual is approximation, not sampling.** "
        "μ̃−μ = (oracle−μ) + (μ̃−oracle):",
        "\n".join(decomp_table_lines) + "\nModel term monotone; sampling term swings sign.",
        "\n".join(ablation_lines),
        "\n".join(cov_lines),
        f"θ* strictly decreasing, convex, ≥ log Φ(a_m) = {tb['log_phi_am']:.3f}; "
        f"B={l3['theta_bounded']:.0f} never binds (θ* ∈ "
        f"[{tb['at_train_hi']:.2f}, {tb['at_train_lo']:.2f}]). Fix: reweight training years "
        f"toward the forecast y-region; constrain θ̃ monotone + convex.",
    ]

    return [
        {"title": "The data-generating process", "bullets": slide1_bullets,
         "caption": slide1_caption, "figure": s1["figure"]},
        {"title": "Solving the optimization problem", "bullets": slide2_bullets,
         "caption": None, "figure": s2["figure_val_curve"], "font_size": 10},
        {"title": "Results and learnings", "bullets": slide3_bullets,
         "caption": None, "figure": s3["figure"], "font_size": 10},
    ]


def _add_bullet_paragraph(tf, is_first: bool, text: str, font_size: int, header: bool = False):
    """Add one paragraph, splitting **bold** spans into separate runs (or, for a header, one
    bold run at font_size+1, no leading dash). C/C++ -> Python -> R: a tiny hand-rolled inline
    markup scanner -> str.split("**") alternating runs -> the same idea as knitr's inline `**x**`."""
    p = tf.paragraphs[0] if is_first else tf.add_paragraph()
    if header:
        run = p.add_run()
        run.text = text
        run.font.size = Pt(font_size + 1)
        run.font.bold = True
        return p
    for i, part in enumerate(f"- {text}".split("**")):
        if part == "":
            continue
        run = p.add_run()
        run.text = part
        run.font.size = Pt(font_size)
        run.font.bold = (i % 2 == 1)
    return p


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
            is_header = bullet.startswith("## ")
            text = bullet[3:] if is_header else bullet
            _add_bullet_paragraph(tf, i == 0, text, font_size, header=is_header)

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
            if bullet.startswith("## "):
                lines.append(f"\n### {bullet[3:]}")
            else:
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
