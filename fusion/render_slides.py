"""fusion/render_slides.py -- Phase 5 step 2: slides/deck.pptx (docs/challenge_text.md §4).

CP5 text = `Go` (notes/decisions.md). Fallback slides/deck.md is written alongside the pptx
per PROMPT_kickoff.md Phase 5 step 2 ("fallback: slides/deck.md for Keynote paste").

C/C++ -> Python -> R: python-pptx is used purely as a templating library here (no math), so no
language parallel applies beyond "build a document object, then serialize it."
"""
from __future__ import annotations

import os

from pptx import Presentation
from pptx.util import Inches, Pt

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SLIDES_DIR = os.path.join(REPO_ROOT, "slides")

SLIDES = [
    {
        "title": "The data-generating process",
        "bullets": [
            "P_t,Y = N(m_t, sigma^2), m_t = -0.5 + 0.25t; sigma=1",
            "Survey selection: pi_t(y) = c_t * Phi(beta*y), c_t = 0.9 - 0.06t, beta=1.5",
            "S_t = Y | R=1; m=6 (P_t years), M=9 (S_t years), n=2000 per (t,z) cell",
            "Survey bias E_St[Y] minus mu(t) shrinks from about 0.78 (t=1) to about 0.12 (t=9) by design",
        ],
        "figure": "phase1a_densities_4b6ab92b.png",
    },
    {
        "title": "Optimization pipeline",
        "bullets": [
            "Pooled convex loss: L = mean[(1-z)*exp(r) - z*r], r = theta(y) + alpha(t)",
            "theta = MLP(1->32->1, ReLU, bounded head B=20); alpha in R^6, alpha(6) anchored to 0",
            "Adam, lr=1e-3, wd=1e-5 (theta only), batch 256, 400 epochs, best-val-loss checkpoint",
            "CP3: first run (lr=3e-3, unbounded) was spiky and theta~ blew up in the tail; fixed by lowering lr and adding the bounded head",
            "Every mu~ is gated on FOC~=1 (per year) and a theta~ vs theta* plot before being trusted",
        ],
        "figure": "phase3_val_curve_8e4fef38.png",
    },
    {
        "title": "Comparison and learnings",
        "bullets": [
            "mu~(t) tracks mu(t) closer than the survey mean or the offset baseline at every t",
            "Learning 1: mu~ beats both baselines in- and out-of-sample (t=9: 1.752 vs true 1.75, survey mean 1.858, offset 1.601)",
            "Learning 2: an unbounded theta net can blow up in sparse tails (theta~ hit about 380 at y=-6 pre-fix); a bounded head fixed both training stability and stress-test robustness",
            "Learning 3: breaking Assumption 1 still biases mu~, but NOT in the direction the math's own qualitative prediction suggested -- reported as observed, not smoothed over",
        ],
        "figure": "phase4_main_figure_8e4fef38.png",
    },
]


def build_pptx(path: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for slide_spec in SLIDES:
        slide = prs.slides.add_slide(blank_layout)

        title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.8))
        title_box.text_frame.text = slide_spec["title"]
        title_box.text_frame.paragraphs[0].font.size = Pt(32)
        title_box.text_frame.paragraphs[0].font.bold = True

        body_box = slide.shapes.add_textbox(Inches(0.4), Inches(1.1), Inches(5.8), Inches(6.0))
        tf = body_box.text_frame
        tf.word_wrap = True
        for i, bullet in enumerate(slide_spec["bullets"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"- {bullet}"
            p.font.size = Pt(15)

        fig_path = os.path.join(REPO_ROOT, "figures", slide_spec["figure"])
        slide.shapes.add_picture(fig_path, Inches(6.4), Inches(1.1), width=Inches(6.5))

    prs.save(path)


def build_markdown_fallback(path: str) -> None:
    lines = ["# Data Fusion Interview Challenge -- slides (Keynote-paste fallback)\n"]
    for slide_spec in SLIDES:
        lines.append(f"## {slide_spec['title']}\n")
        for bullet in slide_spec["bullets"]:
            lines.append(f"- {bullet}")
        lines.append(f"\n![]({os.path.join('..', 'figures', slide_spec['figure'])})\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(SLIDES_DIR, exist_ok=True)
    pptx_path = os.path.join(SLIDES_DIR, "deck.pptx")
    md_path = os.path.join(SLIDES_DIR, "deck.md")
    build_pptx(pptx_path)
    build_markdown_fallback(md_path)
    print("saved:", pptx_path)
    print("saved:", md_path)
