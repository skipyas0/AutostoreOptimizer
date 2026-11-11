"""Table helpers: mean ± std formatting and CSV / Markdown / LaTeX / PNG output."""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


def mean_std(values, fmt="{:.2f}"):
    """'mean ± std' over the non-NaN values ('n/a' if none, the value if only one)."""
    s = pd.Series(values, dtype=float).dropna()
    if len(s) == 0:
        return "n/a"
    if len(s) == 1:
        return fmt.format(s.iloc[0])
    return f"{fmt.format(s.mean())} ± {fmt.format(s.std(ddof=1))}"


def _latex_escape(x):
    return (str(x).replace("\\", r"\textbackslash{}").replace("%", r"\%")
            .replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")
            .replace("±", r"$\pm$").replace("≤", r"$\le$").replace("·", r"$\cdot$"))


def to_latex(df, caption, label):
    cols = "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{tab:{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(_latex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_latex_escape(v) for v in row) + r" \\"
              for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def to_markdown(df, caption=None):
    md = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    head = f"**{caption}**\n\n" if caption else ""
    return head + "\n".join(md) + "\n"


def write_table(df, out_dir, name, caption, png=False):
    """Write <name>.csv, .md and .tex (and a rendered .png if png=True)."""
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write(to_markdown(df, caption))
    tex = to_latex(df, caption, name)
    with open(os.path.join(out_dir, f"{name}.tex"), "w") as f:
        f.write(tex)
    if png:
        render_png(tex, df, os.path.join(out_dir, f"{name}.png"))


# --------------------------------------------------------------------------- #
# PNG rendering (pdflatex if available, matplotlib table otherwise)
# --------------------------------------------------------------------------- #
def _find_binary(name):
    for c in (shutil.which(name), f"/Library/TeX/texbin/{name}",
              f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}", f"/usr/bin/{name}"):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _render_pdflatex(tex, out_png, dpi):
    pdflatex = _find_binary("pdflatex")
    if not pdflatex:
        return False
    doc = (r"\documentclass{article}" "\n"
           r"\usepackage[paperwidth=14in,paperheight=10in,margin=0.3in]{geometry}" "\n"
           r"\usepackage{booktabs}\usepackage{caption}\pagestyle{empty}" "\n"
           r"\begin{document}" "\n" + tex + r"\end{document}" "\n")
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "t.tex"), "w") as f:
            f.write(doc)
        subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                       cwd=tmp, capture_output=True)
        pdf = os.path.join(tmp, "t.pdf")
        if not os.path.isfile(pdf):
            return False
        try:
            import pypdfium2 as pdfium
            img = pdfium.PdfDocument(pdf)[0].render(scale=dpi / 72).to_pil()
        except Exception:
            ppm = _find_binary("pdftoppm")
            if not ppm:
                return False
            subprocess.run([ppm, "-png", "-r", str(dpi), "-singlefile", "t.pdf", "r"],
                           cwd=tmp, capture_output=True)
            if not os.path.isfile(os.path.join(tmp, "r.png")):
                return False
            from PIL import Image
            img = Image.open(os.path.join(tmp, "r.png"))
        from PIL import Image, ImageChops
        img = img.convert("RGB")
        bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
        if bbox:
            pad = int(dpi * 0.08)
            img = img.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                            min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad)))
        img.save(out_png, dpi=(dpi, dpi))
    return True


def _render_matplotlib(df, out_png, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * df.shape[1]), 0.35 * len(df) + 0.8))
    ax.axis("off")
    t = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(list(range(df.shape[1])))
    t.scale(1, 1.4)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#e4e3df")
        if r == 0:
            cell.set_text_props(weight="bold")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_png(tex, df, out_png, dpi=300):
    if not _render_pdflatex(tex, out_png, dpi):
        _render_matplotlib(df, out_png, dpi)


def fmt_num(x, fmt="{:.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)
