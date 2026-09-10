import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd


def find_experiment_folders(instances, experiment):
    # find target config
    for instance in instances:
        if os.path.isdir(
            f"precalculated_instances/{instance}/experiments/{experiment}"
        ) and os.path.isfile(
            f"precalculated_instances/{instance}/experiments/{experiment}/experiment_config.json"
        ):
            with open(
                f"precalculated_instances/{instance}/experiments/{experiment}/experiment_config.json",
                "r",
            ) as f:
                target_config = json.load(f)
            break

    if target_config is None:
        raise ValueError(
            f"Experiment {experiment} is not found in instance list {instances} or it wasn't finished properly."
        )

    # find the latest matching experiment in each instance
    instance_experiments = {}
    for instance in instances:
        print(f"Looking for target config in experiments of instance {instance}")
        exp_folders = os.listdir(f"precalculated_instances/{instance}/experiments/")
        sorted_exp_folders = sorted(
            exp_folders,
            key=lambda x: datetime.strptime(x, "%d-%m-%Y_%H-%M-%S"),
            reverse=True,
        )
        for exp_folder in sorted_exp_folders:
            print(f"Trying experiment {exp_folder}...")
            experiment_path = (
                f"precalculated_instances/{instance}/experiments/{exp_folder}"
            )
            with open(
                f"{experiment_path}/experiment_config.json",
                "r",
            ) as f:
                this_config = json.load(f)

            if this_config == target_config:
                print(" - Found target")
                instance_experiments[instance] = experiment_path
                break
        if instance not in instance_experiments:
            print(
                f"(WARNING) Did not find any experiment with target config for instance {instance}."
            )

    return instance_experiments


def format_to_latex(
    df: pd.DataFrame,
    caption: str = "Comparison of LNS and CP performance across instances.",
    label: str = "tab:lns_cp_comparison",
    include_averages: bool = True,
) -> str:
    """Format experiment results DataFrame into a LaTeX booktabs table string."""
    headers = list(df.columns)
    latex_headers = [h.replace("%", r"\%").replace("_", r"\_") for h in headers]
    col_spec = "l" + "r" * (len(headers) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{h}}}" for h in latex_headers) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        cells = []
        for i, col in enumerate(headers):
            val = row[col]
            if i == 0:
                cells.append(str(val).replace("_", r"\_"))
            else:
                cells.append(f"{float(val):.2f}")
        lines.append(" & ".join(cells) + r" \\")

    if include_averages and not df.empty:
        avg_values = df.mean(numeric_only=True)
        avg_cells = [r"\textbf{Average}"]
        for col in headers[1:]:
            val = avg_values.get(col, np.nan)
            avg_cells.append(f"{float(val):.2f}" if not np.isnan(val) else "--")
        lines.append(r"\midrule")
        lines.append(" & ".join(avg_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def _find_binary(name: str):
    candidates = [
        shutil.which(name),
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/Library/TeX/texbin/{name}",
        f"/usr/bin/{name}",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _find_pdflatex():
    return _find_binary("pdflatex")


def _render_table_matplotlib(latex_str: str, output_path: str, dpi: int = 300):
    import matplotlib.pyplot as plt

    lines = latex_str.strip().split("\n")
    table_lines = []
    in_tabular = False
    for line in lines:
        sline = line.strip()
        if r"\begin{tabular}" in sline:
            in_tabular = True
            continue
        if r"\end{tabular}" in sline:
            in_tabular = False
            continue
        if in_tabular:
            if sline in (r"\toprule", r"\midrule", r"\bottomrule"):
                continue
            if sline.endswith(r"\\"):
                sline = sline[:-2].strip()
            if sline:
                cells = [
                    c.strip()
                    .replace(r"\%", "%")
                    .replace(r"\_", "_")
                    .replace(r"\textbf{", "")
                    .replace("}", "")
                    for c in sline.split("&")
                ]
                table_lines.append(cells)

    if not table_lines:
        return

    headers = table_lines[0]
    data = table_lines[1:]

    fig, ax = plt.subplots(
        figsize=(max(8.0, len(headers) * 2.2), max(2.5, len(data) * 0.42 + 1.0)),
        dpi=dpi,
    )
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif r == len(data):
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(weight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8f9fa")

    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _render_latex_with_pdflatex(
    latex_str: str, output_path: str, dpi: int = 400
) -> bool:
    pdflatex_bin = _find_pdflatex()
    if not pdflatex_bin:
        return False

    tex_dir = os.path.dirname(pdflatex_bin)
    env = os.environ.copy()
    if tex_dir and tex_dir not in env.get("PATH", ""):
        env["PATH"] = f"{tex_dir}:{env.get('PATH', '')}"

    output_pdf = os.path.splitext(output_path)[0] + ".pdf"

    if r"\documentclass" in latex_str:
        full_doc = latex_str
    else:
        full_doc = rf"""\documentclass{{article}}
\usepackage[paperwidth=10in,paperheight=6in,margin=0.3in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{xcolor}}
\pagecolor{{white}}
\pagestyle{{empty}}
\begin{{document}}
{latex_str}
\end{{document}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "table.tex")
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(full_doc)

        proc = subprocess.run(
            [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
        )
        pdf_file = os.path.join(tmpdir, "table.pdf")
        if not os.path.isfile(pdf_file):
            print(f"pdflatex failed:\n{proc.stdout}\n{proc.stderr}")
            return False

        # Save intermediate PDF
        # try:
        #     shutil.copyfile(pdf_file, output_pdf)
        #     print(f"Intermediate PDF saved to: {output_pdf}")
        # except Exception as e:
        #     print(f"Could not copy intermediate PDF: {e}")

        rendered_png = os.path.join(tmpdir, "rendered.png")
        converted = False

        # Strategy 1: pypdfium2 (high quality vector rasterization)
        if not converted:
            try:
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(pdf_file)
                page = pdf[0]
                pil_image = page.render(scale=dpi / 72.0).to_pil()
                pil_image.save(rendered_png)
                converted = True
            except Exception:
                pass

        # Strategy 2: fitz / PyMuPDF
        if not converted:
            try:
                import fitz

                doc = fitz.open(pdf_file)
                page = doc[0]
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(rendered_png)
                converted = True
            except Exception:
                pass

        # Strategy 3: pdf2image
        if not converted:
            try:
                from pdf2image import convert_from_path

                images = convert_from_path(pdf_file, dpi=dpi)
                if images:
                    images[0].save(rendered_png)
                    converted = True
            except Exception:
                pass

        # Strategy 4: pdftoppm (poppler)
        if not converted:
            pdftoppm_bin = _find_binary("pdftoppm")
            if pdftoppm_bin:
                res = subprocess.run(
                    [
                        pdftoppm_bin,
                        "-png",
                        "-r",
                        str(dpi),
                        "-singlefile",
                        "table.pdf",
                        "rendered",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                )
                if res.returncode == 0 and os.path.isfile(rendered_png):
                    converted = True

        # Strategy 5: Ghostscript
        if not converted:
            gs_bin = _find_binary("gs")
            if gs_bin:
                res = subprocess.run(
                    [
                        gs_bin,
                        "-dNOPAUSE",
                        "-dBATCH",
                        "-sDEVICE=png16m",
                        f"-r{dpi}",
                        "-dTextAlphaBits=4",
                        "-dGraphicsAlphaBits=4",
                        "-sOutputFile=rendered.png",
                        "table.pdf",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                )
                if res.returncode == 0 and os.path.isfile(rendered_png):
                    converted = True

        # Strategy 6: ImageMagick
        if not converted:
            magick_bin = _find_binary("magick") or _find_binary("convert")
            if magick_bin:
                res = subprocess.run(
                    [
                        magick_bin,
                        "-density",
                        str(dpi),
                        "table.pdf",
                        "-quality",
                        "100",
                        rendered_png,
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                )
                if res.returncode == 0 and os.path.isfile(rendered_png):
                    converted = True

        # Strategy 7: sips (macOS native, scaled for target DPI)
        if not converted:
            sips_bin = _find_binary("sips") or "/usr/bin/sips"
            if os.path.isfile(sips_bin):
                # 10 inches at target dpi = target pixel width
                target_width = int(10.0 * dpi)
                res = subprocess.run(
                    [
                        sips_bin,
                        "-s",
                        "format",
                        "png",
                        "-s",
                        "dpiHeight",
                        f"{float(dpi)}",
                        "-s",
                        "dpiWidth",
                        f"{float(dpi)}",
                        "--resampleWidth",
                        str(target_width),
                        "table.pdf",
                        "--out",
                        rendered_png,
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                )
                if res.returncode == 0 and os.path.isfile(rendered_png):
                    converted = True

        if not converted:
            return False

        # Trim white margins with proper alpha compositing and high resolution
        try:
            from PIL import Image, ImageChops

            raw_img = Image.open(rendered_png)
            if raw_img.mode in ("RGBA", "LA") or (
                raw_img.mode == "P" and "transparency" in raw_img.info
            ):
                rgba = raw_img.convert("RGBA")
                white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white_bg, rgba).convert("RGB")
            else:
                img = raw_img.convert("RGB")

            bg = Image.new("RGB", img.size, (255, 255, 255))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()
            if bbox:
                pad = max(25, int(dpi * 0.08))
                bbox = (
                    max(0, bbox[0] - pad),
                    max(0, bbox[1] - pad),
                    min(img.width, bbox[2] + pad),
                    min(img.height, bbox[3] + pad),
                )
                img = img.crop(bbox)
            img.save(output_path, dpi=(dpi, dpi))
        except Exception as e:
            print(f"Warning: PIL processing failed ({e}), using raw converted PNG.")
            shutil.copyfile(rendered_png, output_path)

        return True


def render_latex_to_png(
    latex_or_df,
    output_path: str = "experiment_results.png",
    dpi: int = 400,
) -> str:
    """Render LaTeX table string or DataFrame to a PNG image file."""
    if isinstance(latex_or_df, pd.DataFrame):
        latex_str = format_to_latex(latex_or_df)
    else:
        latex_str = str(latex_or_df)

    success = _render_latex_with_pdflatex(latex_str, output_path, dpi=dpi)
    if not success:
        _render_table_matplotlib(latex_str, output_path, dpi=dpi)

    return output_path


def analyze_experiment(
    instance_experiments,
    output_png: str = "experiment_results.png",
    dpi: int = 400,
):
    results = []

    for instance, exp_folder in instance_experiments.items():
        instance_path = f"precalculated_instances/{instance}"

        # 1. Load Heuristic Solution
        heuristic_summary_path = f"{instance_path}/heuristic_summary.txt"
        heuristic_makespan = None
        if os.path.exists(heuristic_summary_path):
            with open(heuristic_summary_path, "r") as f:
                content = f.read()
                # Parse "Makespan: 2600"
                for part in content.split("|"):
                    if "Makespan" in part:
                        heuristic_makespan = float(part.split(":")[1].strip())
                        break

        if heuristic_makespan is None:
            print(f"Warning: Could not find heuristic makespan for {instance}")
            continue

        # 2. Load CP Data
        cp_files = sorted(glob.glob(f"{instance_path}/cp_intermediate_records_*.pkl"))
        cp_5m_bests = []
        cp_final_bests = []

        for fpath in cp_files:
            try:
                with open(fpath, "rb") as f:
                    records = pickle.load(f)

                if not records:
                    continue

                final_best = records[-1]["best"]
                cp_final_bests.append(final_best)

                # Interpolate at 300s
                best_at_300 = records[0]["best"]
                for r in records:
                    if r["time"] <= 300:
                        best_at_300 = r["best"]
                    else:
                        break
                cp_5m_bests.append(best_at_300)

            except (FileNotFoundError, pickle.UnpicklingError, IndexError):
                continue

        if not cp_final_bests:
            print(f"Warning: No CP records found for {instance}")
            continue

        cp_5m_avg = np.mean(cp_5m_bests)
        cp_best_known = np.min(cp_final_bests)

        # 3. Load LNS Data
        df_path = f"{exp_folder}/experiment_dataframe.pkl"
        if not os.path.exists(df_path):
            print(f"Warning: No experiment_dataframe.pkl found in {exp_folder}")
            continue

        df = pd.read_pickle(df_path)

        # Compute cumulative time per run
        df["cum_time"] = df.groupby("run_id")["total_iter_time"].cumsum()

        lns_5m_bests = []
        lns_final_bests = []
        lns_total_times = []

        # statuses: 3 is Optimal_New_Best, 6 is Feasible_New_Best
        new_best_count = len(df[df["statuses"].isin([3, 6])])
        total_iters = len(df)

        for run_id, group in df.groupby("run_id"):
            # Find best at 300s
            under_300 = group[group["cum_time"] <= 300]
            if not under_300.empty:
                best_at_300 = under_300.iloc[-1]["best"]
            else:
                best_at_300 = group.iloc[0]["best"]
            lns_5m_bests.append(best_at_300)

            final_best = group.iloc[-1]["best"]
            lns_final_bests.append(final_best)

            total_time = group["total_iter_time"].sum()
            lns_total_times.append(total_time)

        lns_5m_avg = np.mean(lns_5m_bests)
        lns_final_avg = np.mean(lns_final_bests)
        lns_time_avg = np.mean(lns_total_times)

        # 4. Calculate metrics
        lns_imp_5m = (heuristic_makespan - lns_5m_avg) / heuristic_makespan * 100
        cp_imp_5m = (heuristic_makespan - cp_5m_avg) / heuristic_makespan * 100
        lns_gap_cp = (lns_final_avg - cp_best_known) / cp_best_known * 100

        results.append(
            {
                "Instance": instance,
                "LNS 5m Impr. (%)": lns_imp_5m,
                "CP 5m Impr. (%)": cp_imp_5m,
                "LNS Gap to CP Best (%)": lns_gap_cp,
                "LNS 1000 iter Time (s)": lns_time_avg,
                "New Best Ratio (%)": (new_best_count / total_iters * 100)
                if total_iters > 0
                else 0.0,
            }
        )

    results_df = pd.DataFrame(results)

    # Calculate Averages and Display
    if not results_df.empty:
        print("\n--- Detailed Results ---")
        print(results_df.to_string(index=False, float_format="%.2f"))

        print("\n--- Average Results ---")
        averages = results_df.mean(numeric_only=True)
        print(averages.to_string(float_format="%.2f"))

        latex_str = format_to_latex(results_df)
        print("\n--- LaTeX Table ---")
        print(latex_str)

        if output_png:
            saved_png = render_latex_to_png(latex_str, output_png, dpi=dpi)
            print(f"\nRendered table saved to: {saved_png}")
    else:
        print("No valid data to report.")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze matheuristic experiments.")
    parser.add_argument(
        "instances",
        nargs="+",
        type=str,
        help="Name of the precalculated instance folder under precalculated_instances/",
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Sample experiment to be used to find the target config. The latest experiment with this config will be found in all of the selected instances.",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="experiment_results.png",
        help="Path to save the rendered table PNG.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI resolution for rendered table PNG (default: 400).",
    )
    args = parser.parse_args()

    instance_names = [
        i.split("/")[1] if "precalculated_instances" in i else i for i in args.instances
    ]
    exp_folders = find_experiment_folders(instance_names, args.experiment)
    analyze_experiment(exp_folders, output_png=args.output_png, dpi=args.dpi)
