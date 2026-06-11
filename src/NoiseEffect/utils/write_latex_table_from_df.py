import pandas as pd


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str = "lc",
    resize: bool = True,
    force_placement: bool = True,
    clean_headers: bool = True,
    **kwargs,
):
    """
    Converts a DataFrame to a LaTeX table with standard styling adjustments.
    Adds the resiziing to fit the page and forces table placement [H]
    """

    df_copy = df.copy()

    if clean_headers:
        # Replaces underscores with spaces and capitalizes (e.g., 'gene_name' -> 'Gene Name')
        df_copy.columns = [
            str(col).replace("_", " ").title() for col in df_copy.columns
        ]

    # Generate base LaTeX string
    latex_str = df_copy.to_latex(
        index=False, caption=caption, label=label, column_format=column_format, **kwargs
    )

    # Inject \resizebox
    if resize:
        latex_str = latex_str.replace(
            "\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}"
        )
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}%\n}")

    # Inject [H] placement
    if force_placement:
        latex_str = latex_str.replace("\\begin{table}", "\\begin{table}[H]")

    return latex_str


def save_latex_tables(file_path: str, tables: str | list[str]):
    """
    Writes one or more LaTeX table strings to a .tex file.
    """
    if isinstance(tables, str):
        tables = [tables]

    with open(file_path, "w") as f:
        f.write("\n\n".join(tables))
