# Shiny App

Graphical interface for computing aggregation functions on tabular data and evaluating binary classification.

*Wersja polska: [Polski](https://wgalka.github.io/Means/pl/shiny/)*

## Getting Started

Install the library from the repository root:

```powershell
python -m pip install .
```

Launch the application with:

```powershell
agglib start
```

If the environment is missing packages required by the interface (`shiny`, `pandas`, `matplotlib`, `scikit-learn`, `pygments`), the CLI will ask whether to install them:

```text
Install the graphical interface? [Y/n]
```

You can also install all dependencies upfront:

```powershell
python -m pip install ".[app]"
```

The `agglib serve` command works identically to `agglib start`. Press `Ctrl+C` to stop the server.

> **Test mode:** Appending `?tester=1` to the browser URL (e.g. `http://127.0.0.1:8000/?tester=1`) automatically loads the sample dataset `example.csv` along with ground truth labels.

## Working with Data

1. **Upload a file** – upload a CSV file and choose the column separator and decimal separator. Each row is treated as a separate vector to aggregate.
2. **Column selection** – in the data preview, use the checkboxes in column headers to select which columns to include in calculations (text columns such as identifiers should be unchecked). Clicking a column name sorts the table preview.

By default, the preview shows the first 5 and last 5 rows (`5 … 5`), with an option to switch to the full table (`All rows`, capped at 2,000 rows in the browser view). Sorting and truncating the preview do not affect the row order in calculations or exports.

Uploaded CSV files are validated before loading (up to 20 MB and 100,000 rows), and each task checks estimated memory usage and method complexity before entering the queue. Aggregations run asynchronously in a bounded background queue with a live progress bar.

## Aggregations and Weights

All 21 aggregation functions from `aggregationslib.ndarray` are available, including the Choquet (`A_ch`) and Sugeno (`A_sg`) fuzzy integrals with six types of measures:

- **weighted** and **k-fuzzy** – weight inputs appear directly above the selected columns (for the weighted measure, weights must sum to 1).
- **owa** – weights are assigned to positions after sorting values from largest to smallest (weights must sum to 1).
- **exponential_cardinality** – takes a single parameter $q > 0$.
- **uniform** and **relative** – measures that do not require additional weights.

Because measure constructors generate subsets, the `weighted`, `owa`, and `k-fuzzy` variants are limited to 12 selected columns in the interface. Input data are not automatically normalized, and missing values (`NaN`) or infinite values in selected columns are reported as errors.

At the bottom of the form, you can download the resulting CSV file (containing the original columns plus the computed result) as well as ready-to-run, standalone Python code reproducing the calculations.

## Classification Evaluation

In the *Ground truth labels* section, you can provide true class labels (by pasting them as text or uploading a CSV file) to evaluate binary classification based on the aggregation scores:

- Labels must match the data rows in order (including manually appended vectors) and contain exactly two classes.
- Specify the positive class label (default `1`) and choose whether a higher aggregation score indicates the positive or negative class.

Once labels are provided, the application computes:
- **ROC curve** together with the area under the curve (ROC AUC),
- **Optimal cutoff threshold ($t$)** according to the selected criterion: Youden's index ($J = \text{TPR} - \text{FPR}$), Accuracy, or F1 score. The threshold can also be set manually in *Manual* mode,
- **Metrics summary**: Average Precision, Accuracy, Balanced Accuracy, Precision, Recall, Specificity, NPV, F1 (for the positive class as well as micro/macro/weighted variants), MCC, and Cohen's $\kappa$ (metric names in the table link to `scikit-learn` documentation),
- **Confusion matrices** displayed as heatmaps (raw counts and row-normalized percentages per true class).
