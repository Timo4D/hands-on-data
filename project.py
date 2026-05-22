import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import itertools
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes

    from statsmodels.stats.multitest import multipletests

    from scipy import stats

    return load_diabetes, mo, multipletests, pd, plt, sns, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Project task 1: Data import and organization

    Import data, transform it to make it more readable and save it.
    """)
    return


@app.cell
def _(load_diabetes, pd):
    # Step 1: Loading the dataset
    diabetes = load_diabetes(scaled=False)
    df_diabetes_raw = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df_diabetes_raw['target'] = diabetes.target

    # Step 2: Rename columns to meaningful labels.
    rename_map = {
        "age":    "Age",
        "sex":    "Gender",
        "bmi":    "BMI",
        "bp":     "BP",
        "s1":     "TC",
        "s2":     "LDL",
        "s3":     "HDL",
        "s4":     "TCH",
        "s5":     "LTG",
        "s6":     "Glu",
        "target": "Progression",
    }
    df_diabetes_raw = df_diabetes_raw.rename(columns=rename_map)

    # Step 3: Replace the default integer index.
    df_diabetes_raw.index = [f"Patient_ID_{x:03}" for x in range(len(df_diabetes_raw))]

    # Step 4: Re-encoding the values in the Gender column.
    df_diabetes_raw["Gender"] = df_diabetes_raw["Gender"].astype("string")
    df_diabetes_raw["Gender"] = df_diabetes_raw["Gender"].replace({
        "1.0":"Female",
        "2.0":"Male"
    })
    df_diabetes_raw
    return (df_diabetes_raw,)


@app.cell
def _(df_diabetes_raw):
    #Step 5: Separate the df_diabetes_raw data frame to two data frames
    df_diabetes_metadata = df_diabetes_raw[["Age", "Gender"]]
    df_diabetes_features = df_diabetes_raw[['BMI', 'BP', 'TC', 'LDL', 'HDL',
    'TCH', 'LTG', 'Glu','Progression']]

    #Step 6: Save the two data frames as Excel files
    df_diabetes_features.to_excel("df_diabetes_features.xlsx")
    df_diabetes_metadata.to_excel("Table_diabetes_metadata.xlsx")
    return df_diabetes_features, df_diabetes_metadata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Project task 2: Visualizing the data
    """)
    return


@app.cell
def _(df_diabetes_features, plt, sns):
    #Step 1: Make box-plots to visualize and compare the distribution of the raw (!) values of all the features in df_diabetes_features

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_diabetes_features, showfliers=False, width=0.6)
    sns.stripplot(data=df_diabetes_features, alpha=0.8, legend=False, size=2)
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.title("Coparison of diabetes feature distibutions")
    plt.show()
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The box plots are created to explore and compare the distributions of all features before any preprocessing. It is helpful fore many reasons like detecting Scale Differences, Identifying Outliers and understanding Data Quality.

    Especially before standardization it helps us to understand why and if standardization is helpful or needed. We can see from this Boxplot that featres operate on completely different incomparable scales. After standardization it helps us to confirm that standardization worked, and lets us compare distibutions fairly. The hape of each box stays the same, only teh axis scale changes.
    """)
    return


@app.cell
def _(df_diabetes_metadata, plt, sns):
    #Step 2: Box-plot comparing the Age of women and men participating in the dataset.
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_diabetes_metadata, x="Gender", y="Age", showfliers=False, width=0.5)
    sns.stripplot(data=df_diabetes_metadata, x="Gender", y="Age", alpha=0.5, size=3, legend=False)
    plt.title("Age distribution by Gender")
    plt.show()
    plt.close()
    return


@app.cell
def _(df_diabetes_features, df_diabetes_metadata, pd, plt, sns):
    #Step 3: Scatter Plots for each possible 2-feature pair
    df_all_numeric = pd.concat([df_diabetes_metadata[["Age"]], df_diabetes_features], axis=1)
    g = sns.pairplot(df_all_numeric, plot_kws={"alpha": 0.3, "s": 10}, diag_kind="kde")
    g.fig.suptitle("Scatter plots for all feature pairs", y=1.02)
    plt.show()
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What we can learn from the plots:
    - Plot 1
        - Features operatoe on vastly different scales, standardisation is needed.
        - TD, LDL and Progession have long tails, suggestion some outliers.
    - Plot 2
        - Both Female and Male patients have similar age distibutions, females are a bit younger but nothing to worry about.

    - Plot 3
        - LDL and TC have a strong positive correlation
        - HDL and TCH have a strong negative correlation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Project task 3: Testing for pair-wise associations between features
    ### Are there features (including Age) in the dataset which differ significantly between women and men?
    """)
    return


@app.cell
def _(df_diabetes_features, df_diabetes_metadata, multipletests, pd, stats):
    df_all = pd.concat([df_diabetes_metadata, df_diabetes_features], axis=1)
    features = ["Age", "BMI", "BP", "TC", "LDL", "HDL", "TCH", "LTG", "Glu", "Progression"]

    female = df_all[df_all["Gender"] == "Female"]
    male = df_all[df_all["Gender"] == "Male"]

    rows = []
    for feat in features:
        f_vals = female[feat].dropna()
        m_vals = male[feat].dropna()

        # Normality check (Shapiro-Wilk) for each group
        _, p_norm_f = stats.shapiro(f_vals)
        _, p_norm_m = stats.shapiro(m_vals)
        both_normal = p_norm_f > 0.05 and p_norm_m > 0.05

        # Choose parametric (t-test) or non-parametric (Mann-Whitney U)
        if both_normal:
            _, p_raw = stats.ttest_ind(f_vals, m_vals)
            test = "t-test"
        else:
            _, p_raw = stats.mannwhitneyu(f_vals, m_vals)
            test = "Mann-Whitney U"

        rows.append({"Feature": feat, "Test": test, "p (raw)": round(p_raw, 4)})

    df_gender_tests = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg) for multiple comparisons
    _, p_fdr, _, _ = multipletests(df_gender_tests["p (raw)"], method="fdr_bh")
    df_gender_tests["p (FDR)"] = p_fdr.round(4)
    df_gender_tests["Significant"] = df_gender_tests["p (FDR)"] < 0.05

    df_gender_tests
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are 8 features with significant differences: Age, BMI, BP, LDL, HDL, TCH, LTG, Glu
    and 2 features without significant differences: TC, Progression

    ### Are there features in the dataset which have significant correlation between them?
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
