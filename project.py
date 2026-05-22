import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_diabetes

    return load_diabetes, mo, pd, plt, sns


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
