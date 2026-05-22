import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.datasets import load_diabetes

    return load_diabetes, mo, pd


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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
