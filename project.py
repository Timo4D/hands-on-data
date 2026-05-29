import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes

    from scipy import stats

    return load_diabetes, mo, np, pd, plt, sns, stats


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
    plt.title("Comparison of diabetes feature distributions")
    plt.show()
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The box plots are created to explore and compare the distributions of all features before any preprocessing. It is helpful for many reasons like detecting Scale Differences, Identifying Outliers and understanding Data Quality.

    Especially before standardization it helps us to understand why and if standardization is helpful or needed. We can see from this Boxplot that features operate on completely different incomparable scales. After standardization it helps us to confirm that standardization worked, and lets us compare distributions fairly. The shape of each box stays the same, only the axis scale changes.
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
    return (df_all_numeric,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What we can learn from the plots:
    - Plot 1
        - Features operate on vastly different scales, standardisation is needed.
        - TC, LDL and Progression have long tails, suggesting some outliers.
    - Plot 2
        - Both Female and Male patients have similar age distributions, females are a bit younger but nothing to worry about.

    - Plot 3
        - LDL and TC have a strong positive correlation
        - HDL and TCH have a strong negative correlation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Project task 3: Testing for pair-wise associations between features
    ## Question 1: Are there features (including Age) in the dataset which differ significantly between women and men?

    This is important to test for multiple reasons.
    - If certain features differ strongly between women and men, then any downstream analysis might be driven by gender rather than the feature itself.
    - If a predictive model is trained without knowing which features are gender-dependent, it may learn gender-specific patterns implicitly, leading to biased or poorly generalizable predictions.

    To test for significant differences we will either use __t-tests__ for normally distributed data or __Mann-Whitney U test__ for non-normally distributed data.
    To test for normality, the __Shapiro–Wilk test__ is used.
    To avoid false positives, a __FDR correction__ is applied
    """)
    return


@app.cell
def _(df_all_numeric, df_diabetes_metadata, pd, stats):
    #Step 1: Check for Normality

    # Split by Gender
    females = df_all_numeric[df_diabetes_metadata["Gender"] == "Female"]
    males   = df_all_numeric[df_diabetes_metadata["Gender"] == "Male"]

    # Test normality within each group
    _rows = []
    for _feat in df_all_numeric.columns:
        _, _p_f = stats.shapiro(females[_feat])
        _, _p_m = stats.shapiro(males[_feat])
        _rows.append({
            "Feature":          _feat,
            "p-value (Female)": round(_p_f, 4),
            "p-value (Male)":   round(_p_m, 4),
            "Normal (Female)":  _p_f > 0.05,
            "Normal (Male)":    _p_m > 0.05,
            "Use t-test":       (_p_f > 0.05) and (_p_m > 0.05)
        })

    normality_by_gender = pd.DataFrame(_rows)
    normality_by_gender
    return females, males, normality_by_gender


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TC and Glu are the only features where both Male and Female groups show normal distribution, so they are tested using __t-tests__

    All other features will use __Mann-Whitney U test__.
    """)
    return


@app.cell
def _(females, males, normality_by_gender, pd, stats):
    from statsmodels.stats.multitest import multipletests

    # Use normality_by_gender to automatically pick the right test per feature
    _test_results = []
    for _, _row in normality_by_gender.iterrows():
        _feat = _row["Feature"]
        if _row["Use t-test"]:
            # t-test — does NOT assume equal variances between groups
            _stat, _p = stats.ttest_ind(females[_feat], males[_feat], equal_var=False)
            _test = "t-test"
        else:
            # Mann-Whitney U — non-parametric, no normality assumption
            _stat, _p = stats.mannwhitneyu(females[_feat], males[_feat], alternative='two-sided')
            _test = "Mann-Whitney U"
        _test_results.append({
            "Feature":    _feat,
            "Test used":  _test,
            "Statistic":  round(_stat, 3),
            "p-value (raw)": _p,
        })

    results_gender = pd.DataFrame(_test_results)

    # Benjamini-Hochberg FDR correction across all 10 features
    _reject, _p_corr, _, _ = multipletests(results_gender["p-value (raw)"], method='fdr_bh')
    results_gender["p-corrected (BH)"] = _p_corr.round(4)
    results_gender["Significant"]       = _reject

    results_gender = results_gender.sort_values("p-corrected (BH)").reset_index(drop=True)
    results_gender
    return (multipletests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    8 out of the 10 Features show significant differences between Genders. Only TC and Progression are not significant. BMI is on the verge of being not significant.

    - Females have higher HDL
    - Females have lower TCH
    - Females have lower BP
    - Females have lower Glu
    - Females are younger
    - Females have lower LTG
    - Females have lower LDL
    - Females have marginally lower BMI
    - TC shows no sig. difference
    - Disease progression is not gender biased
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Are there features (including Age) in the dataset which have significant correlation between them?

    To test pairwise correlation, we first need to check whether each feature is normally distributed on the full dataset. If they are normally distirbuted we will use __Pearson__ otherwise __Spearman__ will do.
    """)
    return


@app.cell
def _(df_all_numeric, pd, stats):
    p_values = df_all_numeric.apply(lambda x: stats.shapiro(x)[1])

    p_value_results_df = pd.DataFrame({
        'p-Value': round(p_values, 4),
        'Is Normal': p_values > 0.05
    })

    p_value_results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    None of the features are Normally distributed, this means we will use __Spearman__ for all.
    """)
    return


@app.cell
def _(df_all_numeric, multipletests, pd, stats):
    from itertools import combinations

    # Get all possible feature combinations
    _features = df_all_numeric.columns.tolist()
    _pairs = list(combinations(_features, 2))

    _corr_rows = []
    for _f1, _f2 in _pairs:
        _rho, _p = stats.spearmanr(df_all_numeric[_f1], df_all_numeric[_f2])
        _corr_rows.append({
            "Feature 1":      _f1,
            "Feature 2":      _f2,
            "Spearman ρ":     round(_rho, 3),
            "p-value (raw)":  _p,
        })

    results_corr = pd.DataFrame(_corr_rows)

    # Benjamini-Hochberg FDR correction across all 45 pairs
    _reject, _p_corr, _, _ = multipletests(results_corr["p-value (raw)"], method="fdr_bh")
    results_corr["p-corrected (BH)"] = _p_corr.round(4)
    results_corr["Significant"]       = _reject

    results_corr = results_corr.sort_values("p-corrected (BH)").reset_index(drop=True)
    results_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Project Task 4: Evaluating if there are clusters in the data using hierarchical clustering

    ##Applying a standard scaler.
    """)
    return


@app.cell
def _(df_diabetes_features):
    #4.1 Copy DataFrame and drop Progression Feature
    df_diabetes_basal_features = df_diabetes_features.drop(columns=['Progression'])
    return (df_diabetes_basal_features,)


@app.cell
def _(df_diabetes_basal_features, pd):
    #4.2 Usin hierarchical cluster analysis to assess if there are clear clusters in the data.
    #We need to scale the data before running the cluster analysis
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    df_diabetes_scaled = pd.DataFrame(
        scaler.fit_transform(df_diabetes_basal_features),
        columns=df_diabetes_basal_features.columns,
        index=df_diabetes_basal_features.index
    )
    df_diabetes_scaled.describe().loc[['mean','std']].round(5)

    #Transforms the values of the features such taht each feature will have a mean of 0 and a standard deviation of 1
    return StandardScaler, df_diabetes_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating the Cluster Analysis
    """)
    return


@app.cell
def _(df_diabetes_scaled, plt):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    methods = ['ward', 'average', 'complete', 'single']

    plt.figure(figsize=(16, 10))

    for i, method in enumerate(methods, 1):
        plt.subplot(2, 2, i)
        plt.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)')


        Z = linkage(df_diabetes_scaled, method=method, metric='euclidean')

        dendrogram(
            Z,
            truncate_mode='level', 
            p=5,                   # Show only the last 5 levels of merging
            show_contracted=True
        )
        plt.xlabel('Cluster Size / Sample Index')
        plt.ylabel('Distance')

    plt.tight_layout()
    plt.show()
    return fcluster, linkage


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TODO: Interpretation von den results oben.

    ## fcluster implementation
    """)
    return


@app.cell
def _(
    df_diabetes_basal_features,
    df_diabetes_metadata,
    df_diabetes_raw,
    df_diabetes_scaled,
    fcluster,
    linkage,
    pd,
    stats,
):
    Z_ward = linkage(df_diabetes_scaled, method='ward')
    cluster_labels = fcluster(Z_ward, t=2, criterion='maxclust')

    df_clusters = pd.DataFrame({'Cluster': cluster_labels}, index=df_diabetes_basal_features.index)

    #Becuase Metadata and Progressoin are in other dataframes we will need to merge them to analyze them.
    df_analysis = df_clusters.join([df_diabetes_metadata, df_diabetes_raw['Progression']])

    #Categorize high and low progression based on the median
    median_progression = df_analysis['Progression'].median()
    df_analysis['High_Progression'] = (df_analysis['Progression'] > median_progression)

    median_age = df_analysis['Age'].median()
    df_analysis['High_Age'] = (df_analysis['Age'] > median_age).astype(int)

    features_to_test = ['Gender', 'High_Progression', 'High_Age']

    #TODO: This is overkill
    for feature in features_to_test:
        ct = pd.crosstab(df_analysis['Cluster'], df_analysis[feature])

        # Chi-Quadrat-Test
        chi2, p, dof, expected = stats.chi2_contingency(ct)

        print(f"--- Assoziation zwischen Cluster und {feature} ---")
        print(ct)
        print(f"Chi-Quadrat-Statistik: {chi2:.2f}, p-Wert: {p:.4e}\n")
    return (df_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Task 5: Exploring multidimensional pattersns with PCA

    ## Log-Transforming the data and Standard Scaling
    """)
    return


@app.cell
def _(df_diabetes_basal_features):
    # Test if all values > 0
    print((df_diabetes_basal_features > 0 ).all().all())
    return


@app.cell
def _(StandardScaler, df_diabetes_basal_features, np, pd):
    #Becuase all values are above 0, we will apply log-transformation
    df_diabetes_basal_features_log = np.log(df_diabetes_basal_features)

    log_scaler = StandardScaler()
    scaled_log_features = log_scaler.fit_transform(df_diabetes_basal_features_log)

    df_scaled_log_features = pd.DataFrame(scaled_log_features, columns=df_diabetes_basal_features.columns, index=df_diabetes_basal_features.index)

    df_scaled_log_features
    return (df_scaled_log_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying PCA
    """)
    return


@app.cell
def _(df_scaled_log_features, plt):
    from sklearn.decomposition import PCA

    pca = PCA()
    principal_components = pca.fit_transform(df_scaled_log_features)
    explained_variance = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)
    axes[0].set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
    axes[0].set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
    axes[0].set_title('PCA: PC1 vs PC2')

    axes[1].scatter(principal_components[:, 2], principal_components[:, 3], alpha=0.7, edgecolors='w', s=60, color='coral')
    axes[1].set_xlabel(f'Principal Component 3 ({explained_variance[2]:.2f}%)')
    axes[1].set_ylabel(f'Principal Component 4 ({explained_variance[3]:.2f}%)')
    axes[1].set_title('PCA: PC3 vs PC4')
    return explained_variance, principal_components


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the Results

    - PC1 vs PC2 Plot: The first two principal components capture the majority of the dataset's variance (~56.08% combined). The scatter plot reveals a fairly continuous, dense cloud of data points without any heavily isolated or distinct clusters. This suggests that the basal characteristics of the patients vary along a continuous spectrum rather than forming subgroups. PC1 alone handles over 41% of the variability, making it the most significant axis describing the primary underlying trends across the patients' metrics.

    - PC3 vs PC4 Plot: These components capture secondary and more subtle feature variations, contributing an additional ~21.12% to the explained variance. The plot appears more concentrated in the center, acting almost like a circular "halo" of noise. This indicates that while these components carry meaningful biological information, they represent more nuanced, idiosyncratic patient differences rather than sweeping population-level trends.

    - Overall Dimensionality: Together, the first 4 Principal Components account for 77.19% of the total variance in the original dataset. This means you can reduce the dataset from 10 dimensions down to just 4 while retaining more than three-quarters of the underlying information.



    ## Evaluating and visualizing the loading of the original features on PC1 and PC2.
    TODO

    ## Additional plots
    """)
    return


@app.cell(hide_code=True)
def _(df_analysis, explained_variance, plt, principal_components, sns):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Color Coded by Gender")

        # Optional: Farben festlegen (als Dictionary)
        custom_palette = {'Female': 'coral', 'Male': 'steelblue'}

        # ==========================================
        # Erster Plot (PC1 vs PC2)
        # ==========================================
        sns.scatterplot(
            x=principal_components[:, 0], 
            y=principal_components[:, 1], 
            hue=df_analysis.Gender,
            palette=custom_palette,
            alpha=0.7,
            ax=axes[0]
        )

        axes[0].set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
        axes[0].set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
        axes[0].set_title('PCA: PC1 vs PC2')

        sns.scatterplot(
            x=principal_components[:, 2], 
            y=principal_components[:, 3], 
            hue=df_analysis.Gender, 
            palette=custom_palette,
            alpha=0.7, 
            ax=axes[1]
        )

        axes[1].set_xlabel(f'Principal Component 3 ({explained_variance[2]:.2f}%)')
        axes[1].set_ylabel(f'Principal Component 4 ({explained_variance[3]:.2f}%)')
        return axes[1].set_title('PCA: PC3 vs PC4')


    _()
    return


@app.cell(hide_code=True)
def _(df_analysis, explained_variance, plt, principal_components, sns):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Color Coded by High Progression")

        # Optional: Farben festlegen (als Dictionary)
        custom_palette = {True: 'coral', False: 'steelblue'}

        # ==========================================
        # Erster Plot (PC1 vs PC2)
        # ==========================================
        sns.scatterplot(
            x=principal_components[:, 0], 
            y=principal_components[:, 1], 
            hue=df_analysis.High_Progression,
            palette=custom_palette,
            alpha=0.7,
            ax=axes[0]
        )

        axes[0].set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
        axes[0].set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
        axes[0].set_title('PCA: PC1 vs PC2')

        sns.scatterplot(
            x=principal_components[:, 2], 
            y=principal_components[:, 3], 
            hue=df_analysis.High_Progression, 
            palette=custom_palette,
            alpha=0.7, 
            ax=axes[1]
        )

        axes[1].set_xlabel(f'Principal Component 3 ({explained_variance[2]:.2f}%)')
        axes[1].set_ylabel(f'Principal Component 4 ({explained_variance[3]:.2f}%)')
        return axes[1].set_title('PCA: PC3 vs PC4')


    _()
    return


@app.cell(hide_code=True)
def _(df_analysis, explained_variance, plt, principal_components, sns):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Color Coded by Age")

        # Optional: Farben festlegen (als Dictionary)
        custom_palette = {True: 'coral', False: 'steelblue'}

        # ==========================================
        # Erster Plot (PC1 vs PC2)
        # ==========================================
        sns.scatterplot(
            x=principal_components[:, 0], 
            y=principal_components[:, 1], 
            hue=df_analysis.High_Age,
            palette=custom_palette,
            alpha=0.7,
            ax=axes[0]
        )

        axes[0].set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
        axes[0].set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
        axes[0].set_title('PCA: PC1 vs PC2')

        sns.scatterplot(
            x=principal_components[:, 2], 
            y=principal_components[:, 3], 
            hue=df_analysis.High_Age, 
            palette=custom_palette,
            alpha=0.7, 
            ax=axes[1]
        )

        axes[1].set_xlabel(f'Principal Component 3 ({explained_variance[2]:.2f}%)')
        axes[1].set_ylabel(f'Principal Component 4 ({explained_variance[3]:.2f}%)')
        return axes[1].set_title('PCA: PC3 vs PC4')


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretatoin
    #### Color Coding by Gender
    When collor-codeing the scatterplots by Gender both groups look very similar with a very slight tendency for Males to have a higher PC1. PC3 and PC4 show no clear difference between the groups.

    #### Color Coding by Progression
    The PC1 and PC2 plot shows a clear difference between High and Low progression. Having a High Progression tends to scew the data to a high PC1 and a lower PC2. PC3 and PC4 show no clear difference.

    #### Color Coding by Progression
    Not having a high age tends to lead to a lower PC1 with no clear difference in PC2, PC3 and PC4


    # Project task 6: Applying multiple linear regression.
    """)
    return


@app.cell
def _(StandardScaler, df_diabetes_basal_features, df_diabetes_features, pd):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    def _():
        # Partitioning the data to train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            df_diabetes_basal_features,
            df_diabetes_features['Progression'],
            test_size=0.2,
            random_state=42
        )

        # Fitting the scaler only on the training data to prevent data leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Performing the linear regression analysis
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        r2_train = r2_score(y_train, y_train_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        print("--- Model Evaluation ---")
        print(f"Train R-squared: {r2_train:.4f} | Train MSE: {mse_train:.4f}")
        print(f"Test R-squared:  {r2_test:.4f} | Test MSE:  {mse_test:.4f}\n")

        coef_df = pd.DataFrame({'Feature': df_diabetes_basal_features.columns, 'Coefficient': model.coef_})
        coef_df['Absolute_Importance'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Absolute_Importance', ascending=False)

        return coef_df

    _()

    coef_df = _()
    return coef_df, train_test_split


@app.cell
def _(coef_df, plt, sns):
    # Assessing the importance of the features for the prediction
    def _():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Absolute_Importance', y='Feature', data=coef_df, palette='viridis')
        plt.title('Feature Importance (Absolute Standardized Coefficients)', fontsize=15, pad=15)
        plt.xlabel('Absolute Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)

        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 0.5, p.get_y() + p.get_height() / 2. + 0.05,
                     f'{width:.2f}', ha="left", va="center")

        plt.tight_layout()
        plt.show()
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation
    TC, LTG, BMI and LDL are the most substantial drivers of disease progression in this model. Suprisingly, Glu has very little imact on the progression of the disease compared to physical factors like BMI

    # Project Task 7: Applying logistic regression
    """)
    return


@app.cell
def _(
    StandardScaler,
    coef_df,
    df_diabetes_basal_features,
    df_diabetes_features,
    plt,
    train_test_split,
):
    def _():
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve, auc

        # Creating Progression_class feautre
        median_progression = df_diabetes_features['Progression'].median()
        y = (df_diabetes_features['Progression'] > median_progression).astype(int)

        # Partitioning the data
        X_train, X_test, y_train, y_test = train_test_split(
            df_diabetes_basal_features,
            y,
            test_size=0.2,
            random_state=42
        )

        # Performing Data Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Performing hte logistic regression with elastic net regularization
        model = LogisticRegression(penalty='elasticnet', solver='saga')
        model.fit(X_train_scaled, y_train)
    
        # ROC AUC
        y_probs = model.predict_proba(X_test_scaled)[:, 1] # Get probabilities for class 1
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Feature Importance
        return coef_df

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interpretation
    """)
    return


if __name__ == "__main__":
    app.run()
