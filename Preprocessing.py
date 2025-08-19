import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Drug list (spelling hygiene + variants)
drugs_of_interest = [
    "BUPROPRION",
    "WELLBUTRIN", "ZYBAN", "FORVIVO",
    "VARENICLINE", "CHANTIX", "CHAMPIX", "NICOTINE", "TYRVAYA"
]

PLOTS_DIR = "/home/max/DrugUtil/results/plots"
DATA_ROOT = "/home/max/DrugUtil/Data/"

REQUIRED_YEARS = None #set(range(2008, 2019))  # Includes 2008 through 2018
NORMALIZE_BY_POPULATION = False  # Whether to normalize prescription counts by state population


def read_data(path_to_csvs: str, load = True) -> pd.DataFrame:
    """
    Read and merge drug utilization CSVs with CDC tax data.
    Returns a tidy table with: Year, State, Number of Prescriptions, percent_tax, dollar_tax.
    Caches the merged result as StateTaxDrugUtilizationData.csv in the same folder.
    """
    super_file = os.path.join(path_to_csvs, "StateTaxDrugUtilizationData.csv")
    if os.path.exists(super_file) and load:
         df = pd.read_csv(super_file, low_memory=False)
         print("Loaded cached data from", super_file)
         print(f"Data shape: {df.shape}")
         return df

    # --- Load & stack drug CSVs ---
    files = [f for f in os.listdir(path_to_csvs) if f.endswith(".csv")]
    dfs = []
    for f in tqdm(files, desc="Processing"):
        year_path = os.path.join(path_to_csvs, f)
        year_df = pd.read_csv(year_path, low_memory=False)
        try:
            year = int(f.split(".")[0][-4:])
            year_df["Year"] = year  # set once, as int
            dfs.append(year_df)
        except ValueError as e:
            print(f"Skipping {f}: could not extract year from filename ({e})")

    if not dfs:
        raise FileNotFoundError(f"No source CSVs found in {path_to_csvs}")

    df = pd.concat(dfs, ignore_index=True)

    # --- Filter to drugs of interest (escape names for safety) ---
    pattern = "|".join([s for s in drugs_of_interest])
    mask = df["Product Name"].astype(str).str.contains(pattern, case=False, na=False)
    df_drug = df.loc[mask].copy()  # <-- make an explicit copy to avoid SettingWithCopy
    matching_drugs = [
        [drug for drug in drugs_of_interest if drug.lower() in product_name.lower()]
        for product_name in df_drug["Product Name"]
    ]
    matching_drugs = [m[0] for m in matching_drugs]
    # print(f"{matching_drugs=} {len(matching_drugs)=}")
    df_drug['Product Name'] = matching_drugs

    
    # --- Load CDC / tax data ---
    tax_path = os.path.join(path_to_csvs, "cdc_data.csv")
    if not os.path.exists(tax_path):
        raise FileNotFoundError(f"Missing cdc_data.csv at {tax_path}")
    df_tax = pd.read_csv(tax_path, low_memory=False)
    

    # --- Build percent_tax / dollar_tax from Data_Value / Data_Value_Unit ---
    df_tax["Data_Value"] = pd.to_numeric(df_tax["Data_Value"], errors="raise")
    tpp = df_tax['SubMeasureDesc'].str.contains("Federal and State Tax per pack", case=False, na=False)
    percentage = df_tax['SubMeasureDesc'].str.contains("Federal and State tax as a Percentage of Retail Price", case=False, na=False)
    # Print unique SubMeasureDesc values
    # print("Unique SubMeasureDesc values:")
    # for desc in sorted(df_tax['SubMeasureDesc'].unique()):
    #     print(f"  - {desc}")

    df_tax = df_tax[tpp | percentage].copy()  # keep only relevant rows

    unit = df_tax["Data_Value_Unit"].astype(str).str.strip()
    df_tax["percent_tax"] = np.where(unit.str.contains("%", na=False), df_tax["Data_Value"], np.nan)
    df_tax["dollar_tax"]  = np.where(unit.str.contains(r"\$", na=False), df_tax["Data_Value"], np.nan)

    # --- Type hygiene for join keys ---
    df_drug["Year"] = pd.to_numeric(df_drug["Year"], errors="raise").astype("Int64")
    df_drug["Number of Prescriptions"] = pd.to_numeric(df_drug["Number of Prescriptions"], errors="raise")
    df_tax["Year"]  = pd.to_numeric(df_tax["Year"], errors="raise").astype("Int64")

    # --- Collapse drug data to one row per (Year, State) ---
    drug_min = (
        df_drug
        .groupby(["Year", "State", "Product Name"], as_index=False)
        .agg(Number_of_Prescriptions=("Number of Prescriptions", "sum"))
    )

    # --- Collapse tax data to one row per (Year, State) ---
    df_tax = df_tax.sort_values(["Year", "LocationAbbr", "SubMeasureIdDisplayOrder"])

    tax_min = (
        df_tax
        .groupby(["Year", "LocationAbbr"], as_index=False)
        .agg(
            percent_tax=("percent_tax", "mean"),
            dollar_tax =("dollar_tax",  "mean"),
        )
        .rename(columns={"LocationAbbr": "State"})
    )

    check = drug_min.groupby(["Year","State","Product Name"]).size().reset_index(name="n")
    assert (check["n"]==1).all(), "Drug data not 1:1 after aggregation"

    # one row per Year, State for tax
    check = tax_min.groupby(["Year","State"]).size().reset_index(name="n")
    assert (check["n"]==1).all(), "Tax data not 1:1 after aggregation"

    # --- Merge & tidy ---
    final_df = (
        drug_min
        .merge(tax_min, on=["Year", "State"], how="inner")
        .loc[:, ["Year", "State", "Product Name", "Number_of_Prescriptions", "percent_tax", "dollar_tax"]]
        .rename(columns={"Number_of_Prescriptions": "Number of Prescriptions"})
        .dropna(subset=["percent_tax", "dollar_tax"], how="all")
        .assign(Year=lambda d: d["Year"].astype("Int64"))
        .sort_values(["State", "Year"])
        .reset_index(drop=True)
    )

    if REQUIRED_YEARS is not None:
        final_df = final_df.groupby(['State', 'Product Name']).filter(
            lambda group: REQUIRED_YEARS.issubset(set(group['Year']))
        )

    if NORMALIZE_BY_POPULATION:
        pop_path = os.path.join(path_to_csvs, "state-population.csv")
        df_population = pd.read_csv(pop_path, low_memory=False)

        # keep only total-pop rows; standardize names
        pop_df = df_population.rename(columns={
            "state/region": "State",
            "year": "Year",
            "population": "Population",
        })
        pop_df = pop_df[pop_df["ages"].astype(str).str.lower().eq("total")]

        # make a simple lookup {(Year, State) -> Population}
        pop_df["State"] = pop_df["State"].astype(str).str.upper().str.strip()
        pop_df["Year"] = pd.to_numeric(pop_df["Year"], errors="raise")
        pop_df["Population"] = pd.to_numeric(pop_df["Population"], errors="raise")
        pop_lookup = dict(zip(
            zip(pop_df["Year"].astype(int), pop_df["State"]),
            pop_df["Population"].astype(float)
        ))

        # align and divide (per-capita). add *1e5 if you want per 100k
        pops = pd.Series(
            [pop_lookup.get((int(y), str(s).upper().strip())) for y, s in zip(final_df["Year"], final_df["State"])],
            index=final_df.index,
            dtype="float64"
        )
        final_df["Number of Prescriptions"] = final_df["Number of Prescriptions"].astype(float) / pops


    final_df.to_csv(super_file, index=False)
    return final_df




def plot_state_drug_pairs(df: pd.DataFrame):
    """
    Generates and saves a 3-panel plot for each unique State-Drug pair.

    Args:
        df (pd.DataFrame): DataFrame with columns 'State', 'Product Name', 'Year',
                           'Number of Prescriptions', 'dollar_tax', 'percent_tax'.
    """
    # Ensure the output directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Saving plots to '{PLOTS_DIR}' directory...")

    pair_plot_dir = os.path.join(PLOTS_DIR, "state_drug_pairs")
    os.makedirs(pair_plot_dir, exist_ok=True)

    # Define a consistent set of years for the x-axis across all plots
    years = sorted(df['Year'].unique())

    # Group by state and drug, then iterate through each group
    for (state, drug), group_df in df.groupby(['State', 'Product Name']):
        print(f"  -> Generating plot for {state} - {drug}")

        # Sort data by year for chronological plotting
        group_df = group_df.sort_values('Year')

        # --- Create the Plot ---
        plt.figure(figsize=(15, 18))
        plt.suptitle(f'Annual Data for {drug} in {state}', fontsize=20, fontweight='bold')

        # 1) Prescriptions
        plt.subplot(3, 1, 1)
        plt.bar(group_df["Year"], group_df["Number of Prescriptions"], color="skyblue")
        plt.ylabel("Number of Prescriptions")
        plt.title("Number of Prescriptions")
        plt.xticks(years, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 2) Dollar tax
        plt.subplot(3, 1, 2)
        plt.bar(group_df["Year"], group_df["dollar_tax"], color="limegreen")
        plt.ylabel("Tax ($)")
        plt.title("Dollar Tax")
        plt.xticks(years, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 3) Percent tax
        plt.subplot(3, 1, 3)
        plt.bar(group_df["Year"], group_df["percent_tax"], color="salmon")
        plt.xlabel("Year")
        plt.ylabel("Tax (%)")
        plt.title("Percent Tax")
        plt.xticks(years, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

        # --- Save the Figure ---
        # Sanitize drug name for filename (e.g., replace spaces)
        safe_drug_name = drug.replace(' ', '_')
        filename = f"{state}_{safe_drug_name}_Data.png"
        output_path = os.path.join(pair_plot_dir, filename)
        print(f"  -> Saving plot to {output_path}")
        plt.savefig(output_path)

        # --- Close the plot to free memory ---
        plt.close()

    print("Finished generating all plots.")



def plot_correlation_bars(correlation_df: pd.DataFrame):
    """
    Creates a grouped horizontal bar chart to visualize drug correlations.
    """
    # Sort the dataframe to make the chart easier to read
    correlation_df = correlation_df.sort_values('dollar_tax_vs_prescriptions', ascending=True)

    # --- Setup for plotting ---
    drug_names = correlation_df['Product Name']
    y_pos = np.arange(len(drug_names)) # Positions for each drug group on y-axis
    bar_height = 0.35 # Height of each bar

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- Plotting the bars ---
    # Plot dollar tax correlation bars
    ax.barh(y_pos - bar_height/2, 
            correlation_df['dollar_tax_vs_prescriptions'], 
            bar_height, 
            label='Dollar Tax Correlation', 
            color='limegreen')

    # Plot percent tax correlation bars
    ax.barh(y_pos + bar_height/2, 
            correlation_df['percent_tax_vs_prescriptions'], 
            bar_height, 
            label='Percent Tax Correlation', 
            color='salmon')

    # --- Formatting the plot ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(drug_names)
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.set_title('Correlation of Taxes to Number of Prescriptions by Drug')
    ax.legend()

    # Add a vertical line at 0 for reference
    ax.axvline(0, color='grey', linewidth=0.8)

    # Invert y-axis to have the highest value on top
    ax.invert_yaxis() 
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "drug_correlations.png"))

    # --- Close the plot to free memory ---
    plt.close()


def analyze_and_plot_drug_correlations(df: pd.DataFrame):
    """
    Filters, analyzes, and plots drug prescription vs. tax correlations.

    This function performs three main tasks:
    1. Filters the DataFrame to keep only state-drug pairs that have data
       for every year from 2008 to 2018.
    2. Computes the correlation between 'Number of Prescriptions' and both
       'dollar_tax' and 'percent_tax' for each remaining unique drug.
    3. Generates a grouped horizontal bar chart to visualize these correlations.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'Year', 'State',
                           'Product Name', 'Number of Prescriptions',
                           'percent_tax', and 'dollar_tax'.
    """

    # --- 2. Compute the correlation for each unique drug ---
    print("Computing correlations for the remaining drugs...")
    print(f"Initial DataFrame shape: {df.shape}")
    correlation_df = df.groupby('Product Name').apply(lambda g: pd.Series({
        'dollar_tax_vs_prescriptions': g['dollar_tax'].corr(g['Number of Prescriptions']),
        'percent_tax_vs_prescriptions': g['percent_tax'].corr(g['Number of Prescriptions'])
    })).reset_index()
    print(f"Computed correlations for {len(correlation_df)} unique drugs.")

    if correlation_df.empty:
        print("Could not compute correlations.")
        return
        
    print("\nCorrelation Results:")
    print(correlation_df)

    # --- 3. Plot the correlation results ---
    print("\nGenerating plot...")
    # Sort the dataframe to make the chart easier to read
    correlation_df = correlation_df.sort_values('dollar_tax_vs_prescriptions', ascending=True)

    drug_names = correlation_df['Product Name']
    y_pos = np.arange(len(drug_names))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.barh(y_pos - bar_height/2, 
            correlation_df['dollar_tax_vs_prescriptions'], 
            bar_height, 
            label='Dollar Tax Correlation', 
            color='limegreen')

    ax.barh(y_pos + bar_height/2, 
            correlation_df['percent_tax_vs_prescriptions'], 
            bar_height, 
            label='Percent Tax Correlation', 
            color='salmon')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(drug_names)
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.set_title('Correlation of Taxes to Number of Prescriptions by Drug')
    ax.legend()
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.invert_yaxis() 
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "drug_correlation.png"))

    # --- Close the plot to free memory ---
    plt.close()


def plot_correlation_p_values(df: pd.DataFrame):
    """
    Calculates, prints, and plots p-values (and r) for correlations between prescriptions
    and tax data for each drug, with detailed diagnostics to spot data issues.

    Expects columns:
        'Product Name', 'Number of Prescriptions', 'dollar_tax', 'percent_tax',
        and (optionally) 'State', 'Year' for coverage/dup diagnostics.
    """
    print("Calculating p-values, effect sizes, and diagnostics...\n")

    def corr_stats(x, y):
        pairs = pd.DataFrame({"x": x, "y": y}).dropna()
        n_all = len(x)
        n = len(pairs)
        x_nuniq = pairs["x"].nunique() if n else 0
        y_nuniq = pairs["y"].nunique() if n else 0
        r = np.nan
        p = np.nan
        if n >= 3 and x_nuniq >= 2 and y_nuniq >= 2:
            r, p = pearsonr(pairs["x"], pairs["y"])
        return {
            "n_pairs": n,
            "n_all": n_all,
            "frac_missing": (n_all - n) / n_all if n_all else np.nan,
            "x_nuniq": x_nuniq,
            "y_nuniq": y_nuniq,
            "r": r,
            "p": p,
        }

    def calculate_stats(g):
        s_cols = set(g.columns)
        # Coverage diagnostics
        n_states = g["State"].nunique() if "State" in s_cols else np.nan
        n_years  = g["Year"].nunique()  if "Year"  in s_cols else np.nan
        dup_sy = np.nan
        if {"State","Year"}.issubset(s_cols):
            dup_sy = len(g) - g[["State","Year"]].drop_duplicates().shape[0]

        # Correlations/p-values
        d_stats = corr_stats(g["dollar_tax"], g["Number of Prescriptions"])
        p_stats = corr_stats(g["percent_tax"], g["Number of Prescriptions"])

        # Flatten into a Series
        out = {
            # coverage
            "n_rows": len(g),
            "n_states": n_states, "n_years": n_years, "dup_state_year": dup_sy, 
            # dollar tax stats
            "N_dollar_tax": d_stats["n_pairs"],
            "frac_missing_dollar": d_stats["frac_missing"],
            "xuniq_dollar": d_stats["x_nuniq"], "yuniq_dollar": d_stats["y_nuniq"],
            "r_dollar_tax": d_stats["r"], "dollar_tax_p_value": d_stats["p"],
            # percent tax stats
            "N_percent_tax": p_stats["n_pairs"],
            "frac_missing_percent": p_stats["frac_missing"],
            "xuniq_percent": p_stats["x_nuniq"], "yuniq_percent": p_stats["y_nuniq"],
            "r_percent_tax": p_stats["r"], "percent_tax_p_value": p_stats["p"],
        }
        return pd.Series(out)

    # Compute per-drug stats
    stats_df = df.groupby("Product Name", dropna=False).apply(calculate_stats).reset_index()

    # Drop rows where both p-values are NaN
    keep_mask = stats_df[["dollar_tax_p_value","percent_tax_p_value"]].notna().any(axis=1)
    p_value_df = stats_df.loc[keep_mask].copy()

    if p_value_df.empty:
        print("Could not calculate any p-values. Check if data has enough variance.")
        return

    # Pretty print a compact summary
    print("Per-drug diagnostics:")
    print(p_value_df.loc[:, [
        "Product Name",
        "n_rows", "n_states", "n_years", "dup_state_year",
        "N_dollar_tax", "r_dollar_tax", "dollar_tax_p_value",
        "N_percent_tax", "r_percent_tax", "percent_tax_p_value",
        "frac_missing_dollar", "frac_missing_percent"
    ]].to_string(index=False))
    print()

    # Flag suspicious patterns
    tiny_effect_big_p = p_value_df[
        (p_value_df["dollar_tax_p_value"].notna()) &
        (p_value_df["N_dollar_tax"] >= 200) &
        (p_value_df["dollar_tax_p_value"] < 1e-10) &
        (p_value_df["r_dollar_tax"].abs() < 0.10)
    ]
    if not tiny_effect_big_p.empty:
        print("NOTE: These drugs have tiny |r| (<0.10) but extremely small p-values with large N "
              "(likely 'large-N significance'):")
        for _, r in tiny_effect_big_p.iterrows():
            print(f"  - {r['Product Name']}: N={int(r['N_dollar_tax'])}, r={r['r_dollar_tax']:.3f}, p={r['dollar_tax_p_value']:.2e}")
        print()

    lots_of_missing = p_value_df[
        (p_value_df["frac_missing_dollar"] > 0.3) | (p_value_df["frac_missing_percent"] > 0.3)
    ]
    if not lots_of_missing.empty:
        print("WARNING: High missingness (>30%) detected; results may be unstable:")
        for _, r in lots_of_missing.iterrows():
            print(f"  - {r['Product Name']}: missing dollar={r['frac_missing_dollar']:.1%}, "
                  f"missing percent={r['frac_missing_percent']:.1%}")
        print()

    dup_rows = p_value_df[(p_value_df["dup_state_year"].fillna(0) > 0)]
    if not dup_rows.empty:
        print("WARNING: Duplicate (State, Year) rows within a drug; potential double counting:")
        for _, r in dup_rows.iterrows():
            print(f"  - {r['Product Name']}: duplicate (State,Year) rows = {int(r['dup_state_year'])}")
        print()

    low_coverage = p_value_df[
        (p_value_df["n_states"].fillna(0) < 5) | (p_value_df["n_years"].fillna(0) < 5)
    ]
    if not low_coverage.empty:
        print("Heads-up: low coverage (fewer than 5 states or 5 years) for these drugs:")
        for _, r in low_coverage.iterrows():
            ns = int(r["n_states"]) if pd.notna(r["n_states"]) else -1
            ny = int(r["n_years"]) if pd.notna(r["n_years"]) else -1
            print(f"  - {r['Product Name']}: states={ns}, years={ny}")
        print()

    # Save a full debug CSV alongside the plot outputs
    try:
        debug_path = os.path.join(PLOTS_DIR, "drug_correlation_p_values_debug.csv")
        p_value_df.to_csv(debug_path, index=False)
        print(f"Saved debug diagnostics to {debug_path}\n")
    except Exception as e:
        print(f"Could not save debug CSV: {e}")

    # --- Plot p-values (keep your original look) ---
    print("Generating p-value plot...")
    plot_df = p_value_df.sort_values('dollar_tax_p_value', ascending=False)

    drug_names = plot_df['Product Name']
    y_pos = np.arange(len(drug_names))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.barh(y_pos - bar_height/2,
            plot_df['dollar_tax_p_value'],
            bar_height,
            label='Dollar Tax vs. Prescriptions P-Value',
            color='deepskyblue')

    ax.barh(y_pos + bar_height/2,
            plot_df['percent_tax_p_value'],
            bar_height,
            label='Percent Tax vs. Prescriptions P-Value',
            color='tomato')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(drug_names)
    ax.set_xlabel('P-Value')
    ax.set_title('Statistical Significance (P-Value) of Association (tax vs prescriptions)')
    ax.axvline(0.05, color='red', linestyle='--', linewidth=1.2, label='p = 0.05')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()

    if 'PLOTS_DIR' in locals() or 'PLOTS_DIR' in globals():
        plt.savefig(os.path.join(PLOTS_DIR, "drug_correlation_p_values.png"))
    else:
        print("Warning: PLOTS_DIR not defined. Plot will be shown but not saved.")

    plt.show()
    plt.close(fig)





if __name__ == "__main__":
    # Load & restrict to required columns
    drug_df = read_data(DATA_ROOT, load = False)
    print(f"Loaded {len(drug_df)} rows with {len(drug_df.columns)} columns.")
    print(f"{drug_df['Year'].unique()} unique years found.")
    print(f"{drug_df['State'].nunique()} unique states found.")
    print(f"{drug_df['Product Name'].nunique()} unique drugs found.")
    print(f"{drug_df['Number of Prescriptions'].nunique()} unique number of prescriptions found.")
    print(f"{drug_df['percent_tax'].nunique()} unique percent tax values found.")
    print(f"{drug_df['dollar_tax'].nunique()} unique dollar tax values found.")


    plot_state_drug_pairs(drug_df)

    analyze_and_plot_drug_correlations(drug_df)

    plot_correlation_p_values(drug_df)

    