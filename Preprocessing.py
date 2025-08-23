import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


# Drug list (spelling hygiene + variants)
drugs_of_interest = [
    "BUPROPRION",
    "WELLBUTRIN", "ZYBAN", "FORVIVO",
    "VARENICLINE", "CHANTIX", "CHAMPIX", "NICOTINE", "TYRVAYA"
]

PLOTS_DIR = "/home/max/DrugUtil/results/plots"
DATA_ROOT = "/home/max/DrugUtil/Data/"


REQUIRED_YEARS = set(range(2008, 2019))  # Includes 2008 through 2018


def read_data(path_to_csvs: str, load = True, compress_similarly_named_drugs = True) -> pd.DataFrame:
    """
    Read and merge drug utilization CSVs with CDC tax data.
    Returns a tidy table with: Year, State, Number of Prescriptions, percent_tax, dollar_tax.
    Caches the merged result as StateTaxDrugUtilizationData.csv in the same folder.
    """
    super_file = os.path.join(path_to_csvs, f"StateTaxDrugUtilizationData_CompressedNames:{compress_similarly_named_drugs}.csv")
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
    if compress_similarly_named_drugs:
        matching_drugs = [
            [drug for drug in drugs_of_interest if drug.lower() in product_name.lower()]
            for product_name in df_drug["Product Name"]
        ]
        for m in matching_drugs:
            assert len(m) == 1, f"No matching drugs found in Product Name {m}"
        matching_drugs = [m[0] for m in matching_drugs]
        # print(f"{matching_drugs=} {len(matching_drugs)=}")
        df_drug['Product Name'] = matching_drugs


    # --- Add rows from same year different quarter (if applicable) ---
    print(f"Aggregating prescription data from {df_drug['Quarter'].nunique()} quarters")
    print(f"Before quarterly aggregation: {df_drug.shape[0]} rows")
    
    # Sum prescriptions across quarters for the same year, state, and product   
    df_drug = df_drug.groupby(['Year', 'State', 'Product Name']).agg({
        'Number of Prescriptions': 'sum'
    }).reset_index()
    
    print(f"After quarterly aggregation: {df_drug.shape[0]} rows")
    # print(df_drug)
    # --- Drop rows with state XX ---
    df_drug = df_drug[df_drug['State'] != 'XX'].reset_index(drop=True)
   
    # --- Load CDC / tax data ---
    tax_path = os.path.join(path_to_csvs, "cdc_data.csv")
    if not os.path.exists(tax_path):
        raise FileNotFoundError(f"Missing cdc_data.csv at {tax_path}")
    df_tax = pd.read_csv(tax_path, low_memory=False)
    

    # --- Build percent_tax / dollar_tax from Data_Value / Data_Value_Unit ---
    # Convert Data_Value to numeric
    df_tax["Data_Value"] = pd.to_numeric(df_tax["Data_Value"], errors="raise")

    # --- Get each measure as a column ---
    df_tax = df_tax.pivot_table(
        index=['LocationAbbr', 'Year'],
        columns='SubMeasureDesc',
        values='Data_Value'
    ).reset_index().rename_axis(columns=None)


    # --- Type hygiene for join keys ---
    df_drug["Year"] = pd.to_numeric(df_drug["Year"], errors="raise").astype("Int64")
    df_drug["Number of Prescriptions"] = pd.to_numeric(df_drug["Number of Prescriptions"], errors="raise")
    df_tax["Year"]  = pd.to_numeric(df_tax["Year"], errors="raise").astype("Int64")
    # Rename LocationAbbr to State in tax data for consistent join keys
    df_tax = df_tax.rename(columns={"LocationAbbr": "State"})

    
    

    check = df_drug.groupby(["Year","State","Product Name"]).size().reset_index(name="n")
    assert (check["n"]==1).all(), "Drug data not 1:1 after aggregation"

    # one row per Year, State for tax
    check = df_tax.groupby(["Year","State"]).size().reset_index(name="n")
    assert (check["n"]==1).all(), "Tax data not 1:1 after aggregation"

    print(f"Drug data shape: {df_drug.shape}"
          f"\nTax data shape: {df_tax.shape}")
    print(f"Drug data columns: {df_drug.columns.tolist()}"
            f"\nTax data columns: {df_tax.columns.tolist()}")
    print(f"{df_drug.shape=}")
    print(f"{df_tax.shape=}")

    # --- Merge & tidy ---
    final_df = (
        df_drug.merge(df_tax, on=["Year", "State"], how="inner").sort_values(["State", "Year"]).reset_index(drop=True)
    )
    print(f"{final_df.shape=}")

    if REQUIRED_YEARS is not None:
        # keep a copy BEFORE filtering
        print(f"Checking for (State, Product Name) combinations that lack some REQUIRED_YEARS = {sorted(REQUIRED_YEARS)}")
        print(f"Initial data shape: {final_df.shape}")
        df_before = final_df.copy()

        # normalize year to int
        df_before["Year"] = pd.to_numeric(df_before["Year"], errors="coerce").astype("Int64")

        # years present per (State, Product Name)
        years_present = (
            df_before.groupby(["State", "Product Name"])["Year"]
                    .agg(lambda s: set(x for x in s.dropna().astype(int)))
                    .rename("years_present")
        )

        # which groups fail the subset test, and which years are missing
        dropped = (
            years_present.to_frame()
            .assign(missing_years=lambda d: d["years_present"].apply(lambda ys: sorted(REQUIRED_YEARS - ys)))
        )
        dropped = dropped[dropped["missing_years"].map(len) > 0].drop(columns="years_present").reset_index()

        # print summary + a sample
        print(f"Dropped {len(dropped)} (State, Product Name) combinations "
            f"because they lack some REQUIRED_YEARS = {sorted(REQUIRED_YEARS)}")

        # optional: also show how many rows were removed for those combos
        removed_rows = (
            df_before.merge(dropped[["State","Product Name"]], on=["State","Product Name"], how="inner")
        )
        print(f"Total rows removed: {len(removed_rows)}")

        # show a few examples
        print(dropped.sort_values(by="missing_years", key=lambda s: s.map(len), ascending=False)
                    .head(20)
                    .to_string(index=False))

        # --- now perform the actual keep-filter if you want ---
        final_df = df_before.groupby(["State", "Product Name"]).filter(
            lambda g: REQUIRED_YEARS.issubset(set(g["Year"].dropna().astype(int)))
        ).reset_index(drop=True)
        print(f"Final data shape: {final_df.shape}")

    # --- NORMALIZE POPULATION ---
    print("Normalizing Number of Prescriptions by state population...")
    print(f"Before population normalization: {final_df.shape=}")

    pop_path = os.path.join(path_to_csvs, "state-population.csv")
    df_population = pd.read_csv(pop_path, low_memory=False)

    # keep only total-pop rows; standardize names
    pop_df = df_population.rename(columns={
        "state/region": "State",
        "year": "Year"
    })
    # Drop all rows where ages is not 'total'
    pop_df = pop_df[pop_df["ages"] == "total"]

    pop_df["Year"] = pd.to_numeric(pop_df["Year"], errors="raise")
    pop_df["population"] = pd.to_numeric(pop_df["population"], errors="raise")

    # Create a lookup dictionary for faster population lookups
    pop_dict = {(row['Year'], row['State']): row['population'] 
                for _, row in pop_df.iterrows()}

    # Iterate over rows of final_df and get corresponding population values
    pops = []
    for _, row in final_df.iterrows():
        pop = pop_dict.get((row['Year'], row['State']), float('nan'))
        pops.append(pop)

    pops = pd.Series(pops, index=final_df.index)
    final_df["Number of Prescriptions Per Capita"] = final_df["Number of Prescriptions"].astype(float) / pops
    # Drop rows where population normalization resulted in NaN values
    print(f"After population normalization: {final_df.shape=}")



    final_df.to_csv(super_file, index=False)
    return final_df



def analyze_correlations(df):
    """
    Analyzes and visualizes the correlation between prescription data and tax factors
    for each unique product in the DataFrame.

    For each product, it:
    1. Creates a dedicated output folder.
    2. Calculates correlations and p-values between prescription numbers and tax factors.
    3. Generates and saves separate bar charts for correlations and p-values.
    """
    # --- Configuration ---
    # Define the columns we want to analyze
    prescription_metrics = [
        'Number of Prescriptions',
        'Number of Prescriptions Per Capita'
    ]
    tax_factors = [
        'Federal and State Tax per pack',
        'Federal and State tax as a Percentage of Retail Price',
        'Gross Cigarette Tax Revenue',
        'State Tax per pack',
        'Cigarette Consumption (Pack Sales Per Capita)' # Added for broader analysis
    ]
    
    # Get a list of unique product names from the DataFrame
    unique_products = df['Product Name'].unique()
    
    print(f"Found {len(unique_products)} unique products. Starting analysis...")

    # --- Main Loop: Iterate over each product ---
    for product in unique_products:
        # Sanitize product name to create a valid folder name
        folder_name = product.replace(' ', '_').replace('/', '_')
        output_path = os.path.join('product_analysis', folder_name)
        
        # Create the directory for the product if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\nProcessing: {product} -> Saving results in '{output_path}'")
        
        # Filter the DataFrame to get data for the current product only
        product_df = df[df['Product Name'] == product].copy()
        
        # --- Data Validation ---
        if product_df.shape[0] < 3:
            print(f"  - Skipping '{product}': insufficient data (less than 3 records).")
            continue
            
        # --- Analysis Loop: Iterate over each prescription metric ---
        for metric in prescription_metrics:
            results = {
                'Tax Factor': [],
                'Correlation': [],
                'P-Value': []
            }
            
            if metric not in product_df.columns or product_df[metric].isnull().all():
                print(f"  - Skipping metric '{metric}': column not found or all values are null.")
                continue

            # --- Correlation Loop: Iterate over each tax factor ---
            for factor in tax_factors:
                if factor not in product_df.columns or product_df[factor].isnull().all():
                    print(f"  - Skipping factor '{factor}': column not found or all values are null.")
                    continue

                temp_df = product_df[[metric, factor]].dropna()

                if temp_df.shape[0] < 3:
                    print(f"  - Skipping correlation for '{metric}' vs '{factor}': insufficient paired data.")
                    correlation, p_value = np.nan, np.nan
                else:
                    try:
                        correlation, p_value = pearsonr(temp_df[metric], temp_df[factor])
                    except Exception as e:
                        print(f"  - Could not calculate correlation for '{metric}' vs '{factor}': {e}")
                        correlation, p_value = np.nan, np.nan

                results['Tax Factor'].append(factor)
                results['Correlation'].append(correlation)
                results['P-Value'].append(p_value)

            results_df = pd.DataFrame(results).dropna()

            if results_df.empty:
                print(f"  - No valid correlation results for '{metric}'. Skipping plots.")
                continue

            # --- Visualization ---
            plt.style.use('seaborn-v0_8-whitegrid')
            metric_title = metric.replace('_', ' ')
            sample_size = product_df.shape[0]

            # === PLOT 1: CORRELATION COEFFICIENTS ===
            fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Correlation', y='Tax Factor', data=results_df, palette='viridis', ax=ax_corr)
            
            for index, row in results_df.iterrows():
                ax_corr.text(row['Correlation'] / 2, index, f"r = {row['Correlation']:.3f}", 
                             color='white', ha='center', va='center', fontweight='bold')

            ax_corr.set_title(f'Correlation of {metric_title}\nwith Tax & Economic Factors for {product} (N={sample_size})',
                              fontsize=16, fontweight='bold')
            ax_corr.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12)
            ax_corr.set_ylabel('Tax & Economic Factor', fontsize=12)
            ax_corr.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            
            plt.tight_layout()
            plot_filename_corr = f"{metric.replace(' ', '_')}_correlation.png"
            fig_corr.savefig(os.path.join(output_path, plot_filename_corr))
            plt.close(fig_corr)
            print(f"  - Successfully generated and saved plot: {plot_filename_corr}")

            # === PLOT 2: P-VALUES ===
            fig_p, ax_p = plt.subplots(figsize=(12, 8))
            sns.barplot(x='P-Value', y='Tax Factor', data=results_df, palette='plasma', ax=ax_p)

            for index, row in results_df.iterrows():
                ax_p.text(row['P-Value'] / 2, index, f"r = {row['Correlation']:.3f}",
                          color='white', ha='center', va='center', fontweight='bold')

            ax_p.set_title(f'P-Values for Correlation of {metric_title}\nwith Tax Factors for {product} (N={sample_size})',
                           fontsize=16, fontweight='bold')
            ax_p.set_xlabel('P-Value', fontsize=12)
            ax_p.set_ylabel('Tax & Economic Factor', fontsize=12)
            # Add a line at p=0.05 for significance threshold
            ax_p.axvline(x=0.05, color='red', linewidth=1.2, linestyle='--')
            ax_p.text(0.051, ax_p.get_ylim()[1] * 0.95, 'p = 0.05', color='red', va='center')


            plt.tight_layout()
            plot_filename_p = f"{metric.replace(' ', '_')}_p_values.png"
            fig_p.savefig(os.path.join(output_path, plot_filename_p))
            plt.close(fig_p)
            print(f"  - Successfully generated and saved plot: {plot_filename_p}")

    print("\nAnalysis complete.")





if __name__ == "__main__":
    # Load & restrict to required columns
    drug_df = read_data(DATA_ROOT, compress_similarly_named_drugs=True)
    # print(f"Loaded {len(drug_df)} rows with {len(drug_df.columns)} columns.")
    # print(f"{drug_df['Year'].unique()} unique years found.")
    # print(f"{drug_df['State'].nunique()} unique states found.")
    # print(f"{drug_df['Product Name'].nunique()} unique drugs found. {drug_df['Product Name'].unique()}")
    # print(f"{drug_df['Number of Prescriptions'].nunique()} unique number of prescriptions found.")
    # print(f"{drug_df['percent_tax'].nunique()} unique percent tax values found.")
    # print(f"{drug_df['dollar_tax'].nunique()} unique dollar tax values found.")


    analyze_correlations(drug_df)


    