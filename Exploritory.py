import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
def read_data(path_to_csvs):
    # Read all CSV files in a directory
    # path_to_csvs: str, path to directory containing CSV files
    # returns: pd.DataFrame
    super_file = os.path.join(path_to_csvs, "StateDrugUtilizationData.csv")
    if os.path.exists(super_file):
        df = pd.read_csv(super_file, low_memory=False)
        return df
    files = os.listdir(path_to_csvs)
    #print(f"{files=}")
    dfs = []
    for f in tqdm(files, desc = "Processing"):
        #print(f)
        year = f.split(".")[0][-4:]
        #print(f"{year=}")
        if f.endswith(".csv"):
            year_df = pd.read_csv(os.path.join(path_to_csvs, f), low_memory=False)
            year_df["Year"] = year
            dfs.append(year_df)
    df = pd.concat(dfs)

    with open(os.path.join(path_to_csvs, "unique_drugs.txt"), "w") as f:
        uv = df["Product Name"].unique()
        for v in uv:
            f.write(f"{v}" + "\n")

    drugs_of_interest = ["BUPROPRION", "WELLBUTRIN", "ZYBAN",
                          "FORVIVO", "VARENCICLINE", "CHANTIX", "CHAMPIX", "NICOTINE", "TYRVAYA"]
    
    mask = df["Product Name"].str.contains('|'.join(drugs_of_interest), case=False, na=False)
    df_drug = df[mask]

    df_tax = pd.read_csv(os.path.join(path_to_csvs, "cdc_data.csv"))
    df_drug['Year'] = df_drug['Year'].astype(str)
    df_tax['Year']  = df_tax['Year'].astype(str)
     
    merged_df = pd.merge(df_drug, df_tax, left_on=['Year', 'State'], right_on=['Year', 'LocationAbbr'], how='inner', suffixes=('_drug', '_tax'))

    merged_df.to_csv(super_file, index=False)
    return merged_df


def plot_df(df, state, drug):
    #print(f"before filter state drug: {df.shape}")
    #print(f"{df.columns=}")
    #print(f"Unique states: {df['State'].unique()}")
    #print(f"Unique product names: {df['Product Name'].unique()}")
    mask = df["State"].str.contains(state, case=False, na=False) & df["Product Name"].str.contains(drug, case=False, na=False)
    df = df[mask]
    if df.shape[0] == 0:
        print(f"No data found for {drug} in {state}")
        return
    #print(f"after filter state drug: {df.shape}\n\n\n\n")

    df = df.drop(columns=["State", "Product Name", "Data_Value_Type_tax"])
    df = df.groupby('Year').sum().reset_index()
    df = df.sort_values(by='Year')
    #df.to_csv(f"/home/max/DrugUtil/tmp/{state}_{drug}.csv", index=False)
    plt.figure(figsize=(18, 18))

    # First plot: Year by Number of Prescriptions
    plt.subplot(3, 1, 1)
    plt.bar(df['Year'], df['Number of Prescriptions'], color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Prescriptions')
    plt.title(f'Number of Prescriptions for {drug} in {state} Over Years')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Second plot: Year by Data_Value_tax where Data_Value_Unit_tax == "$"
    df_dollar = df[df["Data_Value_Unit_tax"].str.contains("$", na=False)]
    plt.subplot(3, 1, 2)
    plt.bar(df_dollar['Year'], df_dollar['Data_Value_tax'], color='lightgreen')
    plt.xlabel('Year')
    plt.ylabel('Data Value Tax ($)')
    plt.title(f'Data Value Tax ($) for {drug} in {state} Over Years')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Third plot: Year by Data_Value_tax where Data_Value_Unit_tax == "%"
    df_percent = df[df["Data_Value_Unit_tax"].str.contains("%", na=False)]
    plt.subplot(3, 1, 3)
    plt.bar(df_percent['Year'], df_percent['Data_Value_tax'], color='lightcoral')
    plt.xlabel('Year')
    plt.ylabel('Data Value Tax (%)')
    plt.title(f'Data Value Tax (%) for {drug} in {state} Over Years')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"/home/max/DrugUtil/plots/{state}_{drug}.png")
    #plt.show(block=False)
    #plt.pause(1)
    plt.close()
def compute_correlation(df, state, drug):
    mask = df["State"].str.contains(state, case=False, na=False) & df["Product Name"].str.contains(drug, case=False, na=False)
    df = df[mask]
    df = df.drop(columns=["State", "Product Name", "Data_Value_Type_tax"])
    df = df.groupby('Year').sum().reset_index()
    df = df.sort_values(by='Year')
    df_dollar = df[df["Data_Value_Unit_tax"].str.contains("$", na=False)]
    df_percent = df[df["Data_Value_Unit_tax"].str.contains("%", na=False)]
    if df_dollar.shape[0] == 0:
        print(f"No data $ data found for {drug} in {state}")
    if df_percent.shape[0] == 0:
        print(f"No data $ data found for {drug} in {state}")

    if df_dollar.shape[0] > 0:
        correlation_dollar = df_dollar["Data_Value_tax"].corr(df_dollar["Number of Prescriptions"])
        #print(f"Correlation between tax ($) and number of prescriptions for {drug} in {state}: {correlation_dollar}")
    if df_percent.shape[0] > 0:
        correlation_percent = df_percent["Data_Value_tax"].corr(df_percent["Number of Prescriptions"])
        #print(f"Correlation between tax (%) and number of prescriptions for {drug} in {state}: {correlation_percent}")
    #print(f"{state}, {drug} ({correlation_dollar + correlation_percent / 2})")
    return correlation_dollar + correlation_percent / 2
if __name__ == "__main__":
    drug_df = read_data("/home/max/DrugUtil/Data")
    #print(f"{drug_df['Product Name'].unique()=}")
    #print(f"{drug_df}")
    #print(f"{drug_df.columns}")

    required_columns = [
        "Year", 
        "State", 
        "Product Name", 
        "Number of Prescriptions", 
        "Total Amount Reimbursed", 
        "Data_Value_tax", 
        "Data_Value_Unit_tax", 
        "Data_Value_Type_tax"
    ]

    # Assuming 'df' is your DataFrame
    drug_df = drug_df[required_columns]
    #drug_df = drug_df.dropna()
    #print(drug_df)
    #drug_df = drug_df.sort_values(by="Year")
    #drug_df.to_csv("/home/max/DrugUtil/tmp.csv", index=False)
    #for drug in  ["BUPROPRION"]:
    new_df = pd.DataFrame(columns=["State", "Drug", "Correlation"])
    for state in tqdm(drug_df["State"].unique()):
    #for state in drug_df["State"].unique():
        for drug in  ["WELLBUTRIN","CHANTIX", "NICOTINE"]:
            #plot_df(drug_df, state, drug)
            correlation = compute_correlation(drug_df, state, drug)
            new_df = pd.concat([new_df, pd.DataFrame([{"State": state, "Drug": drug, "Correlation": correlation}])], ignore_index=True)
    new_df = new_df.sort_values(by="Correlation", ascending=False)
    new_df.to_csv("/home/max/DrugUtil/correlations.csv", index=False)
    #plt.show()