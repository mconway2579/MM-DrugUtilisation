from Preprocessing import read_data, DATA_ROOT
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = [
    "Average Cost per pack","Cigarette Consumption (Pack Sales Per Capita)",
    "Federal and State Tax per pack",
    "Federal and State tax as a Percentage of Retail Price",
    "Gross Cigarette Tax Revenue","State Tax per pack"
]


def train_models(drug_df, MODELS):
    # Example: K-Fold Cross Validation
    drugs = drug_df["Product Name"].unique()
    results = {}
    for drug in drugs:
        for per_capita in [False]:
            target_str = "Number of Prescriptions"
            if per_capita:
                target_str = "Number of Prescriptions Per Capita"
            single_drug_df = drug_df[drug_df["Product Name"] == drug]
            if len(single_drug_df) < 100:
                print(f"Skipping {drug} due to insufficient data ({len(single_drug_df)} records).")
                continue
            single_drug_df = single_drug_df.dropna(subset=FEATURES + [target_str])
            folds = KFold(n_splits=5, shuffle=True, random_state=42).split(single_drug_df)
            for fold, (train_index, test_index) in enumerate(folds):
                    X_train = single_drug_df.iloc[train_index][FEATURES]
                    X_test = single_drug_df.iloc[test_index][FEATURES]
                    
                    y_train = single_drug_df.iloc[train_index][target_str]
                    y_test = single_drug_df.iloc[test_index][target_str]
                    for model_constructor in MODELS:
                        model = model_constructor()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        # Evaluate your model here (e.g., calculate metrics)
                        RMSE = ((predictions - y_test) ** 2).mean() ** 0.5
                        key = f"{model.__class__.__name__}, {drug}, {per_capita=}"
                        if key in results:
                            results[key].append(RMSE)
                        else:
                            results[key] = [RMSE]
                        print(f"Trained {key} for fold {fold+1} with RMSE: {RMSE:.4f}")
    print(results)
    return results

def plot_results(results):
    """
    Creates a box and whisker plot for model performance results.
    
    Args:
        results (dict): Dictionary with model descriptions as keys and lists of RMSE values as values.
    """
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(max(14, len(results)*0.8), 8))
    
    # Create box plot
    boxes = ax.boxplot(list(results.values()), patch_artist=True)
    
    # Add colors to boxes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, box in enumerate(boxes['boxes']):
        box.set(facecolor=colors[i % 10])
    
    # Format x-axis labels for better readability
    formatted_labels = [label.replace(', ', '\n').replace('=', '=\n') for label in list(results.keys())]
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
    
    # Add labels and formatting
    ax.set_ylabel('RMSE')
    ax.set_title('Model Performance Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Set y-axis limits
    ax.set_ylim(0, 50000)
    
    # Ensure layout looks good
    plt.tight_layout()
    plt.savefig("model_performance_boxplot.png")
    plt.show(block=False)


if __name__ == "__main__":
    # Load & restrict to required columns
    drug_df = read_data(DATA_ROOT, load = True, compress_similarly_named_drugs=True)
    # Drop rows where 'Product Name' contains 'wellbutrin' (case-insensitive)
    drug_df = drug_df[~drug_df['Product Name'].str.contains('wellbutrin', case=False)]
    NN = lambda: Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(50,50), max_iter=10000))])
    Ridge_Regression = lambda: Ridge(alpha=10.0, max_iter=10000)
    Lasso_Regression = lambda: Lasso(alpha=10.0, max_iter=10000)
    MODELS = [Ridge_Regression, Lasso_Regression, NN]
    training_results = train_models(drug_df, MODELS)
    plot_results(training_results)

    results = {k: np.mean(v) for k, v in training_results.items()}
    print("Average RMSE results across folds:")
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=False))
    for i, (k,v) in enumerate(sorted_results.items()):
        print(f"{i+1}. {k}: {v:.4f}\n")

    plt.show()