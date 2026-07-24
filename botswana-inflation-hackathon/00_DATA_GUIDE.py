print("=== Deep Learning IndabaX Botswana 2026 - Data Guide initialized ===")

# Example: Display expected datasets
datasets = [
    "01_baltic_dry_index_daily.csv",
    "02_brent_crude_monthly.csv",
    "03_botswana_policy_rate.csv",
    "04_fao_botswana_prices.csv",
    "05_human_capital_project.csv"
]

print("Expected files in ./data/ directory:")
for ds in datasets:
    print(f" - {ds}")