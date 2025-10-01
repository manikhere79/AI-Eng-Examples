import json
import pandas as pd
import os

# --- Configuration Constants ---
# Use the same path as your original script
SOURCE_FILE_PATH = "../data/News_Category_Dataset_v3.json" 
# Output file for the filtered test data subset
OUTPUT_FILE_PATH = "./data/test_data_subset.json"
MAX_RECORDS_TO_CHECK = 10000 # Limit initial data pull for category analysis
CATEGORY_COUNT = 8 # Number of diverse categories to select for the test set

def load_data(file_path: str) -> pd.DataFrame or None:
    """
    1. Loads the data from the JSON Lines file and 2. converts it to a pandas DataFrame.
    """
    print(f"--- 1. Loading data from {file_path}...")
    data = []
    
    # Ensure the data directory exists before attempting to read/write
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # The dataset is a JSON Lines file, so we load line by line
            for line in f:
                data.append(json.loads(line))
        
        # 2. Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"    ✅ Loaded {len(df)} total records.")
        return df
        
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Data file not found at '{file_path}'.")
        print("Please ensure your data file is correctly named and placed inside the 'data' folder.")
        return None
    except Exception as e:
        print(f"\nFATAL ERROR during data loading: {e}")
        return None

def process_and_save_test_set(df: pd.DataFrame, output_path: str, max_records: int, cat_count: int):
    """
    3. Filters the first N records, 4. selects top categories, and 5. saves the subset.
    """
    if df is None:
        return
        
    print(f"--- 2. Analyzing first {max_records} records for categories...")
    
    # 3. Filter to the first N records
    df_subset = df.head(max_records).copy()
    
    # 4. Identify the top N most frequent and diverse categories
    # Sorting by frequency helps select categories with enough samples
    top_categories = df_subset['category'].value_counts().head(cat_count).index.tolist()
    
    print(f"    Selected {len(top_categories)} categories for the test set:")
    for cat in top_categories:
        print(f"        - {cat}")

    # Filter the WHOLE dataset (or the first 10k, for simplicity) to include ONLY these selected categories
    # FIX: Adding .copy() to explicitly create a new DataFrame, resolving the SettingWithCopyWarning.
    df_test_set = df_subset[df_subset['category'].isin(top_categories)].copy()
    
    print(f"    ✅ Filtered down to {len(df_test_set)} records belonging to these categories.")
    
    # Prepare data for easy RAG testing by creating a stable ID
    df_test_set['id'] = [f"doc_{i}" for i in range(len(df_test_set))]
    
    # 5. Write the final DataFrame to a new JSON Lines file
    try:
        # Convert DataFrame to JSON Lines format (records)
        json_records = df_test_set.to_json(orient='records', lines=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_records)
            
        print(f"--- 3. Successfully saved {len(df_test_set)} test set records to '{output_path}'")
        print("\nNow you can easily generate sample queries based on the categories in this new file.")

    except Exception as e:
        print(f"\nFATAL ERROR during file saving: {e}")


if __name__ == "__main__":
    # The user is expected to have the News_Category_Dataset_v3.json in the 'data/' folder.
    
    # Create the data folder if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load the data
    news_df = load_data(SOURCE_FILE_PATH)
    
    # Process and save the filtered subset
    if news_df is not None:
        process_and_save_test_set(news_df, OUTPUT_FILE_PATH, MAX_RECORDS_TO_CHECK, CATEGORY_COUNT)