import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 3.1 Data Collection
def load_datasets():
    try:
        # Load CSV file
        bot_detection_data = pd.read_csv('D:/Games/School Stuff/Github/THESIS2_ALGORHYTHM/bot_detection_data.csv')
    except ValueError as e:
        print(f"Error reading file: {e}")
        return None
    return bot_detection_data

# 3.2 Data Preprocessing

# 3.2.1 Tokenize specified columns and add reverted tokens for each tokenized word
def tokenize_and_revert_columns(df, columns):
    tokenizers = {}
    tokenized_data = []
    reverted_data = []
    column_indices = {}  # To keep track of column indices

    current_index = 0  # To keep track of where each column's data starts in the tokenized data

    for column in columns:
        # Ensure column is treated as strings and fill NaN values with empty strings
        df[column] = df[column].astype(str).fillna('')

        # Initialize the tokenizer for each column
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df[column])
        tokenizers[column] = tokenizer

        # Convert texts to sequences and pad sequences
        sequences = tokenizer.texts_to_sequences(df[column])
        max_sequence_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

        # Append the padded sequences to the list
        tokenized_data.append(padded_sequences)
        column_indices[column] = (current_index, current_index + padded_sequences.shape[1])

        # Update the index for the next column
        current_index += padded_sequences.shape[1]

        # Revert tokens back to text
        reverted_sequences = []
        for seq in sequences:
            reverted_tokens = tokenizer.sequences_to_texts([seq])[0].split()
            # If padding has been applied, pad reverted tokens with empty strings
            while len(reverted_tokens) < max_sequence_length:
                reverted_tokens.insert(0, "")
            reverted_sequences.append(reverted_tokens)
        reverted_data.append(reverted_sequences)

    # Concatenate tokenized columns
    tokenized_data = np.hstack(tokenized_data)

    # Return the tokenized data, reverted strings, tokenizers, and column indices
    return tokenized_data, reverted_data, tokenizers, column_indices

# 3.2.2 Extract date-time features
def extract_date_time_features(df, column):
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        
        df['Year'] = df[column].dt.year
        df['Month'] = df[column].dt.month
        df['Day'] = df[column].dt.day
        df['Hour'] = df[column].dt.hour
        df['Minute'] = df[column].dt.minute
        df['Second'] = df[column].dt.second
        df['Day of Week'] = df[column].dt.dayofweek  # 0=Monday, 6=Sunday
        
        df = df.drop(columns=[column])  # Optionally remove the original column if no longer needed
    else:
        print(f"Warning: '{column}' column not found in the DataFrame.")
    return df

# 3.2.3 Check if string values can be converted to float and remove non-convertible rows
def remove_non_convertible_rows(df, columns):
    def can_convert_to_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    
    for column in columns:
        if column in df.columns:
            df = df[df[column].apply(can_convert_to_float)]
        else:
            print(f"Warning: '{column}' column not found in the DataFrame.")
    
    return df

# 3.2.4 Handle Missing Values (this will be replaced by predictive imputation)
def handle_missing_values(df):
    print(f"Missing values before handling: {df.isnull().sum()}")  # Debugging
    df = df.dropna(how='all')  # Drop rows where all elements are missing
    print(f"Missing values after handling: {df.isnull().sum()}")  # Debugging
    return df

# 3.2.5 Correct Inconsistencies
def correct_inconsistencies(df):
    if 'Username' in df.columns:
        df['Username'] = df['Username'].str.lower()
    print(f"After correcting inconsistencies: {df.head()}")  # Debugging
    return df

# 3.2.6 Remove Duplicates
def remove_duplicates(df):
    list_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]
    print(f"Columns with list type values: {list_columns}")  # Debugging
    for col in list_columns:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")  # Debugging
    return df

# 3.2.7 Handle Outliers
def handle_outliers(df, columns):
    for column in columns:
        if column in df.columns:
            if df[column].empty:
                print(f"Warning: Column '{column}' is empty. Skipping outlier handling.")
                continue
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        else:
            print(f"Warning: '{column}' column not found in the DataFrame. Skipping outlier handling for this column.")
    print(f"After handling outliers: {df.shape}")  # Debugging
    return df

# 3.2.8 Normalize Numerical Features
def normalize_features(df, columns, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns for normalization: {missing_columns}. Skipping normalization.")
        return df
    if df[columns].empty:
        print("Warning: No data to normalize. Skipping normalization.")
        return df
    df[columns] = scaler.fit_transform(df[columns])
    print(f"After normalization: {df.head()}")  # Debugging
    return df

# 3.2.10 Remove Rows with NaN Values (if still necessary)
def remove_rows_with_nan(df):
    # Check for rows with any NaN values
    nan_rows_before = df.isnull().sum().sum()
    print(f"Total NaN values before removal: {nan_rows_before}")  # Debugging
    
    # Remove rows with any NaN values
    df = df.dropna()
    
    # Check for rows with any NaN values after removal
    nan_rows_after = df.isnull().sum().sum()
    print(f"Total NaN values after removal: {nan_rows_after}")  # Debugging
    print(f"Data shape after removing rows with NaN: {df.shape}")  # Debugging
    
    return df

def check_for_nans(df):
    if df.isnull().sum().sum() > 0:
        print("Data contains NaN values:")
        print(df.isnull().sum())
    else:
        print("No NaN values in the data.")

def fill_na_usernames(df):
    if 'Username' in df.columns:
        df['Username'] = df['Username'].fillna('unknown')  # Replace NaN with 'unknown' or another placeholder
    return df

def split_and_save_human_bot_accounts(df, label_column='Bot Label'):
    if label_column not in df.columns:
        print(f"Error: '{label_column}' column not found in the DataFrame.")
        return
    
    # Check number of human and bot accounts
    human_accounts = df[df[label_column] == 0]
    bot_accounts = df[df[label_column] == 1]
    
    if human_accounts.empty or bot_accounts.empty:
        print("Error: Either human or bot accounts are missing in the dataset. Splitting aborted.")
        return
    
    print(f"Number of human accounts: {len(human_accounts)}")
    print(f"Number of bot accounts: {len(bot_accounts)}")
    
    # Split human accounts into 80/20
    human_train = human_accounts.sample(frac=0.8, random_state=42)
    human_test = human_accounts.drop(human_train.index)
    
    # Split bot accounts into 80/20
    bot_train = bot_accounts.sample(frac=0.8, random_state=42)
    bot_test = bot_accounts.drop(bot_train.index)
    
    # Combine human and bot splits for train and test
    train_data = pd.concat([human_train, bot_train], ignore_index=True)
    test_data = pd.concat([human_test, bot_test], ignore_index=True)
    
    # Shuffle the combined data for randomness
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save train and test data to CSV files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    
    print("Train and test datasets have been saved:")
    print(f"  - Training data: {train_data.shape} (80% human + bot accounts)")
    print(f"  - Testing data: {test_data.shape} (20% human + bot accounts)")

def main():
    bot_detection_data = load_datasets()
    if bot_detection_data is None:
        return
    
    # 1. Tokenize the specified columns and revert tokens
    tokenized_columns = ['Tweet', 'Location', 'Hashtags']
    tokenized_data, reverted_data, tokenizers, column_indices = tokenize_and_revert_columns(bot_detection_data, tokenized_columns)
    
    # Add tokenized and reverted columns to the dataframe
    for column, (start_idx, end_idx) in column_indices.items():
        tokenized_columns_names = [f'{column}_token_{i}' for i in range(end_idx - start_idx)]
        reverted_columns_names = [f'{column}_reverted_{i}' for i in range(end_idx - start_idx)]

        bot_detection_data[tokenized_columns_names] = tokenized_data[:, start_idx:end_idx]
        for i, reverted_column_name in enumerate(reverted_columns_names):
            bot_detection_data[reverted_column_name] = [reverted_row[i] for reverted_row in reverted_data[tokenized_columns.index(column)]]

    print("Tokenization and reverting done!")
    
    # 2. Extract Date-Time features
    bot_detection_data = extract_date_time_features(bot_detection_data, 'Created At')
    
    # 3. Handle missing values
    bot_detection_data = handle_missing_values(bot_detection_data)
    
    # 4. Correct inconsistencies
    bot_detection_data = correct_inconsistencies(bot_detection_data)
    
    # 5. Remove duplicates
    bot_detection_data = remove_duplicates(bot_detection_data)
    
    # 6. Handle outliers
    bot_detection_data = handle_outliers(bot_detection_data, ['Follower Count', 'Retweet Count'])
    
    # 7. Normalize features
    bot_detection_data = normalize_features(bot_detection_data, ['Follower Count', 'Retweet Count'])
    
    # 8. Remove rows with NaN values
    bot_detection_data = remove_rows_with_nan(bot_detection_data)
    
    # 9. Check for NaN values in the dataset
    check_for_nans(bot_detection_data)
    
    # 10. Split human and bot accounts and save to train/test datasets
    split_and_save_human_bot_accounts(bot_detection_data, label_column='Bot Label')
    
    # Save processed data
    if not bot_detection_data.empty:
        bot_detection_data.to_csv('processed_input_data.csv', index=False)
        print("Processed data saved to 'processed_input_data.csv'.")
    else:
        print("Processed data is empty, not saving.")

if __name__ == "__main__":
    main()