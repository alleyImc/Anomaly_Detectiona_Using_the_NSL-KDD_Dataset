import numpy as np
from scipy.io import arff
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler


# Function to load ARFF files
def load_arff(file_path):
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df

# Load the datasets
train_full = load_arff('datasets/KDDTrain+.arff')
train_20 = load_arff('datasets/KDDTrain+_20Percent.arff')
test_full = load_arff('datasets/KDDTest+.arff')
test_21 = load_arff('datasets/KDDTest-21.arff')

# Check the dimensions of the datasets
datasets = {
    "KDDTrain+": train_full,
    "KDDTrain+_20Percent": train_20,
    "KDDTest+": test_full,
    "KDDTest-21": test_21
}
# Calculate and print class percentages
class_percentages = {}

for name, df in datasets.items():
    class_counts = df['class'].value_counts(normalize=True) * 100
    print(f"{name}:")
    print(f"- Normal: {class_counts.get(b'normal', 0):.2f}%")
    print(f"- Anomaly: {class_counts.get(b'anomaly', 0):.2f}%\n")
    class_percentages[name] = class_counts

# Create a DataFrame for visualization
percentage_df = pd.DataFrame({
    name: {
        'Normal': class_percentages[name].get(b'normal', 0),
        'Anomaly': class_percentages[name].get(b'anomaly', 0)
    } for name in datasets.keys()
}).T

# Plotting the percentages
percentage_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#4CAF50', '#F44336'])
plt.title("Class Distribution Percentages Across Datasets")
plt.ylabel("Percentage")
plt.xlabel("Datasets")
plt.xticks(rotation=45)
plt.legend(title="Class", labels=["Normal", "Anomaly"])
plt.tight_layout()
plt.show()

# Check data types
print(df.dtypes)

################################################################################################
print("------------------------------------------------\n")
################################################################################################

# Function to check missing values
def check_missing_values(dataset_name, df):
    missing_counts = df.isnull().sum()  # Count missing values per column
    missing_percent = (missing_counts / len(df)) * 100  # Calculate percentages
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_counts,
        'Missing Percent': missing_percent
    })
    print(f"Missing values summary for {dataset_name}:\n")
    print(missing_data[missing_data['Missing Count'] > 0])  # Display columns with missing values
    print("\n")

# Check missing value
check_missing_values("KDDTrain+_20Percent", train_20)
check_missing_values("KDDTrain", train_20)
check_missing_values("KDDTest", train_20)
check_missing_values("KDDTest-21", train_20)


################################################################################################
print("------------------------------------------------\n")
################################################################################################
# numeric variable
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# categorical variable
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

print("numeric: ", numeric_columns)
print("categorical: ", categorical_columns)

# function to convert numeric
def convert_categorical_columns_to_numeric(df):
    columns_to_convert = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
    for column in columns_to_convert:
        df[column] = df[column].astype(int)
    return df

# convert to numeric
train_full = convert_categorical_columns_to_numeric(train_full)
train_20 = convert_categorical_columns_to_numeric(train_20)
test_full = convert_categorical_columns_to_numeric(test_full)
test_21 = convert_categorical_columns_to_numeric(test_21)

# check converted data types
print(train_20.dtypes)

################################################################################################
print("------------------------------------------------\n")
################################################################################################


numeric_columns = train_full.select_dtypes(include=['int64', 'float64'])

# Z-score calculation with numerics
z_scores = np.abs(stats.zscore(numeric_columns))

# detect outliers with Z-score
outliers_zscore = (z_scores > 3).all(axis=1)
print(f"outliers with Z-Score calculation : {np.sum(outliers_zscore)}")

# detect outliers with IQR
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"outliers with IQR: {np.sum(outliers_iqr)}")


#plot outliers
plt.figure(figsize=(12, 6))

# Z-Score
plt.subplot(1, 2, 1)
plt.boxplot(numeric_columns)
plt.title('outliers with Z-Score')

# IQR
plt.subplot(1, 2, 2)
plt.boxplot(numeric_columns)
plt.title('outliers with IQR')

plt.tight_layout()
plt.show()

#tespit edilen aykırı değerler src bytes ve dst bytes da tespit edilip
# anormoli değişkeninini yüksek derecede etkileyeceği için düşürülmeyecktir
################################################################################################
print("------------------------------------------------\n")
################################################################################################

# Columns to encode
categorical_columns = ['protocol_type', 'service', 'flag']
target_column = 'class'

# Apply Label Encoding categorical columns
def apply_label_encoding(df, categorical_columns):
    label_encoder = LabelEncoder()

    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df


# Apply Label Encoding target column
def apply_target_label_encoding(df, target_column):
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])
    return df


# Apply Label Encoding on categorical columns
train_full_encoded = apply_label_encoding(train_full, categorical_columns)
train_20_encoded = apply_label_encoding(train_20, categorical_columns)
test_full_encoded = apply_label_encoding(test_full, categorical_columns)
test_21_encoded = apply_label_encoding(test_21, categorical_columns)

# Apply Label Encoding on target column 'class'
train_full_encoded = apply_target_label_encoding(train_full_encoded, target_column)
train_20_encoded = apply_target_label_encoding(train_20_encoded, target_column)

# Check the updated data
print(f"Test 21 Encoded:\n{test_21_encoded.head()}\n")
print(f"Train 20 Encoded:\n{train_20_encoded.head()}\n")
print(f"Test Full Encoded:\n{test_full_encoded.head()}\n")
print(f"Train Full Encoded:\n{train_full_encoded.head()}\n")

# Count the number of columns in the dataset
num_columns = len(train_full_encoded.columns)
print(f"Number of columns: {num_columns}")
################################################################################################
print("------------------------------------------------\n")
################################################################################################