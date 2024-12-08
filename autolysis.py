# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "requests",
#   "python-dotenv",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "scipy",
#   "statsmodels",
#   "chardet"
# ]
# ///


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import chardet
import os
import sys
import warnings

# Suppress the specific warnings related to missing glyphs
warnings.filterwarnings("ignore")

def plot_missing(data , name):
    # Create media folder if it doesn't exist
    if not os.path.exists(name):
        os.makedirs(name)

    # Calculate missing values
    missing_values = data.isnull().sum()

    # Create a bar chart for missing values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title('Number of Missing Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f'media/{name}_MissingValues.png')
    plt.close()

def plot_correlation(data , name):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(f'{name}/correlation_matrix.png')
    plt.close()

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def extractCsvMetadata(file_path):
    # Load the CSV file
    encoding = detect_encoding(file_path)
    try:
        # Load CSV with detected encoding
        data = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        return f"Error loading CSV file: {e}"
    
    data = data.apply(pd.Series.infer_objects)
    
    # Convert columns to appropriate data types
    data = data.convert_dtypes()

    # Extract metadata
    metadata = {
        "Column Names": data.columns.tolist(),
        "Data Types": data.dtypes.astype(str).to_dict(),
        "Number of Rows": data.shape[0],
        "Number of Columns": data.shape[1]
    }
    return data , metadata

def send_metadata_to_openai(metadata, config):
    headers = {
        "Authorization": f"Bearer {config['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }
    prompt = f"""
    I have a CSV file with the following metadata:
    {metadata}
    Generate Python code to perform a complete exploratory data analysis (EDA) on {dataset_name}. 
    
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "system", "content": "You are to generate a python code for the given task. Only output the code and nothing else. The code is run in an interperter so do not add the \"python\" command in the front. Again NEVER add \"python\" command while outputing."},
                        {"role": "user", "content": f"Generate python code to generate a 1) bar chart for showing the number of missing values, do not plot if there are no missing values in the entire dataset , 2) select numerical columns, MinMax scale them and plot correlation matrix for them 3) Make boxplots for the scaled numerical columns to show outliers. a pandas dataframe is named data and use the seaborn library. save both in the {dataset_name} folder in png format. Make the folder if it does not exist. The dataframe is already loaded as 'data' so just provide the remaining code."}
                    ],
        "temperature": 0.7
    }
    try:
        response = requests.post(config["AI_PROXY_URL"], headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

from sklearn.model_selection import train_test_split

def write_readme(df, metadata, dataset_name, config):
    # Perform stratified sampling with a maximum sample size of 5
    sample_size = min(10, len(df))
    stratified_sample = df.sample(n=sample_size, random_state=42)

    # Convert the sample to a dictionary for easy reading by the LLM
    sample_dict = stratified_sample.to_dict(orient="records")

    # Construct the prompt for the LLM
    prompt = f"""
    I have the following dataset metadata:
    {metadata}
    Here is a stratified sample (maximum of 10 records) from the dataset:
    {sample_dict}
    Give a big Heading1 called "Data Analysis for {dataset_name}:" Make a table summarizing the metadata and give a 60 word description of it., in the next line write each column name and it's data type. If any special datatype is there, tell how it would be dealt with.
    Now add a big Heading1 called story and then write a story by taking a column as the target variable and telling how different columns may affect it and how a doctor, an engineer and a child argue about the insights of this data.
    """

    # Send request to OpenAI API to generate the description
    headers = {
        "Authorization": f"Bearer {config['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "system", "content": "Do not add the \"markdown\" in output. Give just the markdown and nothing else as it will be preview in a different md interpreter."},
            {"role": "user", "content": f"Do not describe the data in more than 60 words. The output and must be atleast 200 words and include dialogues and emojis."},
            {"role": "user", "content": f"""Describe within the story:
                    The data you received, briefly
                    The insights you discovered
                    The implications of your findings (i.e. what to do with the insights)"""}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(config["AI_PROXY_URL"], headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            description = result["choices"][0]["message"]["content"]
            # Write the description to a README file
            with open(os.path.join(dataset_name, "README.md"), "w") as f:
                f.write(description)
            print("README generated successfully.")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")

# Main Function begins here 

if len(sys.argv) < 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)
# Extracting dataset name

dataset_path = sys.argv[1]
dataset_name = dataset_path[:-4]
print(dataset_name)
print(f"Analyzing {dataset_path}...\n")

api_key = os.getenv("AIPROXY_TOKEN")
CONFIG = {
    "AI_PROXY_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", 
    "AIPROXY_TOKEN": api_key,
    "OUTPUT_DIR": os.path.dirname(os.path.abspath(__file__))
}

data , metadata = extractCsvMetadata(dataset_path)
result = send_metadata_to_openai(metadata , CONFIG)

try:
    exec(result)
except Exception as e:
    print(f"Error executing generated code: {e}")
    sys.exit(1)

write_readme(data, metadata, dataset_name, CONFIG)
