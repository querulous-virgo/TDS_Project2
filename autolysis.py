import subprocess
import sys
import importlib
import os
from dateutil.parser import parse

# Ensure pip is installed and updated
def ensure_pip():
    try:
        import pip
        print("'pip' is already available.")
    except ImportError:
        print("'pip' not found. Installing pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            print("'pip' has been installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install pip: {e}")
            sys.exit(1)

# Function to check and install a package
def install_and_import(package_name):
    try:
        importlib.import_module(package_name)
        print(f"'{package_name}' is already installed.")
    except ImportError:
        print(f"'{package_name}' not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"'{package_name}' has been installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install '{package_name}': {e}")
            sys.exit(1)

# List of required packages with aliases
packages_with_aliases = {
    "pandas": "pd",
    "matplotlib.pyplot": "plt",
    "seaborn": "sns",
    "openai": None,
    "numpy": "np",
    "tabulate": "tabulate",
    "chardet" : None,
    "requests" : None
}

# Ensure pip is installed
ensure_pip()

# Check and install each package
for package in packages_with_aliases.keys():
    install_and_import(package.split('.')[0])  # Handle submodules

# Import the packages with aliases
for package, alias in packages_with_aliases.items():
    try:
        module = importlib.import_module(package)
        if alias:
            globals()[alias] = module
        else:
            globals()[package] = module
    except ModuleNotFoundError as e:
        print(f"Error importing '{package}': {e}")
        sys.exit(1)

# Example usage to confirm everything is working
print("All packages have been successfully installed and imported!")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import chardet
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def plot_missing(data):

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
    plt.savefig(f'MissingValues.png')
    plt.close()

def plot_correlation(data):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix.png')
    plt.close()

def plot_missing(data):
    """
    Function to plot the number of missing values for each column in a DataFrame.
    The plot is saved as a PNG file named 'MissingValues.png'.

    Parameters:
    data (pd.DataFrame): The input DataFrame to analyze for missing values.

    Returns:
    None
    """

    # Calculate the number of missing values for each column
    missing_values = data.isnull().sum()

    # Create a figure for the bar chart
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Create a barplot using seaborn
    sns.barplot(x=missing_values.index, y=missing_values.values, color="skyblue")

    # Add a title and axis labels to the plot
    plt.title('Number of Missing Values per Column', fontsize=16)  # Set the title with font size
    plt.xlabel('Columns', fontsize=12)  # Set the label for the x-axis
    plt.ylabel('Number of Missing Values', fontsize=12)  # Set the label for the y-axis

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=10)  # Rotate labels and set font size

    # Add a legend to explain the color of the bars
    plt.legend(['Missing Values'], loc='upper right')  # Add a legend to the plot

    # Adjust the layout to ensure nothing overlaps
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('MissingValues.png', dpi=300)  # Save with high resolution

    # Close the figure to free up memory
    plt.close()

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def extractCsvMetadata(file_path):
    """
    Extracts metadata from a CSV file and loads the data into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the loaded DataFrame and a metadata dictionary.
               Metadata includes column names, data types, number of rows, and number of columns.
               If there's an error loading the CSV file, it returns an error message.
    """

    # Step 1: Detect the encoding of the CSV file
    # This step ensures that the file is read correctly, especially for non-UTF-8 encodings.
    encoding = detect_encoding(file_path)

    try:
        # Step 2: Load the CSV file into a pandas DataFrame using the detected encoding
        # This step may raise exceptions if the file is corrupted or inaccessible.
        data = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        # Handle exceptions during file loading and return the error message
        return f"Error loading CSV file: {e}"

    # Step 3: Infer better object types for each column in the DataFrame
    # - This ensures that columns containing mixed types are converted to their most specific type.
    # - For example, a column with numbers stored as strings will be converted to integers or floats.
    data = data.apply(pd.Series.infer_objects)

    # Step 4: Convert columns to their most appropriate pandas data types
    # - This uses pandas' `convert_dtypes`, which attempts to infer the most specific type for each column.
    # - It is useful for optimizing memory usage and ensuring correct data type handling.
    data = data.convert_dtypes()

    # Step 5: Extract metadata from the DataFrame
    # - Metadata includes:
    #   1. "Column Names": A list of all column names in the DataFrame.
    #   2. "Data Types": A dictionary mapping each column to its inferred data type as a string.
    #   3. "Number of Rows": The total number of rows in the DataFrame.
    #   4. "Number of Columns": The total number of columns in the DataFrame.
    metadata = {
        "Column Names": data.columns.tolist(),  # Get the list of column names
        "Data Types": data.dtypes.astype(str).to_dict(),  # Convert dtypes to string and store as a dictionary
        "Number of Rows": data.shape[0],  # Get the number of rows using the shape attribute
        "Number of Columns": data.shape[1]  # Get the number of columns using the shape attribute
    }

    # Step 6: Return the loaded DataFrame and the metadata dictionary
    return data, metadata


def send_metadata_to_openai(metadata, config):
    """
    This function sends a metadata description to OpenAI's API to generate Python code for performing an
    exploratory data analysis (EDA). The generated code is customized based on the provided instructions.

    Parameters:
        metadata (str): Metadata description of the CSV file.
        config (dict): Configuration dictionary containing:
            - AIPROXY_TOKEN: The API token for authorization.
            - AI_PROXY_URL: The endpoint URL for the OpenAI proxy.

    Returns:
        str: The generated Python code from OpenAI, or an error message if the request fails.
    """

    # Step 1: Define the headers for the API request
    headers = {
        "Authorization": f"Bearer {config['AIPROXY_TOKEN']}",  # API token for authentication
        "Content-Type": "application/json"  # Specify JSON format for the request
    }

    # Step 2: Define the prompt that specifies the task for the AI model
    prompt = f"""
    I have a CSV file with the following metadata:
    {metadata}
    Generate Python code to perform a complete exploratory data analysis (EDA) on {dataset_name}.
    """

    # Step 3: Construct the payload with model, messages, and temperature parameters
    payload = {
        "model": "gpt-4o-mini",  # Specify the model to use for code generation
        "messages": [
            {"role": "user", "content": prompt},  # User's task prompt

            # System role defines behavior expectations for the model
            {"role": "system", "content": """You are to generate a python code for the given task.
            Only output the code and nothing else. The code is run in an interpreter so do not add
            the \"python\" command in the front."""},

            # Additional user input to refine the generated code's purpose and behavior
            {"role": "user", "content": """Generate python code to generate a 
            1) bar chart for showing the number of missing values, do not plot if there are no missing values in the entire dataset
            2) select numerical columns, MinMax scale them and plot correlation matrix for them 
            3) Make boxplots for the scaled numerical columns to show outliers. 
            the pandas dataframe is named data and use the seaborn library. The visualizations must have a legend. 
            save both in png format. The dataframe is already loaded as 'data' so just provide the remaining code."""}
        ],
        "temperature": 0.7  # Controls randomness in the AI's response (higher = more diverse responses)
    }

    # Step 4: Make the API call to OpenAI's endpoint
    try:
        # Send the POST request with headers and payload
        response = requests.post(config["AI_PROXY_URL"], headers=headers, json=payload)

        # Step 5: Check the response status and handle appropriately
        if response.status_code == 200:
            # If successful, parse and return the generated code
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            # If an error occurs, return the status code and error details
            return f"Error: {response.status_code} - {response.text}"

    # Step 6: Handle exceptions during the API call
    except Exception as e:
        # Return an error message if the API call fails
        return f"Error calling OpenAI API: {e}"

# Detailed Comments Summary:
# 1. Clearly document the purpose and parameters of the function.
# 2. Break down each step with explanations (headers, prompt, payload, API call).
# 3. Handle errors explicitly and provide meaningful error messages.
# 4. Ensure maintainability with modular and well-commented code.




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
            with open(os.path.join("README.md"), "w") as f:
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

api_key = os.getenv("OPENAI_TOKEN")
CONFIG = {
    "AI_PROXY_URL": "https://api.openai.com/v1/chat/completions", 
    "AIPROXY_TOKEN": api_key,
    "OUTPUT_DIR": os.path.dirname(os.path.abspath(__file__))
}

data , metadata = extractCsvMetadata(dataset_path)
result = send_metadata_to_openai(metadata , CONFIG)
result.replace('`' , '')
result.replace('python' , '')

if result.startswith("```"):
    result = result.split('\n', 1)[-1].rsplit('\n', 1)[0]

exec(result)

write_readme(data, metadata, dataset_name, CONFIG)
