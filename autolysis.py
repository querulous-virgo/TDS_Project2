import sys
import requests
import json
import os
import glob
import base64
from io import StringIO
import pandas as pd

# Retrieve API token from environment variable
token = os.environ.get("AIPROXY_TOKEN")

# Function to generate pairplots for visualizing data
def get_categorical_numerical_column_name(categorical_column, folder, df, numerical_column):
    import seaborn as sns
    import matplotlib.pyplot as plt

    current_dir = os.getcwd()
    os.chdir(folder)  # Change to the folder where images will be saved

    # Parse numerical columns
    num_cols = [col.strip() for col in numerical_column.split(',') if col.strip()]

    try:
        for col in categorical_column.split(','):
            col = col.strip()

            if col and df[col].nunique() < 15:  # Check for reasonable cardinality
                print(f"Generating pairplot for {col}")

                # Add the categorical column to numerical columns for pairplot
                num_cols.append(col)
                sns.pairplot(df[num_cols], corner=True, hue=col)
                plt.savefig(f"pairplot_{col}.png")
                plt.close()
            else:
                print(f"Skipping pairplot generation for {col} due to high cardinality.")

        print("Pairplot generation completed.")
    except Exception as e:
        print(f"Error during pairplot generation: {e}")
    finally:
        os.chdir(current_dir)  # Revert to the original directory

# Function to execute Python code and generate visualizations
def generate_image_from_text_input(folder, text, df):
    text = text.replace("`", '').replace("python", '')  # Clean input code
    current_dir = os.getcwd()
    try:
        os.chdir(folder)  # Change to the folder where images will be saved
        exec(text)  # Execute the Python code
        print("Generated image successfully.")
        return 'success'
    except Exception as e:
        print(f"Error executing Python code: {e}")
        return f"Python code execution failed: {e}"
    finally:
        os.chdir(current_dir)  # Revert to the original directory

# Main function to handle the end-to-end process
def main(filename, token):
    print("Starting visualization process...")

    # Set up headers for API requests
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Load the dataset
    df = pd.read_csv(filename, encoding='latin-1')

    # Create a folder to store results
    base_folder = os.path.splitext(os.path.basename(filename))[0]
    folder = os.path.join(os.getcwd(), base_folder)

    if not os.path.exists(folder):
        os.mkdir(folder)

    # Collect basic dataframe details
    message = {}

    # Capture dataframe info
    buffer = StringIO()
    df.info(buf=buffer)
    message['dataframe column details'] = buffer.getvalue()

    # Include statistical descriptions
    message['dataframe describe'] = df.describe().to_dict()

    # Identify unique values in categorical columns
    categorical_details = ""
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_counts = df[col].nunique()
            categorical_details += f"Unique values in column '{col}': {unique_counts}\n"
    message['dataframe categorical column'] = categorical_details

    # Request Python charting code from OpenAI API
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Generate Python code using seaborn to create 3 different visualizations and save them as PNGs."},
            {"role": "user", "content": json.dumps(message)}
        ]
    }

    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        code = response_json['choices'][0]['message']['content']
        print("Received Python charting code from OpenAI API.")

        # Generate charts
        func_output = generate_image_from_text_input(folder, code, df)

        # Retry if generation fails
        for attempt in range(2):
            if func_output == 'success':
                break
            print(f"Retrying chart generation (attempt {attempt + 1})...")

            # Modify request to include error context
            data["messages"].append({"role": "user", "content": f"Fix the following error: {func_output}"})

            response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=data)

            if response.status_code == 200:
                response_json = response.json()
                code = response_json['choices'][0]['message']['content']
                func_output = generate_image_from_text_input(folder, code, df)

    else:
        print(f"Error fetching charting code: {response.status_code}")

    # Handle pairplot generation using tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_categorical_numerical_column_name",
                "description": "Generate a pairplot using seaborn with appropriate columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "categorical_column": {
                            "type": "string",
                            "description": "Categorical column names for the hue parameter in pairplots."
                        },
                        "numerical_column": {
                            "type": "string",
                            "description": "Numerical column names for pairplots, excluding ID-like columns."
                        }
                    },
                    "required": ["categorical_column", "numerical_column"],
                    "additionalProperties": False
                }
            }
        }
    ]

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Help generate a pairplot using provided tools."},
            {"role": "user", "content": json.dumps(message)},
            {"role": "user", "content": "Generate a seaborn pairplot."}
        ],
        "tools": tools
    }

    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        if "tool_calls" in response_json['choices'][0]['message']:
            tool_call = response_json['choices'][0]['message']['tool_calls'][0]
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])

            if function_name == "get_categorical_numerical_column_name":
                get_categorical_numerical_column_name(
                    function_args.get("categorical_column"),
                    folder,
                    df,
                    function_args.get("numerical_column")
                )

    print("Visualization process completed.")
