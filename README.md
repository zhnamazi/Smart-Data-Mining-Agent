# Smart Data Mining Agent

An intelligent data mining system built with LangChain ReAct Agent, Streamlit, and Metis AI. The agent accepts natural language requests and autonomously executes data mining tasks in the correct order.

---

## Features

- Natural language interface via Streamlit chat
- Automatic dataset loading and summarization
- Configurable preprocessing pipeline
- Classification with 4 algorithms (Decision Tree, Logistic Regression, Random Forest, Linear SVM)
- Frequent pattern mining (Apriori / FP-Growth)
- Automatic `.txt` report generation with download button

---

## Project Structure

```
├── app.py                  # Streamlit UI — handles file upload, chat, and report download
├── agent_controller.py     # LangChain ReAct agent — tools, prompt, and executor
├── modules/
│   ├── data_loader.py      # Loads CSV files into a Pandas DataFrame
│   ├── preprocessing.py    # Executes the preprocessing plan (missing values, encoding, scaling, etc.)
│   ├── frequent_patterns.py# Mines frequent itemsets and association rules (Apriori / FP-Growth)
│   └── classification.py   # Trains and evaluates classifiers; returns summary + full details
├── data/                   # Uploaded CSV files are saved here temporarily (auto-created)
├── outputs/                # Generated report .txt files are saved here (auto-created)
└── requirements.txt
```

---

## Module Descriptions

### `app.py`
The Streamlit front-end. Handles CSV file upload via the sidebar, displays the chat interface, invokes the agent executor on each user message, and shows a download button when a report has been generated.

### `agent_controller.py`
The core of the system. Contains:
- **`agent_state`** — global in-memory state shared across all tools
- **Tool 1: `Analyze_And_Load`** — loads the dataset, computes a full statistical summary, and returns it to the model so it can decide on configuration
- **Tool 2: `Execute_Plan`** — receives the full configuration JSON from the model and runs preprocessing → classification → frequent patterns → report in order
- **`build_executor(file_path)`** — called by `app.py` to rebuild the agent whenever the uploaded file changes

### `modules/data_loader.py`
Reads a CSV file from disk and returns a Pandas DataFrame. Handles basic I/O errors.

### `modules/preprocessing.py`
Executes a preprocessing plan dictionary with five independently togglable steps:
| Step | Description |
|---|---|
| `handle_missing_values` | Fill nulls with median (numeric) or mode (categorical) |
| `remove_duplicates` | Drop exact duplicate rows |
| `encode_categoricals` | Label encoding or One-Hot encoding |
| `handle_outliers` | Clip/remove outliers using IQR or Z-score |
| `scale_numericals` | Standard or MinMax scaling for numeric columns |

### `modules/frequent_patterns.py`
Mines frequent itemsets and association rules from the preprocessed dataset. Supports two algorithms: **Apriori** and **FP-Growth** (default).

### `modules/classification.py`
Trains and evaluates a classifier. Supports four algorithms:
| Key | Algorithm |
|---|---|
| `tree` | Decision Tree (`max_depth=6`) |
| `logistic` | Logistic Regression |
| `random_forest` | Random Forest (`n_estimators=100`) |
| `svm` | Linear SVM (LinearSVC) |

Returns a short **summary string** (for the agent's response) and a full **details dictionary** (for the report), including accuracy, classification report, confusion matrix, and feature importances (tree-based models only).

---

## How It Works

The agent follows a strict two-step flow for any dataset-related request:

```
User message
    │
    ▼
Model decides which tasks are needed
    │
    ▼
Tool 1: Analyze_And_Load  ──►  loads data + returns summary
    │
    ▼
Model decides configuration for each task based on summary
    │
    ▼
Tool 2: Execute_Plan  ──►  runs all tasks in order + generates report
    │
    ▼
Final Answer to user  (+  download button if report was generated)
```

For greetings or non-dataset questions, the model answers directly without calling any tool.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file or set the following variables in your environment:

```
METIS_AI_API_KEY=your-api-key-here
METIS_AI_BASE_URL=https://api.metisai.ir/openai/v1
METIS_MODEL_NAME=gpt-4.1-nano
```

### 3. Run locally

```bash
streamlit run app.py
```

---

## Dataset

This project was developed and tested using the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (UCI Machine Learning Repository). The **approximate accuracy** on this dataset reached an average of **90%** across different cases of classification.
