import os
import json
import datetime
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_classic.tools import Tool
from langchain_openai import ChatOpenAI

from modules.data_loader import load_dataset
from modules.preprocessing import preprocess_data
from modules.frequent_patterns import extract_patterns
from modules.classification import run_classification


# Global in-memory state
agent_state = {
    # file info
    "dataset_name": None,
    "dataframe": None,

    # dataset info
    "dataset_info": None,

    # task flags (set by Tool 1)
    "tasks": None,        # list[str] e.g. ["preprocessing", "classification"]
    
    # preprocess
    "preprocessing_plan": None,
    "preprocessed_dataframe": None,
    "preprocessing_result": None,
    
    # patterns
    "patterns_summary": None,
    "patterns_details": None,
    
    # classification
    "classification_summary": None,
    "classification_results": None, 
    
    # report
    "report_path": None,
}


# Default preprocessing plan (used when preprocessing is not in tasks but
# classification or frequent_patterns require it)
DEFAULT_PREPROCESSING_PLAN = {
    "handle_missing_values": {"apply": True,  "strategy": "median"},
    "remove_duplicates":     {"apply": True},
    "encode_categoricals":   {"apply": True,  "method": "label"},
    "handle_outliers":       {"apply": False, "method": "iqr",      "columns": []},
    "scale_numericals":      {"apply": False,  "method": "standard", "columns": []},
}


# Helper functions
def summarize_data() -> dict:
    df = agent_state.get("dataframe")
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": list(df.select_dtypes(exclude=["number"]).columns),
        "target_hint": df.columns[-1],
        "column_info": {},
    }
    for col in df.columns:
        summary["column_info"][col] = {
            "dtype": str(df[col].dtype),
            "missing_values": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().unique()[:3].tolist(),
        }
    agent_state["dataset_info"] = summary
    return summary

def configure_and_preprocess(config: dict) -> str:
    df = agent_state.get("dataframe")
    target_col = config.get("target_column")
    
    plan = config.get("preprocessing", DEFAULT_PREPROCESSING_PLAN)

    # ensure all required keys are present
    required_steps = {
        "handle_missing_values", "encode_categoricals",
        "scale_numericals", "remove_duplicates", "handle_outliers",
    }
    missing_keys = required_steps - set(plan.keys())
    # fill missing keys from default
    for k in missing_keys:
        plan[k] = DEFAULT_PREPROCESSING_PLAN[k]

    agent_state["preprocessing_plan"] = plan

    df_pre, pre_msg = preprocess_data(df, plan, target_col)
    agent_state["preprocessed_dataframe"] = df_pre
    agent_state["preprocessing_result"] = pre_msg
    results = f"[PREPROCESSING]\n{pre_msg}"
    return results

def configure_and_classify(config: dict) -> str:
    target_col = config.get("target_column") or agent_state["dataset_info"]["target_hint"]
    clf_config = config.get("classification", {})
    model_type = clf_config.get("model_type", "tree").strip().lower()
    if model_type not in ("tree", "logistic", "random_forest", "svm"):
        model_type = "tree"

    df_for_clf = agent_state.get("preprocessed_dataframe")
    if df_for_clf is None:
        df_for_clf = agent_state["dataframe"]
    summary, details = run_classification(df_for_clf, target_column=target_col, model_type=model_type)
    agent_state["classification_results"] = details
    agent_state["classification_summary"] = summary
    return f"[CLASSIFICATION ({model_type})]\n{summary}"

def configure_and_extract_patterns(config: dict) -> str:
    df = agent_state.get("dataframe")
    fp_config = config.get("frequent_patterns", {})
    method = fp_config.get("method", "fpgrowth").strip().lower()
    if method not in ("apriori", "fpgrowth"):
        method = "fpgrowth"

    df_for_fp = agent_state.get("preprocessed_dataframe", df)
    summary, details = extract_patterns(df_for_fp, method=method)
    agent_state["patterns_summary"] = summary
    agent_state["patterns_details"] = details
    results = f"[FREQUENT PATTERNS ({method})]\n{summary}"
    return results

def generate_report(target_col: str, tasks: list) -> str:
    os.makedirs("/tmp/outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/tmp/outputs/report_{timestamp}.txt"

    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("DATA MINING REPORT")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: {agent_state.get('dataset_name', 'Unknown')}")
    lines.append(f"Target column: {target_col}")
    lines.append(f"Tasks performed: {', '.join(tasks)}")
    lines.append(sep)

    # Dataset summary
    info = agent_state.get("dataset_info")
    if info:
        lines.append("\n[DATASET SUMMARY]")
        lines.append(f"  Rows    : {info['rows']}")
        lines.append(f"  Columns : {info['columns']}")
        lines.append(f"  Column names: {', '.join(info['column_names'])}")
        lines.append(f"  Numeric : {', '.join(info['numeric_columns'])}")
        lines.append(f"  Categorical: {', '.join(info['categorical_columns'])}")
        missing = {k: v for k, v in info["missing_values"].items() if v > 0}
        if missing:
            lines.append(f"  Missing values: {missing}")
        else:
            lines.append("  Missing values: None")
        lines.append("  Column details:")
        for col_info in info["column_info"].items():
            col, details = col_info
            lines.append(f"    - {col} (dtype: {details['dtype']}, unique: {details['unique_values']}, sample: {details['sample_values']})")

    # Preprocessing
    pre_result = agent_state.get("preprocessing_result")
    if pre_result:
        lines.append(f"\n[PREPROCESSING]\n{pre_result}")

    # Classification
    clf_details = agent_state.get("classification_results")
    if clf_details:
        lines += [
            "",
            f"[CLASSIFICATION — {clf_details.get('model_name', '')}]",
            f"  Train samples : {clf_details.get('train_samples')}",
            f"  Test samples  : {clf_details.get('test_samples')}",
            f"  Accuracy      : {clf_details.get('accuracy', 0):.4f} "
            f"({clf_details.get('accuracy', 0)*100:.2f}%)",
            "",
            "Detailed Classification Report:",
            clf_details.get("classification_report_str", ""),
            clf_details.get("confusion_matrix_str", ""),
        ]
        top_features = clf_details.get("top_features", {})
        if top_features:
            lines.append("\nTop 5 Most Important Features:")
            for feat, score in top_features.items():
                lines.append(f"  {feat}: {score:.4f}")

    # Frequent patterns
    patterns = agent_state.get("patterns_details")
    if patterns:
        lines.append(f"\n[FREQUENT PATTERNS]\n{patterns}")

    lines.append(f"\n{sep}\nEnd of report")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


# Tool 1 — Analyze_And_Load
def tool_analyze_and_load(input_json: str) -> str:
    """
    Receives a JSON with the task list decided by the model, loads the dataset,
    computes a summary, and returns it so the model can configure Tool 2.

    Expected input schema:
    {
        "tasks": ["preprocessing", "frequent_patterns", "classification", "report"]
    }
    Any subset of the four tasks is valid.
    """
    # parse input
    try:
        data = json.loads(input_json.strip())
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON input — {e}Please fix the format and try again."

    tasks = data.get("tasks", [])
    valid_tasks = {"preprocessing", "frequent_patterns", "classification", "report"}
    unknown = set(tasks) - valid_tasks
    if unknown:
        return f"ERROR: Unknown tasks: {unknown}. Valid tasks: {valid_tasks}"

    # check file availability
    file_path = agent_state.get("_file_path")  # injected by build_executor
    if not file_path or not os.path.exists(file_path):
        return (
            "ERROR: No dataset file is available. "
            "Please ask the user to upload a CSV file using the sidebar."
        )

    # load dataset
    df, msg = load_dataset(file_path)
    if df is None:
        return f"ERROR: Could not load file — {msg}"

    # reset state
    agent_state.update({
        "dataset_name": os.path.basename(file_path),
        "dataframe": df,
        "dataset_info": None,
        "tasks": tasks,
        "preprocessing_plan": None,
        "preprocessed_dataframe": None,
        "preprocessing_result": None,
        "patterns_summary": None,
        "patterns_details": None,
        "classification_summary": None,
        "classification_results": None,
        "report_path": None,
    })

    # build summary
    summary = summarize_data()

    return (
        f"Dataset loaded successfully: {summary['rows']} rows, {summary['columns']} columns.\n"
        f"Tasks to perform: {tasks}\n"
        f"NEXT STEP: Call Execute_Plan with the full configuration JSON.\n\n"
        f"DATASET SUMMARY:\n{json.dumps(summary, ensure_ascii=False)}"
    )



# Tool 2 — Execute_Plan
def tool_execute_plan(config_json: str) -> str:
    """
    Receives the full configuration from the model and executes all selected tasks
    in order: preprocessing → classification → frequent_patterns → report.

    Expected input schema (only include keys for tasks that were selected):
    {
        "target_column": "y",
        "preprocessing": {
            "handle_missing_values": {"apply": true, "strategy": "median"},
            "remove_duplicates":     {"apply": true},
            "encode_categoricals":   {"apply": true, "method": "label"},
            "handle_outliers":       {"apply": false, "method": "iqr", "columns": []},
            "scale_numericals":      {"apply": true, "method": "standard", "columns": ["age"]}
        },
        "classification": {
            "model_type": "tree"
        },
        "frequent_patterns": {
            "method": "fpgrowth"
        }
    }
    """
    # parse config
    config_json = config_json.strip()
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON config — {e}. Please fix and retry."

    df = agent_state.get("dataframe")
    if df is None:
        return "ERROR: No dataset loaded. Please call Analyze_And_Load first."

    tasks = agent_state.get("tasks", [])
    target_col = config.get("target_column") or agent_state["dataset_info"]["target_hint"]
    agent_state["dataset_info"]["target_hint"] = target_col

    results = []

    # PREPROCESSING
    needs_preprocessing = (
        "preprocessing" in tasks
        or "classification" in tasks
        or "frequent_patterns" in tasks
    )

    if needs_preprocessing:
        pre_msg = configure_and_preprocess(config)
        results.append(pre_msg)

    # CLASSIFICATION
    if "classification" in tasks:
        cls_msg = configure_and_classify(config)
        results.append(cls_msg)


    # FREQUENT PATTERNS
    if "frequent_patterns" in tasks:
        frq_pttr_msg = configure_and_extract_patterns(config)
        results.append(frq_pttr_msg)


    # REPORT                                
    if "report" in tasks:
        report_path = generate_report(target_col, tasks)
        agent_state["report_path"] = report_path
        results.append(f"[REPORT]\nReport saved to: {report_path}")


    # Final summary back to model
    combined = "\n\n".join(results)
    if agent_state.get("report_path"):
        combined += f"\n\nREPORT FILE: {agent_state['report_path']}"
 
    return combined



# LangChain Tool definitions
tools = [
    Tool(
        name="Analyze_And_Load",
        func=tool_analyze_and_load,
        description=(
            "Use this tool ONLY for dataset-related requests. "
            "Input: a JSON string with the list of tasks to perform. "
            'Schema: {"tasks": ["preprocessing", "frequent_patterns", "classification", "report"]} '
            "Include only the tasks the user actually requested. "
            "This tool loads the dataset and returns a summary. "
            "Must be called BEFORE Execute_Plan."
        ),
    ),
    Tool(
        name="Execute_Plan",
        func=tool_execute_plan,
        description=(
            "Executes all selected tasks using the configuration you provide. "
            "Must be called AFTER Analyze_And_Load. "
            "Input: a JSON string with configuration for each selected task. "
            "Schema: "
            '{"target_column": "col_name", '
            '"preprocessing": {"handle_missing_values": {"apply": bool, "strategy": "median"|"mode"}, '
            '"remove_duplicates": {"apply": bool}, '
            '"encode_categoricals": {"apply": bool, "method": "label"|"onehot"}, '
            '"handle_outliers": {"apply": bool, "method": "iqr"|"zscore", "columns": [...]}, '
            '"scale_numericals": {"apply": bool, "method": "standard"|"minmax", "columns": [...]}}, '
            '"classification": {"model_type": "tree"|"logistic"|"random_forest"|"svm"}, '
            '"frequent_patterns": {"method": "apriori"|"fpgrowth"} '
            '} '
            "Only include keys for tasks that were selected in Analyze_And_Load."
        ),
    ),
]


# Prompt template
PROMPT_TEMPLATE = """You are an autonomous Data Mining Agent that helps users analyse datasets.

You have access to these tools:
{tools}

FILE STATUS: FILE_STATUS_PLACEHOLDER

RULES:
1. For greetings, general questions, or anything unrelated to dataset analysis:
   Answer directly as Final Answer. Do NOT call any tool.

2. FILE_INSTRUCTION_PLACEHOLDER

3. Respond in the same language as the user.

4. For dataset-related requests, follow this TWO-STEP sequence STRICTLY:

   STEP 1 — Call Analyze_And_Load with the tasks the user requested.
   Valid tasks: "preprocessing", "frequent_patterns", "classification", "report"
   Only include tasks the user explicitly asked for (or that are clearly implied).
   Note: "preprocessing" is always implied if classification or frequent_patterns are requested.

   STEP 2 — After receiving the dataset summary, call Execute_Plan with the full
   configuration for each selected task. Use the dataset summary to decide the best
   settings (e.g. which columns to scale, which encoding method, target column, etc.)

5. Never invent or guess file paths. Use only the path shown in FILE STATUS.

6. Never skip steps or give a Final Answer before Execute_Plan has returned.

Use EXACTLY this format:

Thought: <your reasoning>
Action: <one of [{tool_names}]>
Action Input: <input to the tool>
Observation: <tool result>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: <your response to the user>

Begin!

Question: {input}
Thought: {agent_scratchpad}"""



# Agent factory

def build_executor(file_path: str) -> AgentExecutor:
    """
    Builds and returns a new AgentExecutor.
    Called by app.py whenever the uploaded file changes.
    """
    # Inject file_path into agent_state so Tool 1 can access it
    agent_state["_file_path"] = file_path or ""

    llm = ChatOpenAI(
        api_key=os.getenv("METIS_AI_API_KEY", ""),
        base_url=os.getenv("METIS_AI_BASE_URL", "https://api.metisai.ir/openai/v1"),
        model=os.getenv("METIS_MODEL_NAME", "gpt-4.1-nano"),
        temperature=0,
    )

    if file_path:
        status = f"A dataset file IS available at: {file_path}"
        instruction = (
            "A dataset is ready. For any dataset-related request, "
            "call Analyze_And_Load first with the appropriate tasks."
        )
    else:
        status = "No dataset has been uploaded yet."
        instruction = (
            "No dataset is available. If the user asks for dataset analysis, "
            "tell them to upload a CSV file using the sidebar. Do NOT call any tool."
        )

    filled = (
        PROMPT_TEMPLATE
        .replace("FILE_STATUS_PLACEHOLDER", status)
        .replace("FILE_INSTRUCTION_PLACEHOLDER", instruction)
    )

    prompt = PromptTemplate.from_template(filled)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=15,
        early_stopping_method="generate",
    )