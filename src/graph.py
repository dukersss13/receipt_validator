from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from enum import Enum
import pandas as pd

# Assume these are already implemented and imported
from data.data_reader import DataReader, DataType
from validator import Validator
from utils.currency_conversion_agent import convert_currency_to_usd


class Stage(str, Enum):
    INIT_READER = "init_reader"
    INIT_VALIDATOR = "init_validator"
    TRANSACTIONS = "transactions"
    PROOFS = "proofs"
    DETECT_CURRENCY = "detect_currency"
    CONVERT_CURRENCY = "convert_currency"
    VALIDATE = "validate"
    ANALYZE_RESULTS = "analyze_results"
    

class GraphState(TypedDict):
    reader: DataReader
    transaction_df: pd.DataFrame
    proof_df: pd.DataFrame


# --- Node Functions ---
# Note: The return type hint is changed to `dict` for accuracy.
def init_reader(state: GraphState) -> dict:
    """
    Initializes the DataReader.
    """
    print("---Initializing Data Reader---")

    return {"reader": DataReader()}

def init_validator(state: GraphState) -> dict:
    """
    Initializes the Validator.
    """
    print("---Initializing Validator---")
    # Assuming Validator initialization does not require any parameters
    return {"validator": Validator(state["transaction_df"], state["proof_df"])}

def load_transactions(state: GraphState) -> dict:
    """
    Loads the transactions data into a DataFrame.
    """
    print("---Loading Transactions---")
    df = state["reader"].load_data(DataType.TRANSACTIONS)

    return {"transaction_df": df}

def load_proofs(state: GraphState) -> dict:
    """
    Loads the proofs data into a DataFrame.
    """
    print("---Loading Proofs---")
    df = state["reader"].load_data(DataType.PROOFS)

    return {"proof_df": df}

def detect_foreign_currency(state: GraphState) -> dict:
    """
    Detects the currency for each row in the proof_df.
    """
    print("---Detecting Currencies---")
    df = state["proof_df"].copy() # Use a copy to avoid mutation side-effects
    df["foreign_currency"] = df["currency"] != "USD"

    return {"proof_df": df}

def convert_currency(state: GraphState) -> dict:
    """
    Converts amounts to USD where the currency is not USD.
    """
    print("---Converting Currencies to USD---")
    df = state["proof_df"].copy()
    foreign_currency = df[df["foreign_currency"] == True]
    usd = df[df["foreign_currency"] == False]
    foreign_currency["total"] = df.apply(
        lambda row: convert_currency_to_usd(row),
        axis=1,
    )

    df = pd.concat([usd, foreign_currency], axis=0, ignore_index=True)

    return {"proof_df": df}    

# --- Conditional Edge Logic ---
def should_convert_currency(state: GraphState) -> Literal["convert", "skip"]:
    """
    Determines if any currency conversion is necessary.
    """
    print("---Checking if conversion is needed---")
    proof_df = state["proof_df"]
    # Check if a 'currency' column exists and if any currency is not 'USD'
    if "currency" in proof_df.columns and (proof_df["foreign_currency"]).any():
        print("Foreign currency detected. Proceeding to conversion.")
        return "convert"
    else:
        print("No foreign currency detected. Skipping conversion.")
        return "skip"

def validate(state: GraphState) -> dict:
    """
    Validates the transactions and proofs.
    """
    print("---Validating Transactions and Proofs---")
    validator: Validator = state["validator"]
    validation_results = validator.validate()

    return {"results": validation_results}

def analyze_results(state: GraphState) -> dict:
    """
    Analyzes the validation results.
    """
    print("---Analyzing Validation Results---")
    results = state["results"]
    analysis, recommendations = state["validator"].analyze_results(results)
    
    return {"analysis": analysis, "recommendations": recommendations}


# --- Graph Construction ---
graph = StateGraph(GraphState)

# Add nodes
graph.add_node(Stage.INIT_READER, init_reader)
graph.add_node(Stage.TRANSACTIONS, load_transactions)
graph.add_node(Stage.PROOFS, load_proofs)

graph.add_node(Stage.DETECT_CURRENCY, detect_foreign_currency)
graph.add_node(Stage.CONVERT_CURRENCY, convert_currency)

graph.add_node(Stage.INIT_VALIDATOR, init_validator)

graph.add_node(Stage.VALIDATE, validate)
graph.add_node(Stage.ANALYZE_RESULTS, analyze_results)


# Set entry point
graph.set_entry_point(Stage.INIT_READER)

# Add standard edges
graph.add_edge(Stage.INIT_READER, Stage.TRANSACTIONS)
graph.add_edge(Stage.INIT_READER, Stage.PROOFS)
graph.add_edge(Stage.PROOFS, Stage.DETECT_CURRENCY)

# Add a conditional edge
graph.add_conditional_edges(
    Stage.DETECT_CURRENCY,
    should_convert_currency,
    {
        "convert": Stage.CONVERT_CURRENCY,
        "skip": Stage.INIT_VALIDATOR,
    },
)

graph.add_edge(Stage.TRANSACTIONS, Stage.INIT_VALIDATOR)
graph.add_edge(Stage.CONVERT_CURRENCY, Stage.INIT_VALIDATOR)
graph.add_edge(Stage.INIT_VALIDATOR, Stage.VALIDATE)
graph.add_edge(Stage.VALIDATE, Stage.ANALYZE_RESULTS)
graph.add_edge(Stage.ANALYZE_RESULTS, END)

# Compile the graph
app = graph.compile()


def run_graph():
    """Initializes and runs the graph, returning the final state."""
    # The initial state can be empty; the graph will populate it.
    initial_state = {}
    # Use invoke() instead of the deprecated run()
    final_state = app.invoke(initial_state)

    return final_state


graph_png = app.get_graph().draw_mermaid_png()

# 2. Define a filename
filename = "graph_visualization.png"

# 3. Write the image data to a file
with open(filename, "wb") as f:
    f.write(graph_png)
