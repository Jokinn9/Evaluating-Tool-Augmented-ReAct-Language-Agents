# Evaluating Tool-Augmented ReAct Language Agents

This repository contains the code, data and evaluation setup for the MSc thesis **"Evaluating Tool-Augmented ReAct Language Agents"** developed at the University of Barcelona, as part of the *Fundamental Principles of Data Science* Master's programme.

## Project Overview

This thesis explores how to evaluate ReAct agents that interact with external tools like Wikipedia, Wikidata, Yahoo Finance or PDF readers. The agents are implemented using open-source frameworks like **LangChain** and **LangGraph**, and all models are deployed locally using **Ollama**, avoiding commercial APIs.

Three agents were developed:

- **Agent 1**: A baseline ReAct agent using fake tools (metal prices and currency conversion).
- **Agent 2**: An agent using Wikipedia and Wikidata to answer factual questions.
- **Agent 3**: A more complex agent combining information from a PDF, Yahoo Finance and Wikipedia.

Each agent is tested using synthetic query sets tailored to its specific tools and domain.

## Folder Structure

The project is divided into three main sections:

- `Agent 1 Baseline`: Basic ReAct agent with fake tools.
- `Agent 2 Wikipedia`: Agent that uses Wikipedia and Wikidata APIs.
- `Agent 3 with PDF`: Agent that combines PDF parsing, live prices, and encyclopedic knowledge.

Each folder contains:
- `agent_v*.ipynb`: Notebooks for testing and debugging.
- `utils.py`: Utility functions for tool setup, message formatting, etc.
- `qa_dataset.json`: Synthetic questions for evaluation.
- `Results/`: Folder where all evaluation outputs are stored.

## Evaluation

Agents are evaluated using both **rule-based metrics** and **RAGAS** scores:

- **Rule-based metrics**:
  - ToolCall accuracy (exact match)
  - Substring presence
  - Numeric value match
  - Reasoning steps and execution time

- **RAGAS metrics**:
  - ToolCallAccuracy (multi-turn)
  - Faithfulness
  - Answer Relevancy
  - Context Precision

The evaluation uses a separate model (`Mistral`) and open-source embeddings (`all-MiniLM-L6-v2`) to remain fully local and reproducible.

## Requirements

Make sure to install the dependencies below before running the notebooks or Python scripts:

```bash
pip install langchain langgraph ragas sentence-transformers ollama pandas
````

Additionally, make sure to have **Ollama** installed and running with the required models:

```bash
ollama run mistral
ollama run llama3.2
```

## How to Run

1. Clone the repo and install dependencies.
2. Start Ollama with the required models.
3. Choose an agent folder and open its corresponding notebook or script.
4. Run the evaluation:

   * `evaluate_agent.py` for full pipeline
   * `ploting.ipynb` for visual analysis
5. Results will be saved under the `Results/` folder in each agent directory.

## Author

**Jokin Eguzkitza**
MSc in Data Science – University of Barcelona
[LinkedIn](https://www.linkedin.com/in/jokin-eguzkitza/)
