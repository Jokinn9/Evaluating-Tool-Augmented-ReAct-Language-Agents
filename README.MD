# Evaluating Tool-Augmented ReAct Language Agents

This repository contains the code, datasets and evaluation results for the MSc thesis **"Evaluating Tool-Augmented ReAct Language Agents"**, developed at the University of Barcelona for the *Fundamental Principles of Data Science* Master's programme.

## Project Overview

This project explores how to evaluate ReAct agents that interact with external tools such as Wikipedia, Wikidata, Yahoo Finance and a local PDF. The agents are built using open-source frameworks like **LangChain** and **LangGraph**, and the language models are served locally with **Ollama**, removing the need for commercial APIs.

Three different agents were implemented:

- **Agent 1**: A baseline ReAct agent that uses fake tools for metal prices and currency conversion.
- **Agent 2**: A factual agent using Wikipedia and Wikidata APIs to answer questions about people and political leaders.
- **Agent 3**: A metal-specialised agent that integrates information from a local PDF, Yahoo Finance and Wikipedia.

All evaluations are performed with custom rule-based checks and **RAGAS** metrics.

## Folder Structure

The project is divided into three main sections:

- `Agent 1 Baseline`: Basic ReAct agent with fake tools.
- `Agent 2 Wikipedia`: Agent that uses Wikipedia and Wikidata APIs.
- `Agent 3 with PDF`: Agent that combines PDF parsing, live prices, and encyclopedic knowledge.

Each folder contains:

- `agent_v*.ipynb`: Notebooks for testing and debugging.
- `ploting.ipynb`: Notebook for plotting evaluation results.
- `utils.py`: Utility functions for tool setup, message formatting, etc.
- `qa_dataset.json`: Synthetic questions for evaluation.
- `Results/`: Folder where all evaluation outputs are stored.
- `Images/`: Generated figures (e.g., tool accuracy, answer relevancy).

## Evaluation Metrics

Each agent is evaluated using two categories of metrics:

### Rule-Based Metrics

- **ToolCall Accuracy**: Exact match between expected and predicted tool calls.
- **Substring Match**: Checks if required keywords appear in the response.
- **Value Accuracy**: Compares numeric outputs to expected values within a tolerance.
- **Execution Trace Info**: Includes number of steps and total runtime.

### RAGAS Metrics

- **ToolCallAccuracy (Multi-turn)**: Verifies correct tool use at any point during reasoning.
- **Faithfulness**: Checks whether the response is supported by retrieved context.
- **Answer Relevancy**: Measures semantic alignment between question and response.
- **Context Precision**: Assesses whether the retrieved chunk is relevant to the query.

All RAGAS evaluations use:

- Language model: `Mistral` via Ollama  
- Embeddings: `all-MiniLM-L6-v2` via HuggingFace

## Requirements

Install the dependencies in the requirements.yml

## How to Run

1. Clone the repo and install dependencies.
2. Choose an agent folder and open its corresponding notebook or script.
3. Run the evaluation:

   * `agent_v*.ipynb` for full pipeline
   * `ploting.ipynb` for visual analysis
4. Results will be saved under the `Results/` folder in each agent directory.

## Author

**Jokin Eguzkitza**
MSc in Data Science – University of Barcelona
[LinkedIn](https://www.linkedin.com/in/jokin-eguzkitza/)
