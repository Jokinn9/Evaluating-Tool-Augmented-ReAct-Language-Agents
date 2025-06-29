from copy import deepcopy
import json
import uuid
from langchain.schema.messages import AIMessage,ToolMessage,HumanMessage
from ragas.dataset_schema import SingleTurnSample,MultiTurnSample
import time
import re
from pathlib import Path
from typing import List, Dict
from ragas.metrics import ToolCallAccuracy
from ragas.integrations.langgraph import convert_to_ragas_messages
import ragas.messages as r
import asyncio

def normalize_args(args):
    """
    Converts all keys and values in the argument dictionary to lowercase strings.

    Args:
        args (dict): Dictionary of arguments with string keys and any type of values.

    Returns:
        dict: A new dictionary with all keys and values converted to lowercase strings.
    """
    return {k.lower(): str(v).lower() for k, v in args.items()}


def fix_tool_calls_for_openai_format(messages):
    """
    Reformats a list of LangChain messages into a format compatible with OpenAI-style tool call APIs.
    Handles multiple tool calls by splitting them into separate AI messages and normalising their arguments.

    Args:
        messages (List[BaseMessage]): A list of LangChain messages, which may include HumanMessage, AIMessage, and ToolMessage.

    Returns:
        List[BaseMessage]: A new list of messages, reformatted to match OpenAI-compatible tool call format.
    """
    final_messages = [] 
    tool_message_buffer = {}  
    used_tool_call_ids = set() 

    # Store ToolMessages for easy access later
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_message_buffer[msg.tool_call_id] = msg

    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Add human messages directly            
            final_messages.append(msg)

        elif isinstance(msg, AIMessage) and msg.tool_calls and len(msg.tool_calls) > 1:
            # If there are multiple tool calls, split them into separate messages
            for tool_call in msg.tool_calls:
                norm_args = normalize_args(tool_call["args"])

                new_msg = deepcopy(msg)
                new_msg.tool_calls = [{
                    "name": tool_call["name"],
                    "args": norm_args,
                    "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "tool_call"
                }]
                new_msg.additional_kwargs["tool_calls"] = [{
                    "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(norm_args)
                    }
                }]
                final_messages.append(new_msg)

                # Append the corresponding ToolMessage if available
                tool_msg = tool_message_buffer.get(tool_call["id"])
                if tool_msg:
                    final_messages.append(tool_msg)
                    used_tool_call_ids.add(tool_call["id"])

        elif isinstance(msg, AIMessage) and msg.tool_calls:
            # Handle single tool call
            tool_call = msg.tool_calls[0]
            norm_args = normalize_args(tool_call["args"])

            msg.tool_calls[0]["args"] = norm_args
            msg.additional_kwargs["tool_calls"] = [{
                "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(norm_args)
                }
            }]
            final_messages.append(msg)

        elif isinstance(msg, AIMessage):
           # Standard AI messages without tool calls
           final_messages.append(msg)

        elif isinstance(msg, ToolMessage):
            # Add tool messages that haven't been used yet
            if msg.tool_call_id not in used_tool_call_ids:
                final_messages.append(msg)

    return final_messages


def lc_to_ragas_sample(lc_msgs) -> SingleTurnSample:
    """
    Converts a list of LangChain messages into a RAGAS-compatible SingleTurnSample.

    Args:
        lc_msgs (List[BaseMessage]): Sequence of LangChain messages, typically including HumanMessage, AIMessage, and ToolMessage.

    Returns:
        SingleTurnSample: Object containing the user input, response, and retrieved contexts, suitable for RAGAS evaluation.
    """
    question = str(next(m.content for m in lc_msgs if isinstance(m, HumanMessage)))

    answer = str(next(
        m.content for m in reversed(lc_msgs)
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
    ))

    contexts = [str(m.content) for m in lc_msgs if isinstance(m, ToolMessage)]

    return SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )


async def tool_evaluation(react_graph, dataset: List[Dict], output_path: str = "tool.json"):
    """
    Evaluates an agent's tool usage using RAGAS ToolCallAccuracy over a set of input questions.

    Args:
        react_graph: The agent (e.g. LangGraph or LangChain runnable) used to answer the questions.
        dataset (List[Dict]): A list of dictionaries, each containing a "question" and a list of "expected_tool_calls".
        output_path (str): File path where the evaluation results will be saved as JSON. Default is "tool.json".

    Returns:
        List[Dict]: A list of results for each sample, including the RAGAS ToolCallAccuracy score and metadata.
    """
    results = []
    for sample in dataset:
        question = sample["question"]
        expected_tool_calls = sample["expected_tool_calls"]
        result = react_graph.invoke({"messages": [HumanMessage(content=question)]})

        messages = result["messages"]
        messages = fix_tool_calls_for_openai_format(result["messages"])
        tool_calls_detected = []

        # RAGAS ToolCallAccuracy
        ragas_trace = convert_to_ragas_messages(messages)
        try:
            sample_ragas = MultiTurnSample(
                user_input=ragas_trace,
                reference_tool_calls=[r.ToolCall(**tc) for tc in expected_tool_calls]
            )
            ragas_score = await ToolCallAccuracy().multi_turn_ascore(sample_ragas)
        except Exception as e:
            ragas_score = None

        # Register
        results.append({
            "question": question,
            "expected_tool_calls": expected_tool_calls,
            "tool_calls_detected": tool_calls_detected,
            "ragas_tool_accuracy": ragas_score,
            "num_steps": len(messages),
        })

    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"Saved evaluation results to {output_path}")
    return results

async def evaluate(sample,metrics):
    """
    Evaluates a SingleTurnSample using all defined RAGAS metrics asynchronously.

    Args:
        sample (SingleTurnSample): The input sample containing user input, response, and retrieved contexts.
        metrics (Dict[str, ragas.metrics.BaseMetric]): Dictionary of metric names and their corresponding metric objects.

    Returns:
        Dict: Dictionary mapping metric names to their computed scores for the given sample.
    """
    # Launch one async task per metric
    tasks = {name: metric.single_turn_ascore(sample) for name, metric in metrics.items()}
    
    # Await all tasks and collect results
    results = {name: await coro for name, coro in tasks.items()}
    
    return results


async def evaluate_all_safe(samples,metrics):
    """
    Evaluates a list of SingleTurnSample objects using defined RAGAS metrics, handling errors gracefully.

    Args:
        samples (List[SingleTurnSample]): List of RAGAS-compatible samples to be evaluated.
        metrics (Dict[str, ragas.metrics.BaseMetric]): Dictionary of metric names and their corresponding metric objects.

    Returns:
        List[Dict]: A list of dictionaries with metric scores and additional metadata per sample.
    """
    partial_results = []
    for i, sample in enumerate(samples):
        try:
            # Run async evaluation on each sample
            scores = await evaluate(sample,metrics)
            # Add original input data for traceability
            scores["user_input"] = sample.user_input
            scores["retrieved_contexts"] = sample.retrieved_contexts
            scores["response"] = sample.response
            partial_results.append(scores)
        except Exception as e:
            # Continue even if one sample fails
            print(f"Error en sample {i}: {sample.user_input}\n→ {e}")
            continue 

    return partial_results
