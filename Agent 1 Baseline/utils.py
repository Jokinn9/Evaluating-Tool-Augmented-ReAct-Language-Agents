import time
import json
import re
from pathlib import Path
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage,ToolMessage
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
import ragas.messages as r
import asyncio
from copy import deepcopy
import uuid

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


async def evaluate_agent_with_ragas(react_graph, dataset: List[Dict], output_path: str = ""):
    """
    Evaluates an agent using a dataset of input questions and compares its tool usage and responses
    against expected values. Also computes RAGAS ToolCallAccuracy.

    Args:
        react_graph: The agent to evaluate, typically a LangGraph or LangChain runnable.
        dataset (List[Dict]): List of dictionaries. Each entry must contain:
            - "question" (str): Input query to send to the agent.
            - "expected_tool_calls" (List[Dict]): Tool calls that should have been made.
            - Optional "expected_substring" (str): Text expected in the final response.
            - Optional "expected_value" (float): Numeric value to check in the response.
        output_path (str): Path to save the results in JSON format. Defaults to "" (no save).

    Returns:
        List[Dict]: Evaluation results with fields including tool accuracy, RAGAS score, and metadata.
    """
    results = []

    for sample in dataset:
        # Extract question and expected results for this sample
        question = sample["question"]
        expected_tool_calls = sample["expected_tool_calls"]
        expected_substring = sample.get("expected_substring", "")
        expected_value = sample.get("expected_value", None)

        # Run the agent on the question and time the execution
        start_time = time.time()
        result = react_graph.invoke({"messages": [HumanMessage(content=question)]})
        end_time = time.time()

        # Postprocess messages for OpenAI-style tool call format
        messages = fix_tool_calls_for_openai_format(result["messages"])

        # Collect all tool calls detected in the agent's response
        tool_calls_detected = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_detected.append({"name": tc["name"], "args": tc["args"]})

        # Get the final text response
        final_response = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

        # Compare expected tool calls with detected ones
        matched_calls = sum(
            1 for expected in expected_tool_calls
            if any(
                expected["name"] == detected["name"]
                and normalize_args(expected["args"]) == normalize_args(detected["args"])
                for detected in tool_calls_detected
            )
        )
        tool_accuracy = matched_calls / len(expected_tool_calls) if expected_tool_calls else 1.0

        # Check if expected text is in the response
        output_match = expected_substring.lower() in final_response.lower() if expected_substring else True

        # If applicable, check if the response includes a numeric value close to expected
        value_correct = None
        if expected_value is not None:
            try:
                numbers = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", final_response)]
                value_correct = any(abs(num - expected_value) < 0.001 for num in numbers)
            except:
                value_correct = False

        # Compute RAGAS ToolCallAccuracy score
        ragas_trace = convert_to_ragas_messages(messages)
        try:
            sample_ragas = MultiTurnSample(
                user_input=ragas_trace,
                reference_tool_calls=[r.ToolCall(**tc) for tc in expected_tool_calls]
            )
            ragas_score = await ToolCallAccuracy().multi_turn_ascore(sample_ragas)
        except Exception as e:
            ragas_score = None

        # Save metrics for this question
        results.append({
            "question": question,
            "expected_tool_calls": expected_tool_calls,
            "tool_calls_detected": tool_calls_detected,
            "tool_accuracy": tool_accuracy,
            "ragas_tool_accuracy": ragas_score,
            "output_contains_expected": output_match,
            "value_correct": value_correct,
            "final_response": final_response,
            "num_steps": len(messages),
            "execution_time_sec": round(end_time - start_time, 2)
        })

    # Save all results to a file
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"Saved evaluation results to {output_path}")
    return results
