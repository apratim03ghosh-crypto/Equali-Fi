import os
import asyncio
import json
import re
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 1. Environment Setup
# Ensures the .env file is found regardless of where the terminal starts
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Initialize OpenRouter Client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# -----------------------------
# ⚖️ THE JUDGE PROMPT (STRICT)
# -----------------------------
JUDGE_PROMPT = """
You are an impartial AI Judge. Evaluate accuracy, neutrality, and clarity. 
You MUST return ONLY a JSON object. No markdown, no conversational text.

Required JSON Structure:
{
  "scores": {"AI_1": 0, "AI_2": 0, "AI_3": 0},
  "best_ai": "AI_1",
  "reasoning": "Brief explanation of why this model won.",
  "final_answer": "The refined, most neutral response possible."
}
"""

# -----------------------------
# 🤖 API CALL FUNCTION
# -----------------------------
async def get_model_response(model_name, message_history):
    """Fires a request to a single model with a 30s safety timeout."""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=message_history,
            timeout=30.0,
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Equali-Fi Governance Engine",
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        # Return a structured error so the Judge knows the model failed
        return f"SYSTEM_OFFLINE: {model_name} failed. Error: {str(e)}"

# -----------------------------
# ⚖️ MAIN ORCHESTRATOR
# -----------------------------
async def run_consensus(message_history, models):
    """
    1. Parallel multi-model calls.
    2. Arbitration via Judge AI.
    3. JSON extraction and mapping.
    """

    # STEP 1: Gather responses from all active AIs in parallel
    tasks = [get_model_response(model, message_history) for model in models]
    results = await asyncio.gather(*tasks)

    # STEP 2: Map results and create clean display names for the chart
    resp_map = {}
    display_names = {}
    for i, res in enumerate(results):
        ai_id = f"AI_{i+1}"
        resp_map[ai_id] = res
        # Strip the provider name (e.g., 'google/gemini-pro' becomes 'gemini-pro')
        display_names[ai_id] = models[i].split('/')[-1]

    # STEP 3: Build Input for the Judge
    user_query = message_history[-1]["content"]
    judge_input = f"User Query: {user_query}\n\n"
    for ai_id, content in resp_map.items():
        judge_input += f"--- {ai_id} ({display_names[ai_id]}) ---\n{content}\n\n"

    # STEP 4: Call the Arbitrator (Using gpt-4o-mini for speed and reliability)
    judge_messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": judge_input}
    ]
    
    judge_raw = await get_model_response("openai/gpt-4o-mini", judge_messages)

    # STEP 5: Super-Clean JSON Extraction
    # This prevents the 'Char 0' or 'Double Quote' errors by hunting for { } brackets
    try:
        # Find the first { and last } to strip away AI conversational filler
        json_match = re.search(r"\{.*\}", judge_raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No valid JSON found in Judge response.")
    except Exception as e:
        # Fallback Logic: Keeps the app running even if the Judge glitches
        data = {
            "scores": {f"AI_{i+1}": 5 for i in range(len(models))},
            "best_ai": "AI_1",
            "reasoning": f"Judge Parsing Error: {str(e)}",
            "final_answer": results[0] if "SYSTEM_OFFLINE" not in results[0] else "Critical: Model failure."
        }

    # STEP 6: Final Data Mapping for the Bar Chart
    # This converts "AI_1" scores to "gemini-pro" scores so the chart is readable
    final_scores = {display_names[k]: v for k, v in data["scores"].items() if k in display_names}

    return {
        "final_answer": data.get("final_answer"),
        "scores": final_scores,
        "best_ai_name": display_names.get(data.get("best_ai"), "Unknown"),
        "reasoning": data.get("reasoning"),
        "raw_responses": resp_map,
        "model_names": display_names
    }