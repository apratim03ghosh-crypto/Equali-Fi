import os
import asyncio
import json
import re
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

MODEL_NAME_MAP = {
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "openai/gpt-4o-mini": "GPT-4o Mini",
    "mistralai/mixtral-8x7b-instruct": "Mistral Mixtral",
    "google/gemma-2-9b-it": "Google Gemma 2"
}

# --- ENV SETUP ---
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# --- 🔥 IMPROVED JUDGE PROMPT ---
JUDGE_PROMPT = """
You are an expert AI evaluator.

Evaluate responses based on:
- Accuracy
- Clarity
- Completeness
- Neutrality

You MUST return ONLY valid JSON.

Required format:
{
  "scores": {"AI_1": 0, "AI_2": 0, "AI_3": 0},
  "best_ai": "AI_1",
  "reasoning": "Why this AI is best",
  "key_differences": "Key differences between responses",
  "final_answer": "Best refined answer"
}
"""

# --- API CALL ---
async def get_model_response(model_name, message_history):
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=message_history,
            timeout=18.0,
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Equali-Fi Governance Engine",
            }
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"SYSTEM_OFFLINE: {model_name} failed. Error: {str(e)}"

# --- MAIN ORCHESTRATOR ---
async def run_consensus(message_history, models):

    # --- STEP 1: PARALLEL CALLS ---
    tasks = [get_model_response(model, message_history) for model in models]
    results = await asyncio.gather(*tasks)

    # --- STEP 2: MAP RESPONSES ---
    resp_map = {}
    display_names = {}

    for i, res in enumerate(results):
        ai_id = f"AI_{i+1}"

        clean_name = MODEL_NAME_MAP.get(models[i], models[i].split('/')[-1])

        resp_map[ai_id] = res
        display_names[ai_id] = clean_name

    if len(models) == 1:
        only_model = display_names.get("AI_1", "Unknown")
        only_response = results[0] if results else "System failure"
        score = 0 if "SYSTEM_OFFLINE" in only_response else 10

        return {
            "final_answer": only_response,
            "scores": {only_model: score},
            "best_ai_name": only_model,
            "reasoning": "Only one model was selected, so its response is used directly.",
            "key_differences": "No comparison available with a single active model.",
            "responses": {only_model: only_response},
        }

    # --- STEP 3: JUDGE INPUT ---
    user_query = message_history[-1]["content"]

    judge_input = f"User Query: {user_query}\n\n"

    for ai_id, content in resp_map.items():
        judge_input += f"{ai_id} ({display_names[ai_id]}):\n{content}\n\n"

    # --- STEP 4: JUDGE CALL ---
    judge_messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": judge_input}
    ]

    judge_raw = await get_model_response("openai/gpt-4o-mini", judge_messages)

    # --- STEP 5: JSON EXTRACTION ---
    try:
        json_match = re.search(r"\{.*\}", judge_raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        data = {
            "scores": {f"AI_{i+1}": 5 for i in range(len(models))},
            "best_ai": "AI_1",
            "reasoning": f"Judge parsing failed: {str(e)}",
            "key_differences": "Unable to compute differences",
            "final_answer": results[0] if results else "System failure"
        }

    # --- STEP 6: HANDLE FAILURES SMARTLY ---
    for i, res in enumerate(results):
        if "SYSTEM_OFFLINE" in res:
            data["scores"][f"AI_{i+1}"] = 0

    # --- STEP 7: FINAL FORMATTING ---
    final_scores = {
        display_names[k]: v
        for k, v in data["scores"].items()
        if k in display_names
    }

    # 🔥 THIS FIXES YOUR FRONTEND
    formatted_responses = {
        display_names[k]: v
        for k, v in resp_map.items()
    }

    return {
        "final_answer": data.get("final_answer"),
        "scores": final_scores,
        "best_ai_name": display_names.get(data.get("best_ai"), "Unknown"),
        "reasoning": data.get("reasoning"),
        "key_differences": data.get("key_differences"),
        "responses": formatted_responses,  # 🔥 IMPORTANT
    }
