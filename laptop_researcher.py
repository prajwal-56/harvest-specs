import os
import sys
import json
import time
import re
import pandas as pd
# Updated import to silence the RuntimeWarning
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key from .env file if it exists
load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL           = "gemini-3.1-flash-lite"  # This model has 500 RPD left in your AI Studio!
INPUT_FILE      = "rawScrap.csv"
OUTPUT_FILE     = "laptops_research.csv"
DELAY_SECONDS   = 2  

# ─── PROMPT ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a laptop hardware expert. 
You will be provided with raw search results about a laptop. 
Your job is to extract the technical specifications and return ONLY a raw JSON object.
Absolutely no conversation, markdown, or code fences. If info is missing, use null."""

def build_extraction_prompt(laptop_name, search_data):
    return f"""
Laptop Name: {laptop_name}
Raw Search Data: 
{search_data}

Return a JSON object with these EXACT fields:
{{
  "name": "Full official model name",
  "brand": "Brand name",
  "release_year": null,
  "cpu_model": "e.g. Intel Core i7-1165G7",
  "cores": null,
  "threads": null,
  "ram_gb": null,
  "ram_upgradable": true/false,
  "ram_slots": null,
  "storage_gb": null,
  "storage_type": "SSD/HDD",
  "display_inches": null,
  "display_resolution": "e.g. 1920x1080",
  "windows_11_support": true/false,
  "typical_new_price_inr": null,
  "repairability_notes": "Short summary"
}}
"""

# ─── TOOLS: DUCKDUCKGO SEARCH ───────────────────────────────────────────────
def search_laptop_specs(laptop_name):
    """Fetches raw text from the web for free without using Gemini's search tool."""
    try:
        with DDGS() as ddgs:
            query = f"{laptop_name} technical specifications full specs"
            results = ddgs.text(query, max_results=4)
            return "\n\n".join([r['body'] for r in results])
    except Exception as e:
        print(f"  ⚠️  Search failed for {laptop_name}: {e}")
        return ""

# ─── CORE LOGIC ─────────────────────────────────────────────────────────────
def process_laptop(client, laptop_name, cashify_price):
    print(f"  🔍  Searching: {laptop_name}")
    
    # Get raw data from DuckDuckGo (Free/Unlimited)
    web_data = search_laptop_specs(laptop_name)
    if not web_data:
        return {"_input_name": laptop_name, "_status": "search_failed"}

    # Use Gemini to parse the data into JSON (Standard API quota)
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=build_extraction_prompt(laptop_name, web_data),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            data = json.loads(response.text)
            data["_input_name"] = laptop_name
            data["cashify_price"] = cashify_price
            data["_status"] = "ok"
            print(f"  ✅  Done")
            return data

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                print(f"  ⏳  Rate limit hit (429). Waiting 60 seconds before retrying...")
                time.sleep(60)
            elif "503" in err_str or "unavailable" in err_str or "overloaded" in err_str:
                print(f"  ⏳  Service unavailable (503/Overloaded). Waiting 30 seconds before retrying...")
                time.sleep(30)
            else:
                print(f"  ❌  API Error: {e}")
                return {"_input_name": laptop_name, "_status": f"error: {e}"}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file {INPUT_FILE} not found.")
        return

    # Load input with header handling
    df = pd.read_csv(INPUT_FILE)
    
    # Check for already processed items to support Resuming
    done_names = set()
    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        if "_input_name" in done_df.columns:
            done_names = set(done_df["_input_name"].tolist())

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    start_time = time.time()
    processed_count = 0
    total_remaining = len(df) - len(done_names)
    
    for index, row in df.iterrows():
        name = row['model_name']
        price = row['price']
        
        if name in done_names:
            continue
            
        processed_count += 1
        
        # Calculate Progress Bar & ETA
        percent = processed_count / total_remaining if total_remaining > 0 else 1
        bar_length = 20
        filled_len = int(bar_length * percent)
        bar = '█' * filled_len + '░' * (bar_length - filled_len)
        
        if processed_count > 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / (processed_count - 1)
            eta_seconds = avg_time * (total_remaining - processed_count + 1)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        else:
            eta_str = "Calculating..."

        print(f"\n{'='*55}")
        print(f"[{index + 1}/{len(df)}] [{bar}] {int(percent * 100)}% | ⏳ ETA: {eta_str}")
        result = process_laptop(client, name, price)
        
        # Append to CSV immediately so progress is saved
        pd.DataFrame([result]).to_csv(
            OUTPUT_FILE, 
            mode='a', 
            header=not os.path.exists(OUTPUT_FILE), 
            index=False
        )
        
        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    main()

