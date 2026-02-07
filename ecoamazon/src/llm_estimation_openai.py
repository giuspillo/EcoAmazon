import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv  
from tqdm import tqdm

load_dotenv() 

# CONFIGURATION
API_KEY = os.getenv("API_KEY") 
MODEL = "o3-mini" 
BASE_URL = "https://api.openai.com/v1"

# Inizia con 5. Se non ricevi errori 429, puoi provare ad alzare a 10.
MAX_WORKERS = 10 

# LISTINO PREZZI
PRICING = {
    "o3-mini": {"input": 1.10, "output": 4.40}
}

# File Paths
INPUT_FILE = "C:\\Users\\user\\Desktop\\Uni\\Semantics\\pcf_estimator_PG\\Clothing\\meta_Clothing_processed_only.jsonl"
#INPUT_FILE = "C:\\Users\\user\\Desktop\\Uni\\Semantics\\pcf_estimator_PG\\Clothing\\openai_clothing_test.jsonl"  # File Dati di Input test
OUTPUT_FILE = "metadata_clothing_openai.jsonl"          # File Dati (Pulito)
COST_FILE = "metadata_clothing_openai_costs.jsonl"      # File Costi (Separato)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Lucchetto per la scrittura sicura su file
file_lock = threading.Lock()

def calculate_cost(usage, model_name):
    """Calcola il costo basato sui token usati."""
    if not usage:
        return 0.0
    
    # Cerca il prezzo per il modello corrente
    price_info = None
    for key in PRICING:
        if key in model_name:
            price_info = PRICING[key]
            break
    
    if not price_info:
        return 0.0
    
    p_in = price_info["input"] / 1_000_000
    p_out = price_info["output"] / 1_000_000
    
    cost = (usage.prompt_tokens * p_in) + (usage.completion_tokens * p_out)
    return round(cost, 6)

def estimate_co2_for_product(product_data, llm_model=MODEL):
    """
    Perform estimation using OpenAI's native JSON Mode.
    Returns: (data_dict, usage_object)
    """

    prompt = f"""
    You are an expert in life cycle analysis (LCA) and CO2e emission calculation for consumer products (focus on clothing and accessories).
    You must estimate the CO2e emissions, based on the entire life cycle (cradle to grave), for the following product.

    Product data: {json.dumps(product_data, ensure_ascii=False)} 

    INSTRUCTIONS:
    1. FIRST, check if there are any official carbon footprint reports or environmental product declarations (EPD) 
       from the manufacturer for this specific product.
       If found, use these official values as your primary source.

    2. If NO official manufacturer reports are available, then estimate emissions following these protocols:
       - GHG Protocol Product Standard for system boundaries and calculation methodology
       - ISO 14040/14044 for Life Cycle Assessment principles
       - PAS 2050 and ISO/TS 14067 for carbon footprint calculation guidelines

    3. For estimation, consider:
       - Main materials composition
       - Manufacturing processes
       - Transportation
       - Use phase energy consumption
       - End-of-life disposal

    4. Use the most recent emission factors and scientific data available
    5. Document your sources and assumptions in the explanation
    6. Clearly state if you're using manufacturer data or estimation

    Reply ONLY with a JSON object containing these exact fields:
    {{
        "co2e_kg": <number>,
        "source": <if "manufacturer report" or "estimation">,
        "explanation": "<detailed explanation including data source>"
    }}
    Do not include any markdown formatting or additional JSON wrappers.
    """

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} 
        )
        
        if not response or not response.choices:
            return {"co2e_kg": None, "source": "error", "explanation": "Error: empty response"}, None

        raw_content = response.choices[0].message.content.strip()
        
        # Recupera Usage Stats
        usage_stats = response.usage

        return json.loads(raw_content), usage_stats
        
    except json.JSONDecodeError:
        return {
            "co2e_kg": None, 
            "source": "parsing_error", 
            "explanation": f"Failed to parse JSON despite JSON Mode."
        }, None
    except Exception as e:
        # print(f"Error calling API: {str(e)}") # Commentato per pulizia tqdm
        return {
            "co2e_kg": None, 
            "source": "api_error", 
            "explanation": f"API/Network Error: {str(e)}"
        }, None

def process_single_item(product):
    """
    Funzione eseguita dai Worker: Stima -> Scrittura Thread-Safe
    """
    try:
        # 1. Chiamata API 
        answer_data, usage = estimate_co2_for_product(product)

        # 2. Scrittura su file (Veloce, protetta dal Lock)
        with file_lock:
            # --- FILE 1: DATI ---
            result_record = {
                "parent_asin": product.get("parent_asin"),
                "product_name": product.get("title", "Unknown"),
                "co2e_kg": answer_data.get("co2e_kg"),
                "source": answer_data.get("source"),
                "explanation": answer_data.get("explanation")
            }
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
                out_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                out_file.flush()
                os.fsync(out_file.fileno())

            # --- FILE 2: COSTI ---
            if usage:
                cost_record = {
                    "parent_asin": product.get("parent_asin"),
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost_usd": calculate_cost(usage, MODEL),
                    "model": MODEL
                }
                
                with open(COST_FILE, "a", encoding="utf-8") as cost_file:
                    cost_file.write(json.dumps(cost_record, ensure_ascii=False) + "\n")
                    cost_file.flush()
                    os.fsync(cost_file.fileno())
        
        return True # Successo

    except Exception as e:
        # Gestione errori con lock
        with file_lock:
            error_record = {
                "parent_asin": product.get("parent_asin"),
                "product_name": product.get("title", "Unknown"),
                "co2e_kg": None,
                "explanation": f"Thread Error: {str(e)}"
            }
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
                out_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                out_file.flush()
        return False

def main(num_rows):
    # 1. RESUME LOGIC
    processed_asins = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Checking existing file: {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "parent_asin" in data:
                        processed_asins.add(data["parent_asin"])
                except:
                    pass
        print(f"✅ Found {len(processed_asins)} products already processed. They will be skipped.")

    # 2. LOAD DATA
    products_to_process = []
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            count = 0
            for i, line in enumerate(f):
                if count >= num_rows:
                    break
                try:
                    product = json.loads(line.strip())
                    p_asin = product.get("parent_asin")
                    if p_asin not in processed_asins:
                        products_to_process.append(product)
                        count += 1
                except:
                    continue
    except FileNotFoundError:
        print(f"❌ Error: Input file {INPUT_FILE} not found.")
        return

    total_items = len(products_to_process)
    print(f"🚀 Products to process: {total_items}")
    print(f"⚡ Parallel Workers: {MAX_WORKERS}")

    # 3. PARALLEL PROCESSING LOOP
    # Usiamo ThreadPoolExecutor per gestire i worker
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Sottometti tutti i task
        futures = [executor.submit(process_single_item, p) for p in products_to_process]
        
        # Monitora progresso con TQDM
        for _ in tqdm(as_completed(futures), total=total_items, desc="Processing"):
            pass

    print("\n✅ Processing completed.")

if __name__ == "__main__":
    main(num_rows=1000) 