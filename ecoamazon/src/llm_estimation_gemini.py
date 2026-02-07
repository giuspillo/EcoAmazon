import json
import os
import time
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
from tqdm.auto import tqdm

# Contatori globali per statistiche
success_count = 0
error_count = 0
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print(f"\n\n⚠️ Interruzione richiesta! Attendere chiusura sicura...")
    print(f"📊 Processati con successo: {success_count}")
    print(f"❌ Errori: {error_count}")
    print(f"💾 Output salvato in: {OUTPUT_FILE}")
    stop_flag = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# CONFIGURATION

KEY_PATH = "C:\\Users\\user\\Desktop\\Uni\\Semantics\\pcf_estimator_PG\\Clothing\\chiave-google.json"
INPUT_FILE = "C:\\Users\\user\\Desktop\\Uni\\Semantics\\pcf_estimator_PG\\Clothing\\meta_Clothing_Shoes_and_Jewelry_core12_noimgvid.jsonl"
OUTPUT_FILE = "metadata_estimantion.jsonl"

# 🔥 
# If you see too many 429 errors, drop to 10.
MAX_WORKERS = 10 

# Lock to safely write to file (prevents corrupted files)
file_lock = threading.Lock()

try:
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    vertexai.init(project=credentials.project_id, location="us-central1", credentials=credentials)
    print(f"✅ Authentication OK.")
except Exception as e:
    print(f"❌ Authentication Error: {e}")

model = GenerativeModel("gemini-2.5-flash") 


# WORKER FUNCTION (Executed by Threads)

def process_single_product(product):
    
    prompt = f"""
    You are an expert in life cycle analysis (LCA) and CO2e emission calculation for consumer products (focus on clothing and accessories).
    You must estimate the CO2e emissions, based on the entire life cycle (cradle to grave), for the following product.

    Product data: {json.dumps(product, ensure_ascii=False)}

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

    safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    config = GenerationConfig(temperature=0.0, max_output_tokens=8192, response_mime_type="application/json")

    retries = 0
    while retries < 3:
        try:
            response = model.generate_content(prompt, generation_config=config, safety_settings=safety)
            text = response.text.strip()
            data = json.loads(text)
            
            result_record = {
                "parent_asin": product.get("parent_asin"),
                "product_name": product.get("title", "Unknown"),
                "co2e_kg": data.get("co2e_kg"),
                "source": data.get("source"),
                "explanation": data.get("explanation")
            }
            
            with file_lock:
                global success_count
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                success_count += 1
            return True

        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                time.sleep(5 + (retries * 5)) # Longer wait for quota
                retries += 1
                continue
            else:
                global error_count
                error_count += 1
                return False
    error_count += 1
    return False


# PARALLEL MAIN LOOP

def main():
    print(f"📂 Reading dataset...")
    try:
        import pandas as pd
        df = pd.read_json(INPUT_FILE, lines=True)
    except:
        import pandas as pd
        df = pd.read_json(INPUT_FILE, lines=True, compression='gzip')
    
    products = df.to_dict('records')
    print(f"📦 Total products: {len(products)}")

    # Resume
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        print("🔄 Resuming from existing file...")
        with open(OUTPUT_FILE, 'r', encoding="utf-8") as f:
            for line in f:
                try: processed.add(json.loads(line)['parent_asin'])
                except: pass
    
    todo = [p for p in products if p.get('parent_asin') not in processed]
    print(f"🚀 To do: {len(todo)} (Already done: {len(processed)})")
    print(f"⚡ SPEED: {MAX_WORKERS} parallel threads!")

    # Multithread execution
    print(f"\n💡 Premi Ctrl+C in qualsiasi momento per interrompere in sicurezza")
    print(f"💡 Il file {OUTPUT_FILE} è accessibile anche durante l'esecuzione\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Tqdm will show real progress bar
        list(tqdm(executor.map(process_single_product, todo), total=len(todo)))

    print(f"\n✅ COMPLETATO!")
    print(f"📊 Processati con successo: {success_count}")
    print(f"❌ Errori: {error_count}")
    print(f"💾 Output salvato in: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()