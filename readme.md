# Eco-Amazon: Enriching E-commerce Datasets with Product Carbon Footprint for Sustainable Recommendations

This repository contains the official implementation for estimating the **Product Carbon Footprint (PCF)** of Amazon products (Clothing, Electronics, Home & Kitchen) using state-of-the-art Large Language Models (LLMs).

The project leverages **Google Gemini 2.5 Flash** and **OpenAI o3-mini** to infer CO2e emissions based on product metadata, following strictly defined Life Cycle Assessment (LCA) standards (GHG Protocol, ISO 14040/14044).

## 🚀 Key Features

* **Multi-Model Estimation**: Comparative pipelines for **Gemini 2.5 Flash** (Vertex AI) and **OpenAI o3-mini**.
* **Standards-Compliant Prompting**: The estimation logic prioritizes official Environmental Product Declarations (EPD) and falls back to heuristic estimation based on ISO/PAS standards.
* **High-Performance Processing**: Implements `ThreadPoolExecutor` for parallel processing of large datasets.
* **Robust Error Handling**: Includes automatic retries for API quotas, signal handling for safe shutdowns, and distinct error logging.
* **Cost & Token Tracking**: (OpenAI pipeline) Logs input/output tokens and calculates USD costs per estimation.

## 📂 Repository Structure

```
EcoAmazon/
├── ecoamazon/
	|---src/llm_estimation_gemini.py    # Main script for Google Vertex AI (Gemini 2.5 Flash)
    |---src/llm_estimation_openai.py    # Main script for OpenAI (o3-mini) with cost tracking
    |---data/						    # We publish here our enriched datasets
|-- use_case/						    # Use case for reranking and provide greener reclist
	|---src/train_recsys.py				# Code used to train a RS with RecBole
	|---src/rerakn_rec_list.py			# Code used to load the trained model and perform the re-rankin
	|---src/eval.py							# Code used to perform the evaluation of both OG and green reclists
	|---src/utils.py						# Utility functions
├── README.md                  		# Project documentation
```

## LLM prompting 

To prompt the LLM and obtain PCF estimation, first, these are the dependecied needed:
```
# Shared dependencies
pandas
tqdm
python-dotenv

# Gemini specific
google-cloud-aiplatform
google-auth

# OpenAI specific
openai
```

### LLM Carbon Estimation Pipeline


### 1. Initialization and Authentication
* **Configuration**: Both scripts begin by defining local file paths for input (`.jsonl`), output, and authentication keys. **IMPORTANT:** Due to GitHub file size limit, we *cannot* put here the original input files of the Electronics, Clothing and Home&Kitchen dataset. However, they can be downloaded from the [original source, the Amazon Reviews 23 datasets](https://amazon-reviews-2023.github.io). In particular, the inout files of this step are the metadata files:
	- [Electronics](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz)
	- [Clothing](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz)
	- [Home&Kitchen](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Home_and_Kitchen.jsonl.gz)
* **Gemini Authentication**: Uses a service account JSON key to initialize `vertexai`.
* **OpenAI Authentication**: Loads API keys via environment variables using `python-dotenv`.

### 2. Resume Logic
* **Pre-check**: Before processing, the scripts read the current `OUTPUT_FILE`.
* **Unique Mapping**: They extract all `parent_asin` values that have already been successfully calculated.
* **Filtering**: The scripts compare the input dataset against the processed set and only queue products that are missing, preventing double-billing and redundant work.

### 3. Parallel Processing Architecture
* **Multi-threading**: Both scripts use `ThreadPoolExecutor` to manage multiple API requests simultaneously.
* **Concurrency Control**: `MAX_WORKERS` is set to 10 by default to maximize throughput while respecting API rate limits.
* **Progress Monitoring**: Integration with `tqdm` provides a visual progress bar and estimated time of completion.

### 4. LCA Estimation Prompt Logic
The scripts send a specialized prompt to the models (Gemini 2.5 Flash or OpenAI o3-mini) with the following hierarchy:
1. **Manufacturer Search**: Instructs the AI to look for official environmental product declarations (EPD) first.
2. **Standardized Estimation**: If no data is found, it estimates CO2e based on materials, manufacturing, transport, and disposal following ISO 14040/14044 protocols.
3. **Structured Response**: Forces the model to respond in a strict JSON format containing `co2e_kg`, `source`, and a detailed `explanation`.

### 5. Data Integrity and Thread Safety
* **File Locking**: A `threading.Lock()` ensures that only one worker thread can write to the output file at a time, preventing data corruption or "intertwined" JSON strings.
* **Atomic Saving**: The scripts use `os.fsync()` and `flush()` to ensure data is physically written to the disk immediately after each success.
* **Graceful Shutdown**: The Gemini script includes a `signal_handler` to catch `Ctrl+C`, allowing the user to stop the script safely without losing current progress.

### 6. Cost and Error Monitoring
* **Cost Tracking (OpenAI)**: Calculates the USD cost per request based on token usage and model-specific pricing.
* **Retry Mechanism (Gemini)**: Includes a while-loop to handle `429` (Rate Limit) errors, implementing an exponential backoff before retrying.
* **Error Logging**: Failed requests are logged with an error explanation instead of being skipped silently.


## Recommendation Engine with SaS computing

To train the Recommendation models with RecBole, and apply the SaS re-ranking, we need the following requirements.
```
recbole
torch
pandas
numpy
ray[tune]
tqdm
```

We suggest to refer to the original [RecBole documentation](https://recbole.io/docs/) for further details about the framework.

### Sustainable Re-ranking & Evaluation

This pipeline manages the training of Recommendation Systems (RecSys), the application of a "green" re-ranking algorithm, and the final evaluation using both accuracy and sustainability metrics.

### 0. Process dataset
* The script in `use_case/data/process_data.py` is used to process the original dataset into the RecBole format. Also in this case, due to GitHub file size limit, we *cannot* put here the original input files of the Electronics, Clothing and Home&Kitchen dataset. However, they can be downloaded from the [original source, the Amazon Reviews 23 datasets](https://amazon-reviews-2023.github.io). In particular, the input files of this step are the reveies files:
	- [Electronics](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz)
	- [Clothing](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz)
	- [Home&Kitchen](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Home_and_Kitchen.jsonl.gz)
Once the datasets are processed, they can be put in `use_case/src/dataset/`, and we provide the post-process datasets.

### 1. Model Training & HPO (`use_case/src/train_recsys.py`)
* **Hyperparameter Optimization (HPO)**: Uses the `Ray Tune` library to search for optimal learning rates, embedding sizes, and regularization weights for **BPR** and **LightGCN** models.
* **RecBole Integration**: Leverages the `RecBole` framework to handle standardized training loops on the `amazon_elec` dataset.
* **Final Training**: Once the best hyperparameters are identified (maximizing `Recall@10`), the models are retrained and checkpoints are saved to the `./models/` directory.

### 2. PCF-Aware Re-ranking (`use_case/src/rerank_rec_list.py`)
This script takes standard recommendations and adjusts them based on carbon footprint data.
* **Baseline Retrieval**: The script generates a top-100 recommendation list for every user using the trained models.
* **SaS (Sustainability-aware Score) Calculation**: A re-ranking function applies the following formula to each item:
    $$SaS = \alpha \cdot \text{Pred}_{norm} + (1 - \alpha) \cdot \text{PCF}_{norm}$$
    * **$\alpha = 1.0$**: Purely accuracy-based (original recommendation).
    * **$\alpha < 1.0$**: Introduces a sustainability bias.
* **Normalization**: Both prediction scores and PCF values are min-max normalized to ensure the weights are applied fairly.
* **Batch Saving**: The re-ranked lists for different $\alpha$ values (e.g., 0.25, 0.5, 0.75) are saved as `.pth` files.

### 3. Comprehensive Evaluation (`use_case/src/eval.py`)
The evaluation script compares the original recommendations against the "greener" lists across multiple dimensions.
* **Accuracy Metrics**: Calculates standard RecSys metrics including **Recall**, **NDCG**, **MRR**, **Precision**, and **F1-Score**.
* **Sustainability Metric (Average PCF)**: Calculates the average carbon footprint of the top-$k$ items recommended to users. 
* **Beyond-Accuracy Metrics**: Tracks **Gini Index** (diversity), **Average Popularity**, and **Item Coverage** to see if "green" re-ranking helps or hurts catalog exploration.
* **Comparative Analysis**: Results are aggregated into a final `results.csv`, allowing for a direct comparison of how different $\alpha$ weights impact the "Accuracy vs. Sustainability" trade-off.

### 4. Utility Support (`use_case/src/utils.py`)
* **Checkpoint Management**: Automatically locates the most recent `.pth` model file for evaluation.
* **Data Integration**: Connects the `parent_asin` from the metadata to the internal `item_id` used by RecBole, ensuring that the CO2e estimations generated in the previous workflow are correctly mapped to the items being recommended.
