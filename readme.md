# EcoAmazon: Large Language Models for Product Carbon Footprint Estimation

This repository contains the official implementation for estimating the **Product Carbon Footprint (PCF)** of Amazon products (Clothing, Electronics, Home & Kitchen) using state-of-the-art Large Language Models (LLMs).

The project leverages **Google Gemini 2.5 Flash** and **OpenAI o3-mini** to infer CO2e emissions based on product metadata, following strictly defined Life Cycle Assessment (LCA) standards (GHG Protocol, ISO 14040/14044).

## 🚀 Key Features

* **Multi-Model Estimation**: Comparative pipelines for **Gemini 2.5 Flash** (Vertex AI) and **OpenAI o3-mini**.
* **Standards-Compliant Prompting**: The estimation logic prioritizes official Environmental Product Declarations (EPD) and falls back to heuristic estimation based on ISO/PAS standards.
* **High-Performance Processing**: Implements `ThreadPoolExecutor` for parallel processing of large datasets.
* **Robust Error Handling**: Includes automatic retries for API quotas, signal handling for safe shutdowns, and distinct error logging.
* **Cost & Token Tracking**: (OpenAI pipeline) Logs input/output tokens and calculates USD costs per estimation.

## 📂 Repository Structure

```text
EcoAmazon/
├── ecoamazon/
	|---src/llm_estimation_gemini.py    # Main script for Google Vertex AI (Gemini 2.5 Flash)
    |---src/llm_estimation_openai.py    # Main script for OpenAI (o3-mini) with cost tracking
    |---data/						    # We publish here our enriched datasets
|-- use_case/						    # Use case for reranking and provide greener reclist
	|---src/train_recsys.py				# Code used to train a RS with RecBole
	|---src/rerakn_rec_list.py			# Code used to load the trained model and perform the re-rankin
	|---eval.py							# Code used to perform the evaluation of both OG and green reclists
	|---utils.py						# Utility functions
├── README.md                  		# Project documentation




