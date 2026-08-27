# Trained Models

This public folder includes only lightweight model artifacts that can be redistributed with the repository:

- `ml_watermark_logreg_archive4_llm_only.pkl`
- `xgboost_kaggle.pkl`

Large or locally licensed checkpoints are intentionally not committed. Configure them with local placeholder paths when running the corresponding detector:

- ScamLLM: Hugging Face model `phishbot/ScamLLM`.
- PiMRef identity model: `/path/to/pimref/identity-model`.
- PiMRef company knowledge base: `/path/to/company_database_knowphish_v2.json`.
- T5 phishing detector model: `/path/to/best_t5`.
- T5 tokenizer: `/path/to/t5_tokenizer`.
- SecureNet Llama model: `/path/to/Llama-3.1-8B-Instruct`.
