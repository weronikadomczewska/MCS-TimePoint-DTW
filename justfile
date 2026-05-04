venv:
    source .venv/bin/activate
precommit:
    pre-commit run --all-files

run-synthetic-prep:
    python synthetic_data/generation_pipeline.py
    
run-real-prep:
    python -m real_data.prepare_for_finetuning
