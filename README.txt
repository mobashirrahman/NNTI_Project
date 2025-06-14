README

This repository contains code and instructions for three tasks related to fine-tuning, influence function-based data selection, and data selection with fine-tuning exploration. The tasks are implemented in Python, with support for hyperparameter tuning and experiment tracking using Weights & Biases (WandB). Below are instructions for running the standalone code and reproducing the experiments.

The current date is March 10, 2025, and all instructions are up-to-date as of this date.

---

Table of Contents
1. Prerequisites
2. How to Run Standalone Code
   - Task 1: Fine-Tuning Lipophilicity
   - Task 2: Influence Function-based Data Selection
   - Task 3: Data Selection & Fine-Tuning Exploration
3. How to Reproduce Our Runs
   - Task 1: Notebook
   - Task 1: Hyperparameter Tuning
   - Task 1: Reproducibility
   - Task 2: Influence Function-based Data Selection
   - Task 3: Data Selection & Fine-Tuning Exploration
4. Additional Notes

---

Prerequisites

Before running the code, ensure you have the following:
- Python 3.8 or higher installed.
- Required Python packages installed (e.g., torch, transformers, wandb, etc.). Install them using:
  pip install -r requirements.txt
- A Weights & Biases (WandB) account and API key for experiment tracking.
- Access to a Condor cluster (for hyperparameter sweeps) or a local machine with sufficient resources.
- Datasets:
  - Task 1: Lipophilicity dataset (e.g., scikit-fingerprints/MoleculeNet_Lipophilicity).
  - Task 2 & 3: External dataset (e.g., External-Dataset_for_Task2.csv).

---

How to Run Standalone Code

Task 1: Fine-Tuning Lipophilicity

The script finetune_lipophilicity.py fine-tunes a model on the Lipophilicity dataset.

Usage:
  python finetune_lipophilicity.py [--epochs E] [--mlm_epochs M] [--train_batch_size B] [--lr LR] [--wandb_project PROJECT] [--wandb_api_key YOUR_API_KEY] [--output_dir OUTPUT_DIR]

Arguments:
- --epochs E: Number of fine-tuning epochs (default: varies by implementation).
- --mlm_epochs M: Number of masked language model pre-training epochs (default: varies).
- --train_batch_size B: Batch size for training (default: varies).
- --lr LR: Learning rate (default: varies).
- --wandb_project PROJECT: WandB project name (default: varies).
- --wandb_api_key YOUR_API_KEY: Your WandB API key (required for logging).
- --output_dir OUTPUT_DIR: Directory to save model outputs (default: varies).

Example:
  python finetune_lipophilicity.py --epochs 10 --train_batch_size 16 --lr 2e-5 --wandb_project "nnti-project" --wandb_api_key "your-api-key" --output_dir "./outputs"

---

Task 2: Influence Function-based Data Selection

The script task2.py implements influence function-based data selection for fine-tuning.

Usage:
  python task2.py [--lissa_recursion_depth D] [--lissa_damping DAMP] [--lissa_scale S] [--external_top_percent P] [--fine_tune_epochs E] [--fine_tune_lr LR] [--batch_size_train B] [--batch_size_eval BE] [--max_length L] [--project P]

Arguments:
- --lissa_recursion_depth D: LiSSA recursion depth (default: 100).
- --lissa_damping DAMP: LiSSA damping factor (default: 0.01).
- --lissa_scale S: LiSSA scaling factor (default: 25.0).
- --external_top_percent P: Top percent of external samples to select (default: 0.10).
- --fine_tune_epochs E: Number of fine-tuning epochs (default: 10).
- --fine_tune_lr LR: Fine-tuning learning rate (default: 2e-5).
- --batch_size_train B: Training batch size (default: 16).
- --batch_size_eval BE: Evaluation batch size (default: 32).
- --max_length L: Max tokenization length (default: 128).
- --project P: WandB project name (default: "Task2_InfluenceFunctions").

Example:
  python task2.py --lissa_recursion_depth 50 --external_top_percent 0.15 --fine_tune_epochs 5 --batch_size_train 32 --project "Task2_InfluenceFunctions"

---

Task 3: Data Selection & Fine-Tuning Exploration

The script task3.py explores data selection and fine-tuning with methods like BitFit, LoRA, and iA3.

Usage:
  python task3.py [--selection_percent P] [--data_selection_method M] [--epochs E] [--batch_size B] [--lr_bitfit LB] [--lr_lora LL] [--lr_ia3 LI] [--patience P] [--wandb_project WP] [--wandb_entity WE] [--model_name MN] [--dataset_path DP] [--external_dataset_path EP] [--experiments EX]

Arguments:
- --selection_percent P: Fraction of external data to select (default: 0.50).
- --data_selection_method M: Data selection method (target_alignment, random, clustering; default: target_alignment).
- --epochs E: Number of training epochs (default: 5).
- --batch_size B: Batch size for training (default: 16).
- --lr_bitfit LB: Learning rate for BitFit (default: 5e-3).
- --lr_lora LL: Learning rate for LoRA (default: 2e-5).
- --lr_ia3 LI: Learning rate for iA3 (default: 1e-4).
- --patience P: Patience for early stopping (default: 2).
- --wandb_project WP: WandB project name (default: "nnti-project").
- --wandb_entity WE: WandB entity (default: "mobashirrahman-saarland-university").
- --model_name MN: Pre-trained model name (default: "ibm/MoLFormer-XL-both-10pct").
- --dataset_path DP: Default dataset path (default: "scikit-fingerprints/MoleculeNet_Lipophilicity").
- --external_dataset_path EP: External dataset CSV path (default: "External-Dataset_for_Task2.csv").
- --experiments EX: Comma-separated list of experiments (bitfit, lora, ia3; default: "bitfit,lora,ia3").

Example:
  python task3.py --selection_percent 0.60 --data_selection_method "random" --epochs 10 --lr_bitfit 1e-3 --wandb_project "nnti-project" --experiments "bitfit,lora"

---

How to Reproduce Our Runs

Task 1: Notebook
1. Upload Task1.ipynb to Google Colab.
2. Run all cells in the notebook.

Task 1: Hyperparameter Tuning
1. Create a WandB Project:
   - Log in to WandB and create a project (e.g., "nnti-project").
2. Set Up Sweep:
   - Create a new sweep using task1_hyperparameter_tuning_sweep.config.
   - After creating the sweep, note the sweep_id (e.g., esb7uvc6).
3. Update Condor Submission File:
   - Open submit_finetune_lipophilicity.sub.
   - Insert the sweep_id into the argument field.
   - Update the environment parameters with your details:
     WANDB_API_KEY=<Your WandB API Key>
     WANDB_ENTITY=<Your WandB Username>
     WANDB_PROJECT=<Your Project Name>
     WANDB_SWEEP=true
4. Run the Sweep:
   - In a bash terminal, execute:
     condor_submit submit_finetune_lipophilicity.sub
5. Monitor Progress:
   - Check the WandB dashboard for experiment results.

Task 1: Reproducibility
1. Create a sweep using the contents of task1_reproducibility.yaml.
2. Follow the same steps as in "Task 1: Hyperparameter Tuning" to run the sweep.

Task 2: Influence Function-based Data Selection
1. The best models will be stored in results/task1/checkpoints. Choose the model with highest epochs. Copy it to the ./ folder and rename it to task1_best_model.pt.
1. Load config_task2.yaml and create new sweep in WandB website.
2. Update submit_task2_sweep.sub:
   - Set the sweep_id in the argument field.
   - Update the environment variables (e.g., WANDB_API_KEY, WANDB_PROJECT).
3. Run:
   condor_submit submit_task2_sweep.sub

Task 3: Data Selection & Fine-Tuning Exploration
1. Create Sweeps in WandB website:
   - For iA3: Use config_task3_ia3.yaml.
   - For LoRA: Use config_task3_lora.yaml.
   - For BitFit: Use config_task3_bitfit.yaml.
2. Update Submission File:
   - Open submit_task3_sweep.sub.
   - Set the appropriate sweep_id in the argument field.
   - Update the environment variables (e.g., WANDB_API_KEY, WANDB_PROJECT).
3. Run:
   condor_submit submit_task3_sweep.sub

---

Additional Notes
- WandB API keys and project details must be correctly configured for logging to work.
- If you encounter issues, refer to the WandB dashboard or Condor logs for debugging.

---