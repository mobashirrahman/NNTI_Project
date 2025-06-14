# README

This repository contains code and instructions for three tasks related to fine-tuning, influence function-based data selection, and data selection with fine-tuning exploration. The tasks are implemented in Python, with support for hyperparameter tuning and experiment tracking using Weights & Biases (WandB). Below are instructions for running the standalone code and reproducing the experiments.

The current date is March 10, 2025, and all instructions are up-to-date as of this date.

---

## Table of Contents
1. Objective
2. Prerequisites
3. How to Run Standalone Code
   - Task 1: Fine-Tuning Lipophilicity
   - Task 2: Influence Function-based Data Selection
   - Task 3: Data Selection & Fine-Tuning Exploration
4. How to Reproduce Our Runs
   - Task 1: Notebook
   - Task 1: Hyperparameter Tuning
   - Task 1: Reproducibility
   - Task 2: Influence Function-based Data Selection
   - Task 3: Data Selection & Fine-Tuning Exploration
5. Directory Map
6. Citation / References
7. Contact / Maintainers
8. Additional Notes

---

## Objective

- This project was carried out as a requirement for the course **Neural Networks: Theory and Implementation (WS 2024)** taught by **Prof. Dietrich Klakow** at **Saarland University**.

## Prerequisites

Before running the code, ensure you have the following:
- Python 3.8 or higher installed.
- Required Python packages installed (e.g., torch, transformers, wandb, etc.). Install them using:
  ```bash
  pip install -r requirements.txt
  ```
- A Weights & Biases (WandB) account and API key for experiment tracking.
- Access to a Condor cluster (for hyperparameter sweeps) or a local machine with sufficient resources.
- Datasets:
  - Task 1: Lipophilicity dataset (e.g., scikit-fingerprints/MoleculeNet_Lipophilicity).
  - Task 2 & 3: External dataset (e.g., External-Dataset_for_Task2.csv).

---

## How to Run Standalone Code

### Task 1: Fine-Tuning Lipophilicity

The script `finetune_lipophilicity.py` fine-tunes a model on the Lipophilicity dataset.

**Usage:**
```bash
python finetune_lipophilicity.py [--epochs E] [--mlm_epochs M] [--train_batch_size B] [--lr LR] [--wandb_project PROJECT] [--wandb_api_key YOUR_API_KEY] [--output_dir OUTPUT_DIR]
```

**Arguments:**
- --epochs E: Number of fine-tuning epochs (default: varies by implementation).
- --mlm_epochs M: Number of masked language model pre-training epochs (default: varies).
- --train_batch_size B: Batch size for training (default: varies).
- --lr LR: Learning rate (default: varies).
- --wandb_project PROJECT: WandB project name (default: varies).
- --wandb_api_key YOUR_API_KEY: Your WandB API key (required for logging).
- --output_dir OUTPUT_DIR: Directory to save model outputs (default: varies).

**Example:**
```bash
python finetune_lipophilicity.py --epochs 10 --train_batch_size 16 --lr 2e-5 --wandb_project "nnti-project" --wandb_api_key "your-api-key" --output_dir "./outputs"
```

---

### Task 2: Influence Function-based Data Selection

The script `task2.py` implements influence function-based data selection for fine-tuning.

**Usage:**
```bash
python task2.py [--lissa_recursion_depth D] [--lissa_damping DAMP] [--lissa_scale S] [--external_top_percent P] [--fine_tune_epochs E] [--fine_tune_lr LR] [--batch_size_train B] [--batch_size_eval BE] [--max_length L] [--project P]
```

**Arguments:**
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

**Example:**
```bash
python task2.py --lissa_recursion_depth 50 --external_top_percent 0.15 --fine_tune_epochs 5 --batch_size_train 32 --project "Task2_InfluenceFunctions"
```

---

### Task 3: Data Selection & Fine-Tuning Exploration

The script `task3.py` explores data selection and fine-tuning with methods like BitFit, LoRA, and iA3.

**Usage:**
```bash
python task3.py [--selection_percent P] [--data_selection_method M] [--epochs E] [--batch_size B] [--lr_bitfit LB] [--lr_lora LL] [--lr_ia3 LI] [--patience P] [--wandb_project WP] [--wandb_entity WE] [--model_name MN] [--dataset_path DP] [--external_dataset_path EP] [--experiments EX]
```

**Arguments:**
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

**Example:**
```bash
python task3.py --selection_percent 0.60 --data_selection_method "random" --epochs 10 --lr_bitfit 1e-3 --wandb_project "nnti-project" --experiments "bitfit,lora"
```

---

## How to Reproduce Our Runs

### Task 1: Notebook
1. Upload `Task1.ipynb` to Google Colab.
2. Run all cells in the notebook.

### Task 1: Hyperparameter Tuning
1. Create a WandB Project:
   - Log in to WandB and create a project (e.g., "nnti-project").
2. Set Up Sweep:
   - Create a new sweep using `task1_hyperparameter_tuning_sweep.config`.
   - After creating the sweep, note the `sweep_id` (e.g., `esb7uvc6`).
3. Update Condor Submission File:
   - Open `submit_finetune_lipophilicity.sub`.
   - Insert the `sweep_id` into the argument field.
   - Update the environment parameters with your details:
     ```
     WANDB_API_KEY=<Your WandB API Key>
     WANDB_ENTITY=<Your WandB Username>
     WANDB_PROJECT=<Your Project Name>
     WANDB_SWEEP=true
     ```
4. Run the Sweep:
   ```bash
   condor_submit submit_finetune_lipophilicity.sub
   ```
5. Monitor Progress:
   - Check the WandB dashboard for experiment results.

### Task 1: Reproducibility
1. Create a sweep using the contents of `task1_reproducibility.yaml`.
2. Follow the same steps as in "Task 1: Hyperparameter Tuning" to run the sweep.

### Task 2: Influence Function-based Data Selection
1. The best models will be stored in `results/task1/checkpoints`. Choose the model with highest epochs. Copy it to the `./` folder and rename it to `task1_best_model.pt`.
1. Load `config_task2.yaml` and create new sweep in WandB website.
2. Update `submit_task2_sweep.sub`:
   - Set the `sweep_id` in the argument field.
   - Update the environment variables (e.g., `WANDB_API_KEY`, `WANDB_PROJECT`).
3. Run:
   ```bash
   condor_submit submit_task2_sweep.sub
   ```

### Task 3: Data Selection & Fine-Tuning Exploration
1. Create Sweeps in WandB website:
   - For iA3: Use `config_task3_ia3.yaml`.
   - For LoRA: Use `config_task3_lora.yaml`.
   - For BitFit: Use `config_task3_bitfit.yaml`.
2. Update Submission File:
   - Open `submit_task3_sweep.sub`.
   - Set the appropriate `sweep_id` in the argument field.
   - Update the environment variables (e.g., `WANDB_API_KEY`, `WANDB_PROJECT`).
3. Run:
   ```bash
   condor_submit submit_task3_sweep.sub
   ```

---

## Directory Map

The directory structure of this repository is as follows:
```text
.
├── config/
│   ├── config_task1_hyperparameter_tuning.yaml
│   ├── config_task1_reproducibility.yaml
│   ├── config_task2.yaml
│   ├── config_task3_bitfit.yaml
│   ├── config_task3_lora.yaml
│   └── config_task3_ia3.yaml
├── notebooks/
│   ├── Task1.ipynb
│   └── Task1.pdf
├── results/
│   ├── task1/
│   ├── task2/
│   └── task3/
├── tasks/
│   ├── Task1.md
│   ├── Task2.md
│   ├── Task3.md
│   └── External-Dataset_for_Task2.csv
├── finetune_lipophilicity.py
├── task2.py
├── task3.py
├── run_finetune_lipophilicity.sh
├── run_task2_sweep.sh
├── run_task3_sweep.sh
├── submit_finetune_lipophilicity.sub
├── submit_task2_sweep.sub
├── submit_task3_sweep.sub
├── requirements.txt
└── README.txt
```

## Citation / References

1. Ben Zaken, E., Goldberg, Y., & Ravfogel, S. (2021). *BitFit: Simple parameter-efficient fine-tuning for transformer-based masked language models*. arXiv:2106.10199.
2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv:2106.09685.
3. Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. (2022). *(IA)³: Efficient adapter tuning with input-dependent inference*. arXiv:2205.05638.
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization*. In *International Conference on Learning Representations (ICLR)*. arXiv:1711.05101.
5. Park, H.-S., & Jun, C.-H. (2009). *A simple and fast algorithm for k-medoids clustering*. *Expert Systems with Applications*, 36(2), 3336–3341.
6. Rahman, M. M. (2025a). *Task1.pdf: Detailed report for Task1.ipynb run on Google Colab*. Saarland University. https://gitlab.cs.uni-saarland.de/mdra00001/nnti-project/-/blob/main/notebooks/Task1.pdf
7. Rahman, M. M. (2025b). *Weights & Biases grid sweep report for MoLFormer-XL fine-tuning*. https://api.wandb.ai/links/mobashirrahman-saarland-university/d9kycyqy
8. Rahman, M. M. (2025c). *NNTI Task 1: Reproducing best model*. https://wandb.ai/mobashirrahman-saarland-university/nnti-project/reports/NNTI-Task1-Reproducing-Best-Model--VmlldzoxMTcwNzkyMA
9. Ross, D. S., Xu, Y., Wang, H., Su, J., Zou, J., Yu, B., Khisamutdinov, E. F., Adams, R. P., & Coley, C. W. (2021). *Large-scale chemical language representations capture molecular structure and properties*. arXiv:2106.09553.
10. Ruder, S., & Plank, B. (2017). *Learning to select data for transfer learning with Bayesian optimization*. In *Proceedings of EMNLP* (pp. 372–382).
11. Wang, H., Ross, D. S., Xu, Y., Su, J., Khisamutdinov, E. F., Adams, R. P., & Coley, C. W. (2024). *Regression with large language models for materials and molecular property prediction*. arXiv:2409.06080.

## Contact / Maintainers

For any questions or issues, please contact:
- Md Mobashir Rahman
- Email: mdra000001@stud.uni-saarland.de

## Additional Notes
- WandB API keys and project details must be correctly configured for logging to work.
- If you encounter issues, refer to the WandB dashboard or Condor logs for debugging.

---