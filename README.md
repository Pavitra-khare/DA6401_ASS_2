# DL_Assignment2 - CS24M031

This is **Part A** of the assignment where we build a custom CNN model from scratch and fine-tune its hyperparameters on the iNaturalist dataset. The model is trained and evaluated with extensive experimentation using [Weights & Biases](https://wandb.ai/) (wandb).

All experiments were run on **Kaggle Notebooks** using a GPU-enabled environment.

[GitHub Repository](https://github.com/Pavitra-khare/DA6401_ASS_2A)

[Weights & Biases Report](https://api.wandb.ai/links/3628-pavitrakhare-indian-institute-of-technology-madras/6l8em45m)

---

## Objective

- Implement a deep CNN from scratch with 5 convolutional layers.
- Tune hyperparameters such as activation function, dropout rate, kernel size, number of filters, and dense units.
- Log and visualize experiments using wandb.
- Evaluate model performance on training, validation, and test sets.

---
## Model Architecture

### Custom CNN (ConvNetworkModel)

| Layer Type       | Configuration                                                                 |
|------------------|--------------------------------------------------------------------------------|
| **Input**        | RGB image of size `224 x 224 x 3`                                              |
| **Conv Layer 1** | Conv2d (32 filters, 3x3) → BatchNorm → GELU → MaxPool → Dropout(0.2)            |
| **Conv Layer 2** | Conv2d (64 filters, 3x3) → BatchNorm → GELU → MaxPool → Dropout(0.2)            |
| **Conv Layer 3** | Conv2d (128 filters, 3x3) → BatchNorm → GELU → MaxPool → Dropout(0.2)           |
| **Conv Layer 4** | Conv2d (256 filters, 3x3) → BatchNorm → GELU → MaxPool → Dropout(0.2)           |
| **Conv Layer 5** | Conv2d (512 filters, 3x3) → BatchNorm → GELU → MaxPool → Dropout(0.2)           |
| **Flatten**      | Reshape into a 1D vector                                                       |
| **FC Layer**     | Linear(512 → 256) → BatchNorm → GELU → Dropout(0.2)                            |
| **Output**       | Linear(256 → 10 classes)                                                       |

---

### Best Hyperparameters (from Sweep)

| Hyperparameter   | Value                    |
|------------------|--------------------------|
| `num_filters`    | `[32, 64, 128, 256, 512]` |
| `kernel_size`    | `[3, 3, 3, 3, 3]`         |
| `activation`     | `GELU`                   |
| `dropout`        | `0.2`                    |
| `batch_norm`     | `Yes`                    |
| `fc_size`        | `256`                    |
| `epochs`         | `10`                     |
| `data_aug`       | `No`                     |

Test Accuracy 81.20%
---

---


## 📁 Project File Overview

| File            | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `run.py`        | Main entry point; initiates sweep, loads model/data, trains and logs.  |
| `config.py`     | CLI argument parsing, wandb login, device setup.                        |
| `train_func.py` | Training/validation logic with early stopping and best model selection. |
| `data_load.py`  | Loads and augments train/validation/test datasets using torchvision.    |
| `model.py`      | Custom CNN model definition with 5 conv layers and configurable params. |
| `sweep_config.py` | Wandb sweep configuration using best parameters from CLI.            |
| `plot.py`       | Visualizes predictions for randomly sampled test images.                |

---

## 🗺️ File Dependency Map

<pre>
run.py
├── config.py         # Parses args, logs into wandb
├── model.py          # Defines CNN architecture
├── data_load.py      # Loads datasets with transforms
├── train_func.py     # Training, validation, early stopping logic
├── sweep_config.py   # Sweep configuration (uses bestParam)
└── plot.py           # Logs prediction visualizations to wandb
</pre>

---


## 🔧 Command Line Arguments

These are passed to `run.py` and parsed by `config.py`.

| Argument              | Default Value   | Description                                |
|-----------------------|------------------|--------------------------------------------|
| `--wandb_project`     | DL_ASS2          | Wandb project name                         |
| `--wandb_entity`      | your-entity-name | Wandb username or team                     |
| `--wandb_key`         | (Your API key)   | Wandb API key for login                    |
| `--dpTrain`           | path/to/train    | Path to training dataset                   |
| `--dpTest`            | path/to/test     | Path to test dataset                       |
| `--dropout`           | 0.2              | Dropout rate after conv/fc layers          |
| `--num_dense`         | 256              | Dense layer size                           |
| `--kernel_size`       | [3,3,3,3,3]      | Kernel sizes for 5 conv layers             |
| `--filter_org`        | [32,64,128,256,512] | Filters per conv layer                   |
| `--batch_norm`        | Yes              | Enable/disable batch normalization         |
| `--activation`        | gelu             | Activation function                        |
| `--data_aug`          | No               | Whether to use data augmentation           |
| `--epoch`             | 10               | Number of training epochs                  |

---

## 💻 How to Run

### 1. Install Requirements

```bash
pip install torch torchvision scikit-learn wandb
```

###2. run the run.py
```bash
python run.py --wandb_project <project_name> --wandb_entity <entity_name> --dpTrain <train_data_path> --dpTest <test_data_path>....
```
