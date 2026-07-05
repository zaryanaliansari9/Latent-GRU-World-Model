# Latent GRU World Model for Robotic Manipulation

This repository contains the implementation of a **Latent GRU World Model** trained on a custom MuJoCo (Robosuite PickPlace) dataset.

The model learns a latent representation of the robot state and predicts future latent dynamics using a recurrent neural network (GRU). The project explores long-horizon world modeling for robotic manipulation and evaluates both one-step and multi-step prediction performance.

---

# Repository Structure

```
.
├── dataset/
│   ├── episode_000/
│   ├── episode_001/
│   ├── ...
│   └── episode_019/
│
├── models/
│
├── Latent_GRU/
│   ├── build_latent_sequence_dataset.py
│   ├── train_latent_gru_world_model.py
│   ├── evaluate_latent_gru_horizons.py
│   └── rollout_latent_gru_model.py
│
├── requirements.txt
└── README.md
```

---

# Dataset

The MuJoCo dataset is **not included** in this repository because of its size.

Download the dataset from:

**Google Drive**

> **https://drive.google.com/drive/folders/1bTaR7m7U2uWTZfN_dVCxCdAa6Mz6Tl9V?usp=sharing**

After downloading, extract the dataset into the repository root so that the directory structure becomes

```
dataset/

    episode_000/

        actions.npy
        joint_positions.npy
        joint_velocities.npy
        eef_positions.npy
        eef_quaternions.npy
        object_states.npy

    episode_001/

    ...

    episode_019/
```

---

# Installation

Clone the repository

```bash
git clone <your repository link>
cd <repository name>
```

Create a virtual environment

```bash
python3 -m venv iml_env
source iml_env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Step 1 : Build the Transition Dataset

This script converts the raw episode data into sequential training samples.

Run

```bash
python scripts/Latent_GRU/build_latent_sequence_dataset.py
```

This generates

```
dataset/

    X_latent_seq.npy

    Y_latent_next_state.npy
```

These files are used directly for training.

---

# Step 2 : Train the Latent GRU World Model

Start training

```bash
python scripts/Latent_GRU/train_latent_gru_world_model.py
```

The script automatically

- loads the dataset
- normalizes the data
- trains the encoder, decoder and GRU jointly
- saves the best-performing model

The trained model is saved as

```
models/

    latent_gru_world_model.pt
```

along with the normalization statistics required during evaluation.

---

# Training for Longer

The number of training epochs is controlled inside

```
scripts/Latent_GRU/train_latent_gru_world_model.py
```

Locate

```python
EPOCHS = 75
```

and increase it, for example

```python
EPOCHS = 300
```

or

```python
EPOCHS = 500
```

The script automatically keeps the checkpoint with the lowest validation loss.

---

# Step 3 : Horizon Evaluation

To evaluate short- and long-horizon prediction accuracy

```bash
python scripts/Latent_GRU/evaluate_latent_gru_horizons.py
```

The script reports prediction RMSE over multiple horizons such as

```
1-step

5-step

10-step

20-step

50-step

100-step
```

This measures how prediction error grows as the model predicts further into the future.

---

# Step 4 : Full Rollout Evaluation

To evaluate autonomous rollout performance

```bash
python scripts/Latent_GRU/rollout_latent_gru_model.py
```

The rollout script

- predicts an entire episode autoregressively
- computes rollout RMSE
- computes end-effector RMSE
- reports divergence timestep
- generates trajectory comparison plots

Unlike horizon evaluation, rollout testing feeds the model's own predictions back into itself, making it a more challenging benchmark.

---

# Expected Workflow

```
Download dataset

↓

Build latent transition dataset

↓

Train latent GRU

↓

Evaluate prediction horizons

↓

Evaluate full rollout
```

---

# Model Overview

The world model consists of

```
State + Action Sequence

↓

Encoder

↓

Latent Representation

↓

GRU Dynamics Model

↓

Predicted Future Latent

↓

Decoder

↓

Next Robot State
```

The recurrent model captures temporal dependencies while the latent representation compresses the robot state into a compact feature space for long-horizon prediction.

---

# Notes

- Ensure the dataset folder is located at the repository root.
- The training script automatically stores normalization statistics used during evaluation.
- Evaluation scripts require the trained checkpoint inside the `models/` directory.

---
