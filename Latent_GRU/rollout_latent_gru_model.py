import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configuration
SEQ_LEN = 20
STATE_DIM = 77
ACTION_DIM = 7
LATENT_DIM = 32
GRU_HIDDEN_DIM = 128

EPISODE_DIR = "dataset/episode_001"

# Model definition
class LatentGRUWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM)
        )

        self.gru = nn.GRU(
            input_size=LATENT_DIM + ACTION_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=2,
            batch_first=True
        )

        self.latent_head = nn.Sequential(
            nn.Linear(GRU_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, STATE_DIM)
        )

    def forward(self, x):
        states = x[:, :, :STATE_DIM]
        actions = x[:, :, STATE_DIM:]
        
        z_history = self.encoder(states)
        gru_input = torch.cat([z_history, actions], dim=-1)
        gru_out, _ = self.gru(gru_input)
        final_hidden = gru_out[:, -1, :]

        z_next_pred = self.latent_head(final_hidden)
        next_state_pred = self.decoder(z_next_pred)
        return next_state_pred

# Load model
model = LatentGRUWorldModel()
model.load_state_dict(
    torch.load(
        "models/latent_gru_world_model.pt",
        map_location="cpu"
    )
)

model.eval()

X_mean = np.load("models/latent_gru_X_mean.npy")
X_std = np.load("models/latent_gru_X_std.npy")

Y_mean = np.load("models/latent_gru_Y_mean.npy")
Y_std = np.load("models/latent_gru_Y_std.npy")

# Load episode
joint_pos = np.load(
    os.path.join(
        EPISODE_DIR,
        "joint_positions.npy"
    )
)

joint_vel = np.load(
    os.path.join(
        EPISODE_DIR,
        "joint_velocities.npy"
    )
)

eef_pos = np.load(
    os.path.join(
        EPISODE_DIR,
        "eef_positions.npy"
    )
)

eef_quat = np.load(
    os.path.join(
        EPISODE_DIR,
        "eef_quaternions.npy"
    )
)

object_state = np.load(
    os.path.join(
        EPISODE_DIR,
        "object_states.npy"
    )
)

actions = np.load(
    os.path.join(
        EPISODE_DIR,
        "actions.npy"
    )
)

states = np.concatenate(
    [
        joint_pos,
        joint_vel,
        eef_pos,
        eef_quat,
        object_state
    ],
    axis=1
).astype(np.float32)

print("State shape:", states.shape)

# Initialize with a real 20-step history
sequence = []
for k in range(SEQ_LEN):
    sequence.append(np.concatenate([states[k], actions[k]]))

sequence = np.array(sequence, dtype=np.float32)
predicted_states = []

# Preserve the observed initialization section.
for t in range(SEQ_LEN):
    predicted_states.append(states[t].copy())

# Open-loop rollout
for t in range(SEQ_LEN, len(states) - 1):
    sequence_norm = (sequence - X_mean) / X_std

    sequence_tensor = torch.tensor(sequence_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        next_state_norm = model(sequence_tensor)

    next_state_norm = (next_state_norm.cpu().numpy()[0])
    next_state = (next_state_norm * Y_std) + Y_mean
    predicted_states.append(next_state)

    next_input = np.concatenate([next_state,actions[t]])
    sequence = np.vstack([sequence[1:], next_input])

predicted_states = np.array(predicted_states)
true_states = states[:len(predicted_states)]

rollout_rmse = np.sqrt(np.mean((predicted_states - true_states) ** 2))

eef_rmse = np.sqrt(np.mean((predicted_states[:, 14:17] - true_states[:, 14:17]) ** 2))

print(f"Full rollout RMSE = {rollout_rmse:.6f}")
print(f"EEF rollout RMSE = {eef_rmse:.6f}")

# Error growth
error_per_step = np.linalg.norm(predicted_states - true_states, axis=1)

threshold = 0.10

diverged = np.where(error_per_step > threshold)[0]

if len(diverged) > 0:
    print("Divergence timestep:", diverged[0])

else:
    print("No divergence within episode")

# Plots
plt.figure(figsize=(10, 5))

plt.plot(
    true_states[:, 14],
    label="True"
)

plt.plot(
    predicted_states[:, 14],
    label="Predicted"
)

plt.title("Latent GRU: EEF X Rollout")
plt.xlabel("Timestep")
plt.ylabel("EEF X Position")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))

plt.plot(
    true_states[:, 15],
    label="True"
)

plt.plot(
    predicted_states[:, 15],
    label="Predicted"
)

plt.title("Latent GRU: EEF Y Rollout")
plt.xlabel("Timestep")
plt.ylabel("EEF Y Position")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))

plt.plot(
    true_states[:, 16],
    label="True"
)

plt.plot(
    predicted_states[:, 16],
    label="Predicted"
)

plt.title("Latent GRU: EEF Z Rollout")
plt.xlabel("Timestep")
plt.ylabel("EEF Z Position")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))

plt.plot(error_per_step)

plt.title("Latent GRU: Prediction Error Growth")
plt.xlabel("Timestep")
plt.ylabel("L2 State Error")
plt.grid()

plt.figure(figsize=(10, 5))

plt.plot(
    true_states[:, 21],
    label="True"
)

plt.plot(
    predicted_states[:, 21],
    label="Predicted"
)

plt.title("Latent GRU: Object State[0]")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.grid()

plt.show()
