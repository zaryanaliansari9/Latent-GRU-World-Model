import os
import numpy as np

DATASET_DIR = "dataset"
SEQ_LEN = 20

all_X = []
all_Y = []

episodes = sorted([d for d in os.listdir(DATASET_DIR) if d.startswith("episode_")])

print(f"Found {len(episodes)} episodes")

for episode in episodes:

    print(f"Processing {episode}")

    ep_dir = os.path.join(DATASET_DIR, episode)

    actions = np.load(os.path.join(ep_dir, "actions.npy"))
    object_states = np.load(os.path.join(ep_dir, "object_states.npy"))
    joint_positions = np.load(os.path.join(ep_dir, "joint_positions.npy"))
    joint_velocities = np.load(os.path.join(ep_dir, "joint_velocities.npy"))
    eef_positions = np.load(os.path.join(ep_dir, "eef_positions.npy"))
    eef_quaternions = np.load(os.path.join(ep_dir, "eef_quaternions.npy"))

    states = np.concatenate(
        [
            joint_positions,
            joint_velocities,
            eef_positions,
            eef_quaternions,
            object_states
        ],
        axis=1
    )

    T = len(states)

    # Input sequence ends at time t.
    # Target is state at t + 1.
    for t in range(SEQ_LEN, T - 1):
        sequence = []
        for k in range(t - SEQ_LEN, t):
            state_action = np.concatenate([states[k], actions[k]])
            sequence.append(state_action)

        all_X.append(np.array(sequence))
        all_Y.append(states[t + 1])

X = np.array(all_X, dtype=np.float32)
Y = np.array(all_Y, dtype=np.float32)

print("X_latent_seq shape:", X.shape)
print("Y_latent_next_state shape:", Y.shape)

np.save("dataset/X_latent_seq.npy", X)
np.save("dataset/Y_latent_next_state.npy", Y)

print("\nSaved:")
print("dataset/X_latent_seq.npy")
print("dataset/Y_latent_next_state.npy")
