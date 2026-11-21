"""
TA-07: Implementasi Q-Learning pada Simulasi Agen Pengiriman Barang
Jalankan: python app.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- KONFIGURASI LINGKUNGAN --------------------
ROWS, COLS = 4, 4
START, GOAL = 0, 15
OBSTACLES = {5, 7, 12}  # petak terlarang
ACTIONS = {0: "Atas", 1: "Kanan", 2: "Bawah", 3: "Kiri"}
DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
N_STATES, N_ACTIONS = ROWS * COLS, len(ACTIONS)
ALPHA, GAMMA = 0.8, 0.9
EPISODES, MAX_STEPS = 5000, 50
EPSILON, MIN_EPSILON, DECAY = 1.0, 0.02, 0.002
OUT = Path(__file__).parent / "hasil"
OUT.mkdir(exist_ok=True)

def step(state, action):
    """Mengembalikan (state_baru, reward, done) sesuai aturan lingkungan."""
    r, c = divmod(state, COLS)
    dr, dc = DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return state, -3, False
    new_state = nr * COLS + nc
    if new_state in OBSTACLES:
        return new_state, -10, True
    if new_state == GOAL:
        return new_state, 20, True
    return new_state, -1, False

def train(gamma=GAMMA, alpha=ALPHA, episodes=EPISODES, seed=42):
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS), dtype=float)
    rewards = []
    epsilon = EPSILON
    for ep in range(episodes):
        state, total, done = START, 0.0, False
        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(q[state]))
            new_state, reward, done = step(state, action)
            future = 0.0 if done else np.max(q[new_state])
            target = reward + gamma * future
            q[state, action] += alpha * (target - q[state, action])
            state = new_state
            total += reward
            if done:
                break
        rewards.append(total)
        epsilon = max(MIN_EPSILON, epsilon * (1.0 - DECAY))
    return q, np.asarray(rewards)

def moving_average(values, window=100):
    return np.convolve(values, np.ones(window)/window, mode="valid")

def draw_environment():
    fig, ax = plt.subplots(figsize=(6, 6))
    for s in range(N_STATES):
        r, c = divmod(s, COLS)
        color = "#d9ead3"
        label = str(s)
        if s in OBSTACLES:
            color, label = "#e6b8af", "RINTANGAN"
        if s == START:
            color, label = "#cfe2f3", "MULAI"
        if s == GOAL:
            color, label = "#ffe599", "TUJUAN"
        ax.add_patch(plt.Rectangle((c, ROWS-1-r), 1, 1, facecolor=color, edgecolor="black"))
        ax.text(c+0.5, ROWS-1-r+0.5, label, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Lingkungan Grid World Agen Pengiriman")
    fig.tight_layout()
    fig.savefig(OUT/"lingkungan_grid.png", dpi=180)
    plt.close(fig)

def save_policy(q):
    symbols = {0:"↑", 1:"→", 2:"↓", 3:"←"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for s in range(N_STATES):
        r, c = divmod(s, COLS)
        color, label = "#d9ead3", symbols[int(np.argmax(q[s]))]
        if s in OBSTACLES: color, label = "#e6b8af", "X"
        if s == START: color, label = "#cfe2f3", "S"
        if s == GOAL: color, label = "#ffe599", "G"
        ax.add_patch(plt.Rectangle((c, ROWS-1-r), 1, 1, facecolor=color, edgecolor="black"))
        ax.text(c+0.5, ROWS-1-r+0.5, label, ha="center", va="center", fontsize=20, weight="bold")
    ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Kebijakan Optimal Agen (S=Mulai, G=Tujuan, X=Rintangan)")
    fig.tight_layout()
    fig.savefig(OUT/"policy_optimal.png", dpi=180)
    plt.close(fig)

def main():
    q, rewards = train()
    ma = moving_average(rewards, 100)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(100, len(rewards)+1), ma, linewidth=1.5)
    ax.set_title("Kurva Pembelajaran Q-Learning (Moving Average 100 Episode)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Rata-rata Reward")
    ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(OUT/"kurva_pembelajaran.png", dpi=180); plt.close(fig)
    draw_environment()
    save_policy(q)
    np.savetxt(OUT/"q_table.csv", q, delimiter=",", fmt="%.4f")
    success = evaluate(q)
    print("=== HASIL PELATIHAN Q-LEARNING ===")
    print(f"Jumlah episode       : {EPISODES}")
    print(f"Ukuran Q-Table       : {q.shape}")
    print(f"Rata-rata reward 100 episode terakhir: {np.mean(rewards[-100:]):.2f}")
    print(f"Keberhasilan evaluasi greedy: {success*100:.1f}%")
    print(f"File hasil disimpan di: {OUT}")

def evaluate(q, trials=200):
    success = 0
    for _ in range(trials):
        state = START
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q[state]))
            state, reward, done = step(state, action)
            if done:
                success += int(state == GOAL)
                break
    return success / trials

if __name__ == "__main__":
    main()
