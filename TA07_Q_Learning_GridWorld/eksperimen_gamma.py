"""Eksperimen pengaruh discount factor (gamma) terhadap Q-Learning."""
import matplotlib.pyplot as plt
import numpy as np
from app import train, moving_average, OUT

def main():
    _, rewards_high = train(gamma=0.9, seed=42)
    _, rewards_low = train(gamma=0.05, seed=42)
    ma_high = moving_average(rewards_high, 100)
    ma_low = moving_average(rewards_low, 100)
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(100, len(rewards_high)+1), ma_high, label="Gamma = 0.90")
    plt.plot(np.arange(100, len(rewards_low)+1), ma_low, label="Gamma = 0.05")
    plt.title("Perbandingan Kurva Pembelajaran berdasarkan Discount Factor")
    plt.xlabel("Episode"); plt.ylabel("Rata-rata Reward (100 episode)")
    plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"eksperimen_gamma.png", dpi=180)
    print("Rata-rata reward 100 episode terakhir:")
    print(f"Gamma 0.90: {np.mean(rewards_high[-100:]):.2f}")
    print(f"Gamma 0.05: {np.mean(rewards_low[-100:]):.2f}")
    print("Grafik tersimpan di hasil/eksperimen_gamma.png")

if __name__ == "__main__":
    main()
