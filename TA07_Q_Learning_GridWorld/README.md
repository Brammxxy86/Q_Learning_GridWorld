# TA-07 Q-Learning: Agen Pengiriman Barang

Project simulasi Reinforcement Learning diskrit menggunakan Q-Learning pada Grid World 4x4.

## Menjalankan di VS Code
1. Pastikan Python 3.10 atau lebih baru sudah terpasang.
2. Buka folder project di VS Code.
3. Buat virtual environment (opsional): `python -m venv venv`
4. Install dependensi: `pip install -r requirements.txt`
5. Jalankan simulasi utama: `python app.py`
6. Jalankan eksperimen gamma: `python eksperimen_gamma.py`

## Skenario
Agen bergerak dari petak awal (S) menuju tujuan (G), menghindari rintangan (X).
- State: 16 posisi pada grid 4x4.
- Action: Atas, Kanan, Bawah, Kiri.
- Reward: +20 mencapai tujuan, -10 mengenai rintangan, -1 setiap langkah, -3 menabrak batas.
- Q-learning: alpha=0.8, gamma=0.9, epsilon-greedy dengan epsilon decay.

## Output
Grafik dan tabel Q tersimpan pada folder `hasil/`.
Tidak memerlukan deployment. Repository GitHub cukup berisi source code dan dokumentasi.
