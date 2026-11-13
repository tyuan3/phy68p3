import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

OUT_DIR = os.path.join(os.getcwd(), "plots")   # ./plots relative to where you run the script
os.makedirs(OUT_DIR, exist_ok=True)

L = 20
J = 1.0
k_B = 1.0
temps = np.linspace(0.5, 5.0, 46)   # step = 0.1
num_iterations = 1_000_000
burn_in = 500_000
chunk = 50_000
nT = len(temps)
idx = np.arange(nT)

# Padded lattices: zero border => open boundaries, neighbour sum is just the 4 neighbours
P = np.zeros((nT, L + 2, L + 2), dtype=np.int8)
P[:, 1:-1, 1:-1] = np.random.choice([-1, 1], size=(nT, L, L))
P[temps < 2.0, 1:-1, 1:-1] = 1   # start low-T lattices fully ordered, comment out for incorrect results for slides

def total_energy(P):
    c = P[:, 1:-1, 1:-1].astype(np.int64)
    bonds = (c * P[:, 2:, 1:-1]).sum((1, 2)) + (c * P[:, :-2, 1:-1]).sum((1, 2)) \
          + (c * P[:, 1:-1, 2:]).sum((1, 2)) + (c * P[:, 1:-1, :-2]).sum((1, 2))
    return -J * bonds / 2.0   # each bond counted twice

energy = total_energy(P)
mag = P[:, 1:-1, 1:-1].mean((1, 2)).astype(float)

# running sums after burn-in
n = 0
S_absm = np.zeros(nT); S_m = np.zeros(nT); S_m2 = np.zeros(nT)
S_e = np.zeros(nT); S_e2 = np.zeros(nT)

t0 = time.time()
for start in range(0, num_iterations, chunk):
    I = np.random.randint(1, L + 1, size=(chunk, nT))
    Jj = np.random.randint(1, L + 1, size=(chunk, nT))
    R = np.random.rand(chunk, nT)
    for k in range(chunk):
        i, j = I[k], Jj[k]
        s = P[idx, i, j].astype(np.int64)
        nb = P[idx, i - 1, j] + P[idx, i + 1, j] + P[idx, i, j - 1] + P[idx, i, j + 1]
        dE = 2 * J * s * nb
        acc = (dE <= 0) | (R[k] < np.exp(-np.minimum(dE, 50) / (k_B * temps)))
        a = idx[acc]
        P[a, i[acc], j[acc]] *= -1
        mag[acc] -= 2 * s[acc] / L**2
        energy[acc] += dE[acc]
        if start + k >= burn_in:
            n += 1
            S_absm += np.abs(mag); S_m += mag; S_m2 += mag**2
            S_e += energy; S_e2 += energy**2
print(f"sampling took {time.time()-t0:.1f}s")

# sanity checks
print("energy consistent:", np.allclose(energy, total_energy(P)))
print("mag consistent:", np.allclose(mag, P[:, 1:-1, 1:-1].mean((1, 2))))

absm = S_absm / n
m2 = S_m2 / n
e_mean = S_e / n / L**2
chi = L**2 * (m2 - absm**2) / temps          # susceptibility (using |m|)
C = (S_e2 / n - (S_e / n)**2) / (L**2 * temps**2)   # specific heat per spin

# Tc estimates
dm = np.gradient(absm, temps)
Tc_slope = temps[np.argmin(dm)]
Tc_chi = temps[np.argmax(chi)]
Tc_C = temps[np.argmax(C)]
print(f"Tc from steepest |m| drop : {Tc_slope:.2f}")
print(f"Tc from chi peak          : {Tc_chi:.2f}")
print(f"Tc from C peak            : {Tc_C:.2f}")
print("Onsager infinite-lattice  : 2.269")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(temps, absm, "o-", ms=4, label=r"$\langle |m| \rangle$")
ax.axvline(Tc_chi, color="r", ls="--", label=f"$T_c$ est. (χ peak) ≈ {Tc_chi:.2f}")
ax.axvline(2.269, color="gray", ls=":", label=r"Onsager $T_c$ = 2.269 (infinite lattice)")
ax.set_xlabel("Temperature (J/k_B)")
ax.set_ylabel("Mean |magnetization| per spin")
ax.set_title(f"2D Ising model (open boundaries), L={L}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ising_curve.png"), dpi=300)

# Results table (one row per temperature) -> CSV
df = pd.DataFrame({
    "temperature": temps,
    "mean_abs_magnetization_per_spin": absm,
    "mean_energy_per_spin": e_mean,
    "susceptibility_chi": chi,
    "specific_heat_C": C,
})
df.to_csv(os.path.join(OUT_DIR, "ising_sweep_results.csv"), index=False, float_format="%.6f")