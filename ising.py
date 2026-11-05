import numpy as np
import matplotlib.pyplot as plt

class State:
    def __init__(self, lattice_size, J=1.0):
        self.lattice_size = lattice_size
        self.J = J
        self.state = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
        self.adjacency_dict = self._build_adjacency()

    def _build_adjacency(self):
        L = self.lattice_size
        adjacency_dict = {}
        for i in range(L):
            for j in range(L):
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1, j))   # up
                if i < L - 1:
                    neighbors.append((i + 1, j))   # down
                if j > 0:
                    neighbors.append((i, j - 1))   # left
                if j < L - 1:
                    neighbors.append((i, j + 1))   # right
                adjacency_dict[(i, j)] = neighbors
        return adjacency_dict

    def flip_spin(self, i, j):
        self.state[i, j] *= -1

    def get_state(self):
        return self.state

    def set_state(self, new_state):
        self.state = new_state

    def get_magnetization(self):
        """Magnetization per spin, between -1 and +1."""
        return np.mean(self.state)

    def get_energy(self):
        """Total energy: E = -J * sum over bonds of s_i s_j."""
        energy = 0.0
        for (i, j), neighbors in self.adjacency_dict.items():
            for (ni, nj) in neighbors:
                energy += self.state[i, j] * self.state[ni, nj]
        # Each bond was visited twice (once from each end)
        return -self.J * energy / 2

    def delta_energy(self, i, j):
        """Energy change if spin (i, j) were flipped (does not flip it)."""
        neighbor_sum = sum(self.state[ni, nj] for (ni, nj) in self.adjacency_dict[(i, j)])
        return 2 * self.J * self.state[i, j] * neighbor_sum


def metropolis_sampler(state, temp, num_iterations=100_000, k_B=1.0):
    """
    Single-spin-flip Metropolis sampler (k_B = 1.0).
    One iteration = one proposed flip of one randomly chosen spin.

    Returns:
    - magnetizations: magnetization per spin after each iteration.
    - energies: total energy after each iteration.
    """
    L = state.lattice_size
    magnetizations = np.zeros(num_iterations)
    energies = np.zeros(num_iterations)

    energy = state.get_energy()
    magnetization = state.get_magnetization()

    for it in range(num_iterations):
        # 1. Pick one random spin
        i = np.random.randint(L)
        j = np.random.randint(L)

        # 2. Compute dE for flipping it
        dE = state.delta_energy(i, j)

        # 3. Accept or reject
        if dE <= 0 or np.random.rand() < np.exp(-dE / (k_B * temp)):
            magnetization -= 2 * state.state[i, j] / L**2  # old spin value
            state.flip_spin(i, j)
            energy += dE

        magnetizations[it] = magnetization
        energies[it] = energy

    return magnetizations, energies

np.random.seed(0)

L = 20
temp = 1000
num_iterations = 200_000

s = State(L)
mags, energies = metropolis_sampler(s, temp, num_iterations)

# Sanity checks
print(np.isclose(energies[-1], s.get_energy()))
print(np.isclose(mags[-1], s.get_magnetization()))

iterations = np.arange(num_iterations)

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(iterations, mags)
ax[0].set_ylabel("Magnetization per spin")
ax[0].set_title(f"Ising model (open boundaries), L={L}, T={temp}")
ax[1].plot(iterations, energies / L**2)
ax[1].set_ylabel("Energy per spin")
ax[1].set_xlabel("Iteration (one proposed flip each)")
plt.tight_layout()
plt.show()

from scipy.stats import norm

def plot_histogram(ax, samples, xlabel, bins=50):
    mu, sigma = np.mean(samples), np.std(samples)

    ax.hist(samples, bins=bins, density=True, alpha=0.6, label="samples")

    # Normal curve with the same mean and std
    x = np.linspace(samples.min(), samples.max(), 400)
    ax.plot(x, norm.pdf(x, mu, sigma), "r-", lw=2, label=f"normal fit (σ={sigma:.4f})")

    ax.axvline(mu, color="k", ls="--", label=f"mean = {mu:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()

# Discard burn-in (equilibration) before building the histograms
burn_in = 50_000
m_samples = mags[burn_in:]
e_samples = energies[burn_in:] / L**2

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
plot_histogram(ax[0], m_samples, "Magnetization per spin")
plot_histogram(ax[1], e_samples, "Energy per spin")
ax[0].set_title(f"L={L}, T={temp}")
plt.tight_layout()
plt.show()