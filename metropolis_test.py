import numpy as np

def metropolis_sampler(target_density, proposal_sampler, initial_state, temp, num_samples=1000):
    """
    Metropolis-Hastings sampler.

    Parameters:
    - target_density: function that computes the target density at a given state.
    - proposal_sampler: function that generates a proposal state given the current state.
    - initial_state: initial state of the Markov chain.
    - temp: temperature ratio for the Boltzmann distribution.
    - num_samples: number of samples to generate.

    Returns:
    - samples: array of generated samples.
    """
    samples = np.zeros((num_samples, len(initial_state)))
    current_state = initial_state

    for i in range(num_samples):
        # Generate a proposal state
        proposal_state = proposal_sampler(current_state)

        # Compute acceptance probability
        acceptance_ratio = target_density(proposal_state, temp) / target_density(current_state, temp)
        acceptance_probability = min(1, acceptance_ratio)

        # Accept or reject the proposal
        if np.random.rand() < acceptance_probability:
            current_state = proposal_state

        samples[i] = current_state

    return samples


def boltzmann(state, T, k_B=1.380649e-23):
    """
    Unnormalized Boltzmann distribution for a given state.
    """
    E = Ecalc(state)  # Example energy function (negative sum of spins)
    return np.exp(-E/(k_B*T))

#----------------------# Example usage for Ising model: #----------------------#


NUM_TEMPS = 100
DT = 0.1
START_TEMP = 0.0
J = 1.0  # Exchange parameter
LATTICE_SIZE = 4  # dimension of the lattice

# Quick sweep to determine temperature region of interest (implement later), this will return an informed value for START_TEMP

# Create initial lattice
initial_state = np.random.choice([-1, 1], size=(LATTICE_SIZE, LATTICE_SIZE))  # Example LATTICE_SIZE x LATTICE_SIZE lattice

# Idk what this does copilot made it will investigate soon
# proposal_sampler = lambda state: state + np.random.choice([-2, 2], size=state.shape)

# Adjacency dictionary for lattice neighbors
# each spin has 2, 3, or 4 neighbors in a simple 2D lattice

adjacency_dict = {}
for i in range(LATTICE_SIZE):
    for j in range(LATTICE_SIZE):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))   # up
        if i < LATTICE_SIZE - 1:
            neighbors.append((i + 1, j))   # down
        if j > 0:
            neighbors.append((i, j - 1))   # left
        if j < LATTICE_SIZE - 1:
            neighbors.append((i, j + 1))   # right
        adjacency_dict[(i, j)] = neighbors


# test initialization of lattice and adjacency dictionary


# Ecalc function for the Ising model
def Ecalc_initial(state):
    """
    Calculate the energy of a given state in the Ising model.
    """
    energy = 0
    for i in range(LATTICE_SIZE):
        for j in range(LATTICE_SIZE):
            spin = state[i, j]
            neighbors = adjacency_dict[(i, j)]
            for neighbor in neighbors:
                neighbor_spin = state[neighbor]
                energy -= J * spin * neighbor_spin  # Interaction energy
    return energy / 2  # Each pair counted twice

def Ecalc_it(E_old, state, i, j):
    """ Calculate the new energy of the given state more efficiently by only considering the change in energy due to a single spin flip. """
# Constants

# 


# for i in range(NUM_TEMPS):

#     temp = START_TEMP + i * DT  # Example temperature
#     samples = metropolis_sampler(boltzmann, proposal_sampler, initial_state, temp, 1000)

# # Calc expected value magnetisation for each sample

#     magnetizations = np.mean(samples, axis=(1, 2))  # Average magnetization for each sample
#     expected_magnetization = np.mean(magnetizations)  # Expected value of magnetization

#     print(f"Temperature: {temp:.2f}, Expected Magnetization: {expected_magnetization:.4f}")