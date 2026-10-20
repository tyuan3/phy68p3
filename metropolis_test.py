import numpy as np

def metropolis_sampler(target_density, proposal_sampler, initial_state, num_samples):
    """
    Metropolis-Hastings sampler.

    Parameters:
    - target_density: function that computes the target density at a given state.
    - proposal_sampler: function that generates a proposal state given the current state.
    - initial_state: initial state of the Markov chain.
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
        acceptance_ratio = target_density(proposal_state) / target_density(current_state)
        acceptance_probability = min(1, acceptance_ratio)

        # Accept or reject the proposal
        if np.random.rand() < acceptance_probability:
            current_state = proposal_state

        samples[i] = current_state

    return samples