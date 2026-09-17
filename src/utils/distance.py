import numpy as np
from utils.constants import ANGSTROM_TO_BOHR


class ReactionChannelHysteresis:
    """Confirm a pair-condition reaction channel only after persistence.

    A candidate channel must be detected on ``required_steps`` consecutive
    trajectory frames.  A missing match or a different channel resets the
    previous candidate, preventing a transient threshold crossing from being
    assigned as a product channel.
    """

    def __init__(self, required_steps=1):
        if isinstance(required_steps, (bool, np.bool_)):
            raise ValueError("reaction_persistence_steps must be a positive integer")
        try:
            required_steps_int = int(required_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reaction_persistence_steps must be a positive integer") from exc
        if required_steps_int != required_steps or required_steps_int < 1:
            raise ValueError("reaction_persistence_steps must be a positive integer")
        self.required_steps = required_steps_int
        self.candidate_channel = None
        self.consecutive_steps = 0

    def reset(self):
        self.candidate_channel = None
        self.consecutive_steps = 0

    def update(self, q, pairs_to_test):
        """Evaluate the raw channel condition and return its confirmed state."""
        matched, channel = test_to_stop_specific(q, pairs_to_test)
        if not matched:
            self.reset()
            return False, "default"

        if channel == self.candidate_channel:
            self.consecutive_steps += 1
        else:
            self.candidate_channel = channel
            self.consecutive_steps = 1

        if self.consecutive_steps >= self.required_steps:
            return True, channel
        return False, "default"

    def restart_state(self):
        return {
            "required_steps": self.required_steps,
            "candidate_channel": self.candidate_channel,
            "consecutive_steps": self.consecutive_steps,
        }

    def restore_restart_state(self, state):
        if int(state.get("required_steps", -1)) != self.required_steps:
            raise ValueError(
                "Restart reaction_persistence_steps does not match the checkpoint"
            )
        consecutive_steps = int(state.get("consecutive_steps", 0))
        if consecutive_steps < 0 or consecutive_steps >= self.required_steps:
            raise ValueError("Restart reaction hysteresis state is invalid")
        candidate_channel = state.get("candidate_channel")
        if consecutive_steps == 0:
            candidate_channel = None
        elif candidate_channel is None:
            raise ValueError("Restart reaction hysteresis state is incomplete")
        self.candidate_channel = candidate_channel
        self.consecutive_steps = consecutive_steps

def distance_matrix(q):
    N = len(q) // 3
    coordinates = np.reshape(q, (N, 3))
    dist_matrix = np.sqrt(np.sum((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2, axis=-1))
    return dist_matrix

def test_to_stop_general(q, tol=16.0):

    dist = distance_matrix(q)
    
    # Check if any distance in the lower triangle of the matrix exceeds the tolerance
    too_large = np.any(dist[np.tril_indices(dist.shape[0], -1)] > tol)

    return too_large


def test_to_stop_specific(q, pairs_to_test):
    '''
    pairs_to_test = {
    'channel_1': [((2, 17), 'GT', 9.0)],  # Only one pair for this channel
    'channel_2': [((2, 18), 'GT', 9.0), ((2, 4), 'GT', 9.0), ((2, 5), 'GT', 9.0)],  # Multiple pairs for channel_2
    'channel_3': [((2, 7), 'GT', 9.0)],  # Only one pair for this channel
    # add more channels and pairs as needed
    }
    '''

    dist = distance_matrix(q)
    if not isinstance(pairs_to_test, dict) or not pairs_to_test:
        raise ValueError("pairs_to_test must be a non-empty reaction-channel dictionary")

    for channel, channel_spec in pairs_to_test.items():
        conditions = channel_spec.get("conditions") if isinstance(channel_spec, dict) else channel_spec
        if not isinstance(conditions, (list, tuple)):
            raise ValueError(f"Conditions for reaction channel {channel!r} must be a list")
        if not conditions:
            raise ValueError(f"Reaction channel {channel!r} must contain at least one condition")
        all_conditions_met = True  #Assume all pairs for this channel need to pass

        for condition in conditions:
            try:
                pair, cond, tolerance = condition
                i, j = pair
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed condition {condition!r} in reaction channel {channel!r}"
                ) from exc
            if not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)):
                raise ValueError(f"Atom indices in reaction channel {channel!r} must be integers")
            if i == j or not (0 <= i < len(dist)) or not (0 <= j < len(dist)):
                raise ValueError(f"Invalid atom pair {(i, j)!r} in reaction channel {channel!r}")
            cond = str(cond).upper()
            if cond not in {'GT', 'LT'}:
                raise ValueError(
                    f"Unknown condition {cond!r} in reaction channel {channel!r}; use 'GT' or 'LT'"
                )
            tolerance = float(tolerance)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    f"Distance tolerance in reaction channel {channel!r} must be non-negative"
                )

            if cond == 'GT':
                if not (dist[i, j] > tolerance * ANGSTROM_TO_BOHR):
                    all_conditions_met = False
                    break  # Stop checking this channel if one condition fails
            else:  # cond == 'LT'
                if not (dist[i, j] < tolerance * ANGSTROM_TO_BOHR):
                    all_conditions_met = False
                    break  # Stop checking this channel if one condition fails

        if all_conditions_met:
            return (True, channel)  # If all pairs met conditions, return the channel

    return (False, 'default')
