import numpy as np

class CorrectnessScorer:
    """
    A scorer that projects hidden states onto the correctness direction.

    Attributes:
        direction (np.ndarray): The direction vector from incorrect to correct (μ)
        origin (np.ndarray): The new origin vector (o)
        direction_norm (float): The norm of the direction vector
        quantiles (np.ndarray): 0.2, 0.4, 0.6, 0.8 quantiles on a validation set
    """

    def __init__(self, direction: np.ndarray, origin: np.ndarray, quantiles: np.ndarray):
        """
        Initialize the scorer with a direction and origin.

        Args:
            direction: The direction vector from incorrect to correct
            origin: The origin vector (midpoint of centroids)
        """
        self.direction = direction
        self.origin = origin
        self.direction_norm = np.linalg.norm(direction)
        self.quantiles  = quantiles

    def score(self, hidden_state: np.ndarray) -> float:
        """
        Compute the correctness score for a hidden state.

        Formula: score(h) = (1 / ||μ||) * μ^T · (h - o)

        Args:
            hidden_state: The hidden state vector to score

        Returns:
            The correctness score (higher means more likely correct)
        """
        if hidden_state.ndim == 1:
            # Single vector
            centered = hidden_state - self.origin
            score = np.dot(self.direction, centered) / self.direction_norm
            return float(score)
        else:
            # Batch of vectors
            centered = hidden_state - self.origin
            scores = np.dot(centered, self.direction) / self.direction_norm
            return scores

    def save(self, filepath: str):
        """Save the scorer to disk."""
        np.savez(filepath, direction=self.direction, origin=self.origin, quantiles=self.quantiles)

    @classmethod
    def load(cls, filepath: str):
        """Load a scorer from disk."""
        data = np.load(filepath)
        return cls(direction=data["direction"], origin=data["origin"], quantiles=data["quantiles"])
