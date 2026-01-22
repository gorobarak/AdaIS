import numpy as np

class CorrectnessScorer:
    """
    A scorer that projects hidden states onto the correctness direction.

    Attributes:
        direction (np.ndarray): The direction vector from incorrect to correct (μ)
        origin (np.ndarray): The new origin vector (o)
        direction_norm (float): The norm of the direction vector
    """

    def __init__(self, direction: np.ndarray, origin: np.ndarray):
        """
        Initialize the scorer with a direction and origin.

        Args:
            direction: The direction vector from incorrect to correct
            origin: The origin vector (midpoint of centroids)
        """
        self.direction = direction / np.linalg.norm(direction)
        self.origin = origin
        

    def normalize_to_0_1(self, cosine_sim):
        """Normalize cosine similarity from [-1, 1] to [0, 1]."""
        return (cosine_sim + 1) / 2

    
    def score(self, hidden_state: np.ndarray) -> float:
        """
        Compute the correctness score for a hidden state.

        Formula: score(h) = shift_to_0_1(cosine_similarity(direction, (h-origin))

        Args:
            hidden_state: The hidden state vector to score

        Returns:
            The correctness score (higher means more likely correct)
        """
        assert len(hidden_state.shape) > 1, "Hidden state expected to have batch dimension"
        centered = hidden_state - self.origin
        norms = np.linalg.norm(centered, axis=-1)
        norms = np.where(norms == 0, 1e-8, norms)  # Avoid division by zero
        cosine_sim = np.dot(centered, self.direction) / norms
        scores = self.normalize_to_0_1(cosine_sim)
        if scores.shape[0] == 1:
            return scores.item()
        return scores

    def save(self, filepath: str):
        """Save the scorer to disk."""
        np.savez(filepath, direction=self.direction, origin=self.origin)

    @classmethod
    def load(cls, filepath: str):
        """Load a scorer from disk."""
        data = np.load(filepath)
        return cls(direction=data["direction"], origin=data["origin"])
