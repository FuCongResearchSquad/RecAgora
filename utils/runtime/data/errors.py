class DataContractError(ValueError):
    """Raised when a sample violates the dataset contract."""


class DropSample(Exception):
    """Internal signal used when a configured policy drops a bad sample."""
