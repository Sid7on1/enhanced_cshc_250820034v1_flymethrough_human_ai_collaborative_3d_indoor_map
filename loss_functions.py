# loss_functions.py

"""
Custom loss functions for the computer_vision project.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LossFunction:
    """
    Base class for custom loss functions.
    """

    def __init__(self, name: str):
        """
        Initialize the loss function.

        Args:
            name (str): Name of the loss function.
        """
        self.name = name

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss.

        Args:
            predictions (Tensor): Predictions.
            targets (Tensor): Targets.

        Returns:
            Tensor: Loss.
        """
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def __init__(self):
        """
        Initialize the mean squared error loss function.
        """
        super().__init__("Mean Squared Error")

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the mean squared error.

        Args:
            predictions (Tensor): Predictions.
            targets (Tensor): Targets.

        Returns:
            Tensor: Mean squared error.
        """
        return (predictions - targets).pow(2).mean()

class MeanAbsoluteError(LossFunction):
    """
    Mean absolute error loss function.
    """

    def __init__(self):
        """
        Initialize the mean absolute error loss function.
        """
        super().__init__("Mean Absolute Error")

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the mean absolute error.

        Args:
            predictions (Tensor): Predictions.
            targets (Tensor): Targets.

        Returns:
            Tensor: Mean absolute error.
        """
        return (predictions - targets).abs().mean()

class VelocityThresholdLoss(LossFunction):
    """
    Velocity threshold loss function.
    """

    def __init__(self, threshold: float):
        """
        Initialize the velocity threshold loss function.

        Args:
            threshold (float): Velocity threshold.
        """
        super().__init__("Velocity Threshold")
        self.threshold = threshold

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the velocity threshold loss.

        Args:
            predictions (Tensor): Predictions.
            targets (Tensor): Targets.

        Returns:
            Tensor: Velocity threshold loss.
        """
        velocity = (predictions - targets).abs()
        return (velocity > self.threshold).float().mean()

class FlowTheoryLoss(LossFunction):
    """
    Flow theory loss function.
    """

    def __init__(self, alpha: float, beta: float):
        """
        Initialize the flow theory loss function.

        Args:
            alpha (float): Alpha parameter.
            beta (float): Beta parameter.
        """
        super().__init__("Flow Theory")
        self.alpha = alpha
        self.beta = beta

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the flow theory loss.

        Args:
            predictions (Tensor): Predictions.
            targets (Tensor): Targets.

        Returns:
            Tensor: Flow theory loss.
        """
        velocity = (predictions - targets).abs()
        return self.alpha * velocity + self.beta * velocity.pow(2)

class CustomLoss(LossFunction):
    """
    Custom loss function.
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize the custom loss function.

        Args:
            weights (Dict[str, float]): Weights for each loss component.
        """
        super().__init__("Custom")
        self.weights = weights

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the custom loss.

        Args:
            predictions (Tensor): Predictions.
            targets (Tensor): Targets.

        Returns:
            Tensor: Custom loss.
        """
        mse = MeanSquaredError()(predictions, targets)
        mae = MeanAbsoluteError()(predictions, targets)
        vtl = VelocityThresholdLoss(0.5)(predictions, targets)
        ftl = FlowTheoryLoss(0.1, 0.2)(predictions, targets)
        return self.weights["mse"] * mse + self.weights["mae"] * mae + self.weights["vtl"] * vtl + self.weights["ftl"] * ftl

def get_loss_function(name: str, **kwargs) -> LossFunction:
    """
    Get a loss function by name.

    Args:
        name (str): Name of the loss function.

    Returns:
        LossFunction: Loss function.
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "velocity_threshold":
        return VelocityThresholdLoss(**kwargs)
    elif name == "flow_theory":
        return FlowTheoryLoss(**kwargs)
    elif name == "custom":
        return CustomLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}")

# Example usage
if __name__ == "__main__":
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.0, 2.0, 3.0])
    loss_function = get_loss_function("custom", weights={"mse": 0.5, "mae": 0.3, "vtl": 0.1, "ftl": 0.1})
    loss = loss_function(predictions, targets)
    print(loss)