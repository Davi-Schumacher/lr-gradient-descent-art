from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def predict(X: np.ndarray, b: np.ndarray, b_0: float) -> np.ndarray:
    """Returns predictions given by the equation y = bx + b_0.

    Args:
        X (np.ndarray): An N x 1 dimensional array of numbers.
        b (np.ndarray): A singleton array containing the slope.
        b_0 (float): A number giving the y-intercept.

    Returns:
        np.ndarray: An N x 1 dimensional array of linear predictions.
    """

    return X * b + b_0


def grad_0(X: np.ndarray, y: np.ndarray, b: np.ndarray, b_0: float) -> float:
    """Calculates the partial derivative of the cost function for linear
    regression gradient descent with respect to b_0, or the y-intercept.
    The cost function used is MSE.

    Args:
        X (np.ndarray): An N x 1 dimensional array of numbers.
        y (np.ndarray): An N x 1 dimensional array of linear predictions.
        b (np.ndarray): A singleton array containing the slope.
        b_0 (float): A number giving the y-intercept.

    Returns:
        float: The value given by the partial derivative of the linear
        regression cost function with respect to b_0.
    """

    N = y.shape[0]
    y_hat = predict(X, b, b_0).reshape([N, 1])

    return np.sum(y_hat - y)


def grad_1(X: np.ndarray, y: np.ndarray, b: np.ndarray, b_0: float) -> np.ndarray:
    """Calculates the partial derivative of the cost function for linear
    regression gradient descent with respect to b, or the slope.
    The cost function used is MSE.

    Args:
        X (np.ndarray): An N x 1 dimensional array of numbers.
        y (np.ndarray): An N x 1 dimensional array of linear predictions.
        b (np.ndarray): A singleton array containing the slope.
        b_0 (float): A number giving the y-intercept.

    Returns:
        np.ndarray: A singleton array containing the value given by the
        partial derivative of the linear regression cost function with
        respect to b_0.
    """

    N = y.shape[0]
    y_hat = predict(X, b, b_0).reshape([N, 1])
    residuals = (y_hat - y).reshape([1, -1])
    return np.sum(np.matmul(residuals, X))


def generate_somewhat_linear_data(
    num_points: int,
    val_range: tuple = (0, 1),
    max_epsilon: float = 0.5,
    seed: int = 123,
) -> Tuple[np.ndarray]:
    """Takes num_data linear data points and jitters them by a maximum
    amount given by max_epsilon. Val_range is a tuple that gives the min
    and max values to use in generating the linear data points.

    Args:
        num_points (int): The number of data points to generate.
        val_range (tuple, optional): The min and max values to use for
            generating the linear data points. Defaults to (0, 1).
        max_epsilon (float, optional): The max amount of jitter to use.
            Defaults to 0.5.
        seed (int, optional): A random seed to use. Defaults to 123.

    Returns:
        Tuple[np.ndarray]: A tuple with two N dimensional arrays. The
        first contains the x values and the second contains the y values.
    """

    np.random.seed(seed)
    x = np.linspace(val_range[0], val_range[1], num_points)
    epsilons = np.random.uniform(-max_epsilon, max_epsilon, num_points)
    y = x + epsilons
    y = np.abs(y - np.max(y)) / np.max(y)

    return x, y


def create_line(b: np.ndarray, b_0: float, color: str) -> plt.Line2D:
    """Generates a line object for adding to a plot. Calculates the line
    with linear points and a given slope and y-intercept.

    Args:
        b (np.ndarray): The slope to use for the line.
        b_0 (float): The y-intercepy to use for the line.
        color (str): The color to use for the line.

    Returns:
        plt.Line2D: A 2D line object for adding to a pyplot.
    """

    x = np.linspace(0, 1, 100)
    new_y = b * x + b_0
    line = plt.Line2D(x, new_y, color=color)

    return line


def main(num_points: int, seed: int = 123):
    """A gradient descent implementation of linear regression that also
    makes pretty plots.

    Args:
        num_points (int): The number of points to use when generating
            somewhat linear data.
        seed (int, optional): A random seed for reproducibility.
            Defaults to 123.
    """

    # Create some somewhat linear data
    X, y = generate_somewhat_linear_data(num_points)
    X = X.reshape([num_points, -1])
    y = y.reshape([num_points, -1])

    # I like reproducibility, usually
    np.random.seed(seed)

    # Initialize the initial parameters
    b = np.array([1])
    b_0 = 0

    # Initialize the plot and add the line given by the initial parameters
    _ = plt.figure(figsize=(4, 3))
    ax = plt.axes()
    ax.scatter(X, y)
    ax.add_line(create_line(b, b_0, "tab:green"))

    convergence_condition = 0.001  # Diff between last_b and b that stops descent
    learning_rate = 0.01  # Size of each parameter update
    last_b = 999  # Set to anything large enough to not meet convergence condition

    i = 0  # Increments to help with adding lines

    # Perform gradient descent until convergence condition is met
    while np.abs(last_b - b) > convergence_condition:

        # Copy the last parameter values before updating
        last_b = b
        last_b_0 = b_0

        # Update the parameters using the gradient of the cost function
        b_0 = last_b_0 - learning_rate * grad_0(X, y, last_b, last_b_0)
        b = last_b - learning_rate * grad_1(X, y, last_b, last_b_0)
        print(f"b: {b}")
        print(f"b_0: {b_0}")
        print(f"diff: {np.abs(last_b - b)}")

        # Add a line every 10 iterations
        i += 1
        if i % 10 == 0:
            ax.add_line(create_line(b, b_0, "tab:blue"))

    # Add the final solution line and show the plot
    ax.add_line(create_line(b, b_0, "tab:red"))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":

    main(30, 123)
