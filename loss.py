import numpy as np
from typing import Tuple

def cross_entropy_loss(y_hat:float, y:float) -> Tuple[float, float]:
  '''
  Returns the cross entropy loss and the derivative for the output.
  '''
  if np.all(y == 1.0):
    return -np.log(y_hat), -1.0 / y_hat
  else:
    return -np.log(1 - y_hat), 1.0 / y_hat

def quadratic_error(y_hat:float, y:float) -> Tuple[float, float]:
  '''
    Returns the cross entropy loss and the derivative for the output.
  '''
  return 0.5 * ((y - y_hat) ** 2), (y_hat - y)
