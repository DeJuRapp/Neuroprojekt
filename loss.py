import numpy as np
from typing import Tuple

def cross_entropy_loss(y_hat:float, y:float) -> Tuple[float, float]:
    '''
    Returns the cross entropy loss and the derivative for the output.
    '''
    if y == 1:
      return -np.log(y_hat), -1.0 / y_hat
    else:
      return -np.log(1 - y_hat), 1.0 / y_hat