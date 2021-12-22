import pytest
from neurons import RBF
import numpy as np

@pytest.fixture
def basic_rbfs():
    n = RBF()
    n.num_neurons = 4
    n._c = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(4, 1, 2)
    n._b = np.ones((4,1,1))
    return n

class TestRBF:

    @pytest.mark.parametrize("batch_size", (1, 2, 3, 6))
    def test_propagate(self, basic_rbfs:RBF, batch_size:int):
        inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [-1.0e15, 0.0]]).reshape(-1, batch_size, 2)
        e_to_the_minus_half = np.exp(-0.5)
        e_to_the_minus_one = np.exp(-1.0)
        e_to_the_minus_two_five = np.exp(-0.25)
        expected = np.array([[1.0, e_to_the_minus_half, e_to_the_minus_half, e_to_the_minus_one],
                             [e_to_the_minus_half, 1.0, e_to_the_minus_one, e_to_the_minus_half],
                             [e_to_the_minus_half, e_to_the_minus_one, 1.0, e_to_the_minus_half],
                             [e_to_the_minus_one, e_to_the_minus_half, e_to_the_minus_half, 1.0],
                             [e_to_the_minus_two_five, e_to_the_minus_two_five, e_to_the_minus_two_five, e_to_the_minus_two_five],
                             [0.0, 0.0, 0.0, 0.0]])
        for i, batch in enumerate(inputs):
            returned_vals = basic_rbfs.propagate(batch.reshape(1, *batch.shape))
            assert np.array_equal(returned_vals, expected[i * batch_size : i * batch_size + batch_size].reshape(4, batch_size, 1))

    @pytest.mark.parametrize("batch_size", (1, 2, 3, 6))
    def test_back_propagate(self, basic_rbfs:RBF, batch_size:int):
        inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [-1.0e15, 0.0]]).reshape(-1, batch_size, 2)
        
        for i, batch in enumerate(inputs):
            _ = basic_rbfs.propagate(batch.reshape(1, *batch.shape))
            d_x = basic_rbfs.back_propagate(np.ones((1, batch_size)), True, 1.0)
