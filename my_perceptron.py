import numpy as np
import matplotlib.pyplot as plt

def my_perceptron(x, y, b=1, n_iter=int(1e6), rng = np.random.default_rng(seed=42)):
    """
    This function executes the perceptron algorithm from section 3.1.

    By means of the third (optional) argument it shall be distinguished whether a homogeneous linear hypothesis is to be learned.

    :param x:      (d, m)-Matrix consisting of the m training features in R^d
    :param y:      (m)-Vector consisting of the m associated labels {-1, +1}
    :param b:      Optional argument that learns a homogeneous linear hypothesis from the data for the value 0, otherwise a general linear   hypothesis
    :param n_iter: Maximum number of interations for the algorithm (by default infinity)

    :returns:
      - w     Vector containing the learned weights and bias in the form (w_1, w_2, ... w_d, b)
      - T     Integer of the number of executed steps in the algorithm
      - ws    Matrix with T+1 columns, the t-th column contains the t-th step Iterated of the procedure
      - RSs   Row vector containing the empirical risk for each vector ws

    """

    # Reading the dimension d and the data number m from x and y, respectively.
    d = x.shape[0]
    m = y.shape[0]
    
    # Case discrimination, whether homogeneous hypothesis should be learned
    xb = np.append(x, np.ones((1, m)), axis=0) if b==1 else x

    # Function to check the constraints
    check = lambda w, xb, y, m: y * (w @ xb)

    # Calculation of the obtained empirical risk
    RS = lambda chk: np.mean(chk <= 0)

    # Initialize extended weight vector
    w = np.zeros(d+b)
    # First entry in ws:
    ws = [ w ]
    # Empirical risk of the current w:
    RSs = [ RS(check(w, xb, y, m)) ]
    # Iteration variable of while loop
    t = 0
    while np.min(check(w, xb, y, m)) <= 0 and t < n_iter:
        # Find all unsatisfied constraints
        ch = check(w, xb, y, m)
        inds = np.where(ch<=0)[0]

        # Select an unfulfilled constraint
        i = rng.choice(inds)
        
        # Update according to iteration rule
        v = xb[:, i]
        w = w + y[i] * v

        # Save current w in ws
        ws.append(w)

        # Calculate empirical risk and store in RSs
        RSs.append(RS(check(w, xb, y, m)))

        # Increase step counter
        t += 1

    return [w, t, np.array(ws), np.array(RSs)]
