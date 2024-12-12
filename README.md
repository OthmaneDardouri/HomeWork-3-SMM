# Homework: Optimization via Gradient Descent and Stochastic Gradient Descent

## Part 1: Gradient Descent (GD)

### Goal
Solve optimization problems for differentiable functions using the Gradient Descent (GD) method. Implement the GD algorithm and test it on a variety of functions.

### Instructions
1. Implement the GD algorithm with the following structure:
   - **Inputs**:
     - `f`: Function to optimize (Python callable).
     - `grad_f`: Gradient of the function (Python callable).
     - `x0`: Initial point (n-dimensional array).
     - `kmax`: Maximum number of iterations.
     - `tolf`: Relative tolerance for convergence based on the gradient.
     - `tolx`: Tolerance for convergence in the input domain.
   - **Outputs**:
     - `x`: Array of all iterates.
     - `k`: Number of iterations to converge.
     - `f_val`: Array of function values for each iterate.
     - `grads`: Array of gradients for each iterate.
     - `err`: Array of gradient norms for each iterate.

2. Extend GD with backtracking for step size selection:
   - Use a provided backtracking algorithm to dynamically adjust the step size `α`.

3. Test the algorithm on the following functions:
   - **Function 1**: \( f(x_1, x_2) = (x_1 - 3)^2 + (x_2 - 1)^2 \)
   - **Function 2**: \( f(x_1, x_2) = 10(x_1 - 1)^2 + (x_2 - 2)^2 \)
   - **Function 3**: \( f(x) = \frac{1}{2} \|Ax - b\|_2^2 \), where \( A \) is a Vandermonde matrix.
   - **Function 4**: \( f(x) = \frac{1}{2} \|Ax - b\|_2^2 + \frac{\lambda}{2} \|x\|_2^2 \), with \( \lambda \in [0, 1] \).
   - **Function 5**: \( f(x) = x^4 + x^3 - 2x^2 - 2x \)

4. Experiment with:
   - Fixed step sizes (`α > 0`) and backtracking.
   - Visualizing gradient norm \( \|\nabla f(x_k)\|_2 \) and convergence rate.
   - Plotting error \( \|x_k - x^*\|_2 \) when \( x^* \) is known.

5. Analyze the non-convex function (Function 5):
   - Plot it in \([-3, 3]\).
   - Test convergence for various initial points and step sizes.
   - Observe global vs. local minima convergence.

6. Optional: For Functions 1 and 2, plot contour lines and the GD path using `plt.contour`.

---

## Part 2: Stochastic Gradient Descent (SGD)

### Goal
Solve large-scale optimization problems common in Machine Learning using Stochastic Gradient Descent (SGD).

### Instructions
1. Implement the SGD algorithm with the following structure:
   - **Inputs**:
     - `l`: Loss function \( \ell(w; D) \) (Python callable).
     - `grad_l`: Gradient of the loss function (Python callable).
     - `w0`: Initial weights (n-dimensional array, default randomly sampled).
     - `data`: Tuple \((X, Y)\), where \( X \) is the input data, \( Y \) is the output data.
     - `batch_size`: Number of samples in each batch.
     - `n_epochs`: Number of epochs to repeat the iterations.
   - **Outputs**:
     - `w`: Array of weights for each iteration.
     - `f_val`: Loss function values for each epoch.
     - `grads`: Gradients for each epoch.
     - `err`: Gradient norms for each epoch.

2. Test SGD on the MNIST dataset:
   - Select two digits (user-specified).
   - Split dataset into training and testing sets.
   - Train using logistic regression for binary classification.

3. Compare GD and SGD:
   - Use both methods to compute the optimal weights \( w^* \).
   - Evaluate and compare the results for different digits and training set sizes.

4. Analyze results:
   - Comment on the accuracy of the logistic regression classifier.
   - Observe differences in convergence behavior between GD and SGD.
