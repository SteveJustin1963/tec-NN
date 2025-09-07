
# tec-NN

**TEC-1 Neural Network**

---

## Definition

Artificial Neural Networks (ANNs) are computational systems inspired by the structure and function of biological neural networks in animal brains. An ANN is composed of *artificial neurons* (nodes), organized into layers (input, hidden, output). Each neuron receives inputs, multiplies them by trainable weights, adds a bias, applies a nonlinear **activation function**, and produces an output.

Mathematically, for a neuron $j$:

$$
y_j = f\left(\sum_i w_{ij} x_i + b_j\right)
$$

where:

* $x_i$ = input values
* $w_{ij}$ = trainable weights
* $b_j$ = bias
* $f$ = activation function (sigmoid, ReLU, softmax, etc.)

The **power of neural networks** comes from their ability to approximate nonlinear functions, learn from labeled or unlabeled data, and generalize to unseen patterns.

**Key ANN types include:**

* **Feedforward Neural Networks (FFNs):** Simple, one-directional flow of information.
* **Recurrent Neural Networks (RNNs):** Include loops to model sequential data.
* **Convolutional Neural Networks (CNNs):** Specialized for spatial data like images.
* **Generative Adversarial Networks (GANs):** Use competition between generator and discriminator networks to create realistic synthetic data.

The **training process** adjusts weights and biases by minimizing an error or cost function (e.g., Mean Squared Error, Cross-Entropy Loss) using **backpropagation** and optimization algorithms such as **gradient descent** or adaptive optimizers like **Adam**.

Applications span **pattern recognition (handwriting, speech, faces), robotics, reinforcement learning, natural language processing, drug discovery, and autonomous vehicles**.

---

## Pseudo-code (Basic Neural Network Loop)

```pseudo
Program Neural_Network

  Initialize:
    Set number_of_inputs = 2
    Set number_of_outputs = 1
    Set number_of_hidden_layers = 2
    Set learning_rate = 0.1

    Initialize weight matrices with random values (e.g., Xavier initialization)
    Initialize bias vectors with small values (e.g., 0 or random small)

  Repeat until convergence or max_epochs:
    For each training example (inputs, target):

      Forward_Propagate(inputs, hidden, outputs)

      Back_Propagate(inputs, hidden, outputs, target,
                     weights, biases, learning_rate)

  End Program
```

---

## Forward Propagation (Conceptual)

```pseudo
Function Forward_Propagate(inputs, hidden, outputs, weights, biases):

  // Hidden layer
  For each hidden_neuron:
    z = Σ (input * weight) + bias
    hidden_value = Activation(z)

  // Output layer
  For each output_neuron:
    z = Σ (hidden * weight) + bias
    output_value = Activation(z)

  return outputs
End Function
```

---

## Backpropagation (Conceptual)

```pseudo
Function Back_Propagate(inputs, hidden, outputs, targets,
                        weights, biases, learning_rate):

  // Step 1: Compute output errors
  For each output_neuron:
    error = target - output
    gradient = error * Activation_Derivative(output)
    store gradient

  // Step 2: Update output weights and biases
  For each hidden_neuron, output_neuron:
    weight += learning_rate * gradient * hidden_value
  bias += learning_rate * gradient

  // Step 3: Propagate errors to hidden layer
  For each hidden_neuron:
    error = Σ (output_gradient * corresponding_weight)
    gradient = error * Activation_Derivative(hidden_value)

  // Step 4: Update hidden weights and biases
  For each input_neuron, hidden_neuron:
    weight += learning_rate * gradient * input_value
  bias += learning_rate * gradient

End Function
```

---

## Potential Issues and Improvements

1. **Activation Functions**

   * Sigmoid suffers from vanishing gradients → use **ReLU** or **Leaky ReLU** for hidden layers.
   * Softmax is best for multi-class outputs.

2. **Weight Initialization**

   * Replace uniform random with **Xavier/Glorot** (for sigmoid/tanh) or **He initialization** (for ReLU).

3. **Optimizers**

   * Instead of fixed learning rate, use adaptive optimizers (Adam, RMSProp).

4. **Batch Training**

   * Instead of single samples, process data in **mini-batches** for faster convergence.

5. **Regularization**

   * Add **Dropout** to prevent overfitting.
   * Add **L2 regularization** for weight decay.

6. **Scalability**

   * Use vectorized/matrix operations instead of nested loops.
   * GPU acceleration for large datasets.

---

# MINT Implementations

Below are TEC-1 MINT implementations of neural networks, written to comply strictly with the MINT manual.

---

## Forward Propagation (MINT, 2–3–2 network)

```mint
// Forward propagation: 2 inputs → 3 hidden → 2 outputs

:I                                // Initialize network
  2 n!                            // number of inputs
  3 h!                            // number of hidden neurons
  2 o!                            // number of outputs

  [0 0] i!                        // input layer
  [0 0 0] s!                      // hidden layer
  [0 0] r!                        // output layer

  [1 2 3 4 5 6] w!                // input→hidden weights (2×3)
  [7 8 9 1 2 3] v!                // hidden→output weights (3×2)

  [1 1 1] b!                      // hidden biases
  [1 1] c!                        // output biases
;

:A                                // ReLU activation
  k!                              // input value
  k 0 > ( k ) /E ( 0 )
;

:H                                // Hidden layer calculation
  h(
    0 t!                          // reset sum
    n(
      /i h * /j + m!              // weight index = i*h + j
      w m ? i /i ? * t + t!       // accumulate weighted input
    )
    t b /j ? + A s /j ?!          // add bias, apply activation, store
  )
;

:O                                // Output layer calculation
  o(
    0 t!                          // reset sum
    h(
      /i o * /j + m!              // weight index
      v m ? s /i ? * t + t!       // accumulate weighted hidden
    )
    t c /j ? + A r /j ?!          // add bias, apply activation, store
  )
;

:F                                // Run forward pass
  H O
;
```

---

## Backpropagation (MINT, 2–3–2 network)

```mint
// Backpropagation: update weights using ReLU derivative

:D                                // ReLU derivative
  k!                              // input value
  k 0 > ( 1 ) /E ( 0 )
;

:E                                // Output errors
  o(
    t /i ? r /i ? -               // target - output
    r /i ? D * e /i ?!            // derivative * error → store
  )
;

:Herr                             // Hidden errors
  h(
    0 u!                          // reset sum
    o(
      /j o * /i + m!              // weight index
      v m ? e /j ? * u + u!       // accumulate weighted errors
    )
    u s /i ? D * d /i ?!          // apply derivative, store
  )
;

:Upd                              // Update weights
  // Hidden→Output
  h(
    o(
      /i o * /j + m!              // weight index
      v m ? e /j ? s /i ? * l * + v m ?!
    )
  )
  // Input→Hidden
  n(
    h(
      /i h * /j + m!              // weight index
      w m ? d /j ? i /i ? * l * + w m ?!
    )
  )
;

:B                                // Backpropagation step
  E Herr Upd
;
```

---

## Hopfield Network (MINT, 4×4 grid = 16 neurons)

```mint
// Hopfield Network (4x4 = 16 neurons)

// Variables:
// n  = number of neurons
// s  = neuron state array [16 elements]
// w  = weight matrix [256 elements = 16x16]
// e  = energy
// h  = weighted sum temp

:I                                // Initialize network
  16 n!                           // 16 neurons
  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] s!   // states
  256 /A w!                       // weight matrix
  n n * ( 0 w /i ?! )             // init weights to 0
  0 e!                            // reset energy
;

:W                                // Initialize weights (Hebbian rule)
  n(
    n(
      /i /j = ( 0 ) /E (          // skip self-connections
        /i n * /j + k!
        1 w k ?!                  // set weight to 1
      )
    )
  )
;

:P                                // Print state as 4x4 grid
  `State:` /N
  4(
    4(
      /j 4 * /i + k!
      s k ? 1 = ( `■ ` ) /E ( `□ ` )
    )
    /N
  )
;

:U                                // Update one neuron
  k! 0 h!                         // reset sum
  n(
    /i k = /F (
      /i n * k + j!
      w j ? s /i ? * h + h!
    )
  )
  h 0 > ( 1 s k ?! ) /E ( 0 s k ?! )
;

:E                                // Calculate energy
  0 e!
  n(
    n(
      /i /j = /F (
        /i n * /j + k!
        w k ? s /i ? s /j ? * + e!
      )
    )
  )
  e -1 * e!
;

:S                                // One simulation step
  n( /i U )                       // update all neurons
  E
  `Energy: ` e . /N
  P
;

:L                                // Load test pattern (cross)
  16( 0 s /i ?! )
  5 s ?! 6 s ?! 9 s ?! 10 s ?!
;

:R                                // Run simulation
  I W L
  `Initial:` /N
  P
  10(
    `Step ` /i 1 + . `:` /N
    S
  )
;
```

---

## Related Projects

* [tec-AI](https://github.com/SteveJustin1963/tec-AI) – General AI experiments
* [tec-memR](https://github.com/SteveJustin1963/tec-memR) – Memory and recall systems
* [tec-Generative-Adversarial-Network](https://github.com/SteveJustin1963/tec-Generative-Adversarial-Network) – GAN experiments
* [tec-BOT](https://github.com/SteveJustin1963/tec-BOT) – Bot controllers
* [tec-BRAIN](https://github.com/SteveJustin1963/tec-BRAIN) – Brain-inspired models
* [tec-GA](https://github.com/SteveJustin1963/tec-GA) – Genetic Algorithms

---

## References

* [Wikipedia – Neural network](https://en.wikipedia.org/wiki/Neural_network)
* [IBM – What are neural networks?](https://www.ibm.com/cloud/learn/neural-networks)
* [Machine Learning Mastery – Pooling layers in CNNs](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/#:~:text=Average%20pooling%20involves%20calculating%20the,6%C3%976%20feature%20map.)
* [Analytics India Mag – Types of ANNs](https://analyticsindiamag.com/6-types-of-artificial-neural-networks-currently-being-used-in-todays-technology/)
* [Investopedia – Neural Networks](https://www.investopedia.com/terms/n/neuralnetwork.asp)

---


---

# 1. **Forward Propagation** with actual numbers
# 2. **Backpropagation update** for one iteration
# 3. A **mini Hopfield run** with a simple pattern



# Worked Example: Forward + Backward Pass

We’ll use the **2–3–2 network** defined in the MINT code.

### Initialization

* **Inputs:**

  ```
  i = [1, 0]
  ```

* **Weights (Input→Hidden):**

  ```
  w = [
    [0.5, -0.3, 0.8],   // from input1
    [0.2,  0.7, -0.5]   // from input2
  ]
  ```

* **Biases (Hidden):**

  ```
  b = [0.1, -0.2, 0.05]
  ```

* **Weights (Hidden→Output):**

  ```
  v = [
    [0.4, 0.6],   // from hidden1
    [-0.1, 0.2],  // from hidden2
    [0.3, -0.4]   // from hidden3
  ]
  ```

* **Biases (Output):**

  ```
  c = [0.0, 0.1]
  ```

* **Target (desired output):**

  ```
  t = [1, 0]
  ```

---

## Step 1: Forward Propagation

**Hidden Layer**

For each hidden neuron $h_j$:

$$
h_j = f\Big(\sum_i x_i w_{ij} + b_j\Big)
$$

with ReLU activation.

* $h_1 = f(1*0.5 + 0*0.2 + 0.1) = f(0.6) = 0.6$
* $h_2 = f(1*(-0.3) + 0*0.7 - 0.2) = f(-0.5) = 0$
* $h_3 = f(1*0.8 + 0*(-0.5) + 0.05) = f(0.85) = 0.85$

So:

```
s = [0.6, 0, 0.85]
```

**Output Layer**

For each output neuron $o_j$:

$$
o_j = f\Big(\sum_i h_i v_{ij} + c_j\Big)
$$

* $o_1 = f(0.6*0.4 + 0* -0.1 + 0.85*0.3 + 0.0) = f(0.24 + 0.255) = f(0.495) = 0.495$
* $o_2 = f(0.6*0.6 + 0*0.2 + 0.85*(-0.4) + 0.1) = f(0.36 - 0.34 + 0.1) = f(0.12) = 0.12$

So:

```
r = [0.495, 0.12]
```

---

## Step 2: Compute Output Error

Target: \[1, 0]
Output: \[0.495, 0.12]

Error:

```
e = [1 - 0.495, 0 - 0.12] = [0.505, -0.12]
```

For ReLU derivative:

* derivative is 1 if value > 0
* here both outputs > 0 → derivative = 1

So gradients:

```
g = [0.505, -0.12]
```

---

## Step 3: Weight Update

Learning rate $\eta = 0.1$.

**Hidden→Output Updates**

$$
v_{ij} = v_{ij} + \eta \cdot g_j \cdot h_i
$$

For $g = [0.505, -0.12]$:

* From $h_1 = 0.6$:

  * $v_{11} = 0.4 + 0.1*0.505*0.6 = 0.4303$
  * $v_{12} = 0.6 + 0.1*(-0.12)*0.6 = 0.5928$

* From $h_2 = 0.0$: no change.

* From $h_3 = 0.85$:

  * $v_{31} = 0.3 + 0.1*0.505*0.85 = 0.3429$
  * $v_{32} = -0.4 + 0.1*(-0.12)*0.85 = -0.4102$

**Output Bias Updates**

$$
c_j = c_j + \eta \cdot g_j
$$

* $c_1 = 0.0 + 0.1*0.505 = 0.0505$
* $c_2 = 0.1 + 0.1*(-0.12) = 0.088$

---

## Step 4: Hidden Error Propagation

Hidden errors:

$$
d_i = f'(h_i) \cdot \sum_j g_j v_{ij}
$$

* $h_1 = 0.6 > 0$ → derivative = 1

  * $d_1 = 0.505*0.4 + (-0.12*0.6) = 0.202 - 0.072 = 0.13$

* $h_2 = 0$ → derivative = 0 → $d_2 = 0$

* $h_3 = 0.85 > 0$ → derivative = 1

  * $d_3 = 0.505*0.3 + (-0.12* -0.4) = 0.1515 + 0.048 = 0.1995$

So:

```
d = [0.13, 0, 0.1995]
```

---

## Step 5: Input→Hidden Updates

$$
w_{ij} = w_{ij} + \eta \cdot d_j \cdot x_i
$$

* For input1 = 1:

  * $w_{11} = 0.5 + 0.1*0.13*1 = 0.513$
  * $w_{12} = -0.3 + 0.1*0*1 = -0.3$
  * $w_{13} = 0.8 + 0.1*0.1995*1 = 0.82$

* For input2 = 0: no updates.

**Hidden Bias Updates**

$$
b_j = b_j + \eta \cdot d_j
$$

* $b_1 = 0.1 + 0.1*0.13 = 0.113$
* $b_2 = -0.2 + 0.1*0 = -0.2$
* $b_3 = 0.05 + 0.1*0.1995 = 0.07$

---

# Result After One Training Step

Updated Parameters:

* **w (input→hidden):**

  ```
  [[0.513, -0.3, 0.82],
   [0.2,   0.7, -0.5]]
  ```
* **b (hidden biases):** `[0.113, -0.2, 0.07]`
* **v (hidden→output):**

  ```
  [[0.4303, 0.5928],
   [-0.1,   0.2],
   [0.3429, -0.4102]]
  ```
* **c (output biases):** `[0.0505, 0.088]`

---

# Mini Hopfield Example

Initialize 4×4 grid, load a **cross pattern** (as in the MINT `:L` function).

Initial State (X = active neuron):

```
□ ■ □ ■
□ □ □ □
□ □ □ □
□ ■ □ ■
```

Run `:R` → the network will converge to a stable attractor (energy decreases at each step).

---

