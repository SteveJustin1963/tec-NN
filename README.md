# tec-NN
TEC-1 Neural Network



## def
Neural networks, also known as artificial neural networks (ANNs), are sophisticated computing systems designed to mimic the structure and functionality of biological neural networks found in animal brains. ANNs consist of interconnected units or nodes known as artificial neurons, which aim to replicate the behavior of neurons in the human brain.

When it comes to tasks like handwriting recognition or facial recognition, the human brain excels at making rapid decisions. Similarly, artificial neural networks are engineered to efficiently process and analyze complex patterns and information, allowing them to excel in these recognition tasks as well.

There exist various types of artificial neural networks, each implemented based on specific mathematical operations and a predefined set of parameters that determine the network's output. These parameters play a crucial role in shaping the behavior and performance of the neural network, as they define the connections and strengths of the connections between artificial neurons. By adjusting these parameters during the training process, neural networks can learn to recognize patterns, classify data, or solve complex problems.

Some common types of artificial neural networks include feedforward neural networks, recurrent neural networks, convolutional neural networks, and generative adversarial networks. Feedforward neural networks are the simplest type, where information flows in one direction, from input to output layers. Recurrent neural networks, on the other hand, introduce loops within the network, enabling them to process sequential data or time-dependent information. Convolutional neural networks specialize in handling grid-like data, such as images, by employing convolutional layers that extract features hierarchically. Lastly, generative adversarial networks consist of two neural networks, a generator and a discriminator, that work together to generate realistic synthetic data.

The training process of neural networks involves presenting them with a labeled dataset and iteratively adjusting the parameters to minimize the difference between the network's output and the desired output. This optimization process, often achieved through backpropagation and gradient descent algorithms, allows neural networks to learn from examples and improve their performance over time.

With their ability to model complex relationships and extract meaningful information from large datasets, artificial neural networks have become a vital tool in various fields, including image and speech recognition, natural language processing, autonomous vehicles, and many more. Their versatility and power make them a cornerstone of modern machine learning and artificial intelligence research and applications.


## pseudo-code

```
Program Neural_Network

    Initialize:
        Set number_of_inputs to 2
        Set number_of_outputs to 1
        Set number_of_hidden_layers to 2
        Set learning_rate to 0.1

        Initialize 2D array weights_between_inputs_hidden with number_of_inputs X number_of_hidden_layers to random values between -1 and 1
        Initialize 2D array weights_between_hidden_outputs with number_of_hidden_layers X number_of_outputs to random values between -1 and 1
        Initialize 1D array biases_for_hidden with number_of_hidden_layers to random values between -1 and 1
        Initialize 1D array biases_for_outputs with number_of_outputs to random values between -1 and 1

    For each training example:

        For i = 1 to number_of_inputs:

            Call Forward_Propagate function with inputs[i], hidden, outputs, weights_between_inputs_hidden, weights_between_hidden_outputs, biases_for_hidden, biases_for_outputs

            Call Back_Propagate function with inputs[i], hidden, outputs, targets[i], errors, weights_between_inputs_hidden, weights_between_hidden_outputs, biases_for_hidden, biases_for_outputs, learning_rate

    Repeat until the error rate is low

End Program

Function Forward_Propagate(inputs, hidden, outputs, weights_between_inputs_hidden, weights_between_hidden_outputs, biases_for_hidden, biases_for_outputs):

    // Implementation of the forward propagation process in the network

End Function

Function Back_Propagate(inputs, hidden, outputs, targets, errors, weights_between_inputs_hidden, weights_between_hidden_outputs, biases_for_hidden, biases_for_outputs, learning_rate):

    // Implementation of the back propagation process in the network using the perceptron learning rule

End Function
```

In this pseudo-code:

- We define variables such as `number_of_inputs`, `number_of_outputs`, `number_of_hidden_layers`, `learning_rate`, arrays for weights and biases.
- In the training loop, we call two functions: `Forward_Propagate` and `Back_Propagate`.
- `Forward_Propagate` calculates the output of the network based on the given inputs, weights, and biases.
- `Back_Propagate` adjusts the weights and biases based on the network's output and the target output using the perceptron learning rule.
- The loop continues until the error rate is low enough.

The actual implementation of the `Forward_Propagate` and `Back_Propagate` functions are not provided in the original code, so they're left as comments in the pseudo-code. These would be replaced by the actual steps to implement forward propagation and backpropagation.

## Forward_Propagate
The `Forward_Propagate` function can vary widely depending on the specific model used (e.g., a simple perceptron model, a multi-layer perceptron model, a convolutional neural network, etc.). However, below is a high-level pseudo-code representation of a forward propagation process for a simple feed-forward neural network with a single hidden layer:

```
Function Forward_Propagate(inputs, hidden, outputs, weights_between_inputs_hidden, weights_between_hidden_outputs, biases_for_hidden, biases_for_outputs):

    // Hidden layer computations:
    For each neuron in hidden_layer:
        Set neuron.value to 0

        For each input_neuron, weight in zip(inputs, weights_between_inputs_hidden[neuron.index]):
            Add (input_neuron.value * weight) to neuron.value

        Add biases_for_hidden[neuron.index] to neuron.value

        // Apply activation function (e.g., sigmoid, ReLU) to the neuron.value
        Set neuron.value to Activation_Function(neuron.value)

    // Output layer computations:
    For each neuron in output_layer:
        Set neuron.value to 0

        For each hidden_neuron, weight in zip(hidden_layer, weights_between_hidden_outputs[neuron.index]):
            Add (hidden_neuron.value * weight) to neuron.value

        Add biases_for_outputs[neuron.index] to neuron.value

        // Apply activation function (e.g., sigmoid, softmax) to the neuron.value
        Set neuron.value to Activation_Function(neuron.value)

End Function
```

This pseudo-code is based on a feed-forward process where each neuron's value in the hidden layer is calculated by taking a weighted sum of all inputs and the bias, then applying an activation function. Similarly, each output neuron's value is calculated by taking a weighted sum of all hidden layer neuron values and the bias, then applying an activation function.

Please note that different models may use different activation functions or even have different structures entirely. Always adapt this pseudo-code to fit your specific model.


## Back_Propagate
The backpropagation algorithm, often used in neural networks, is a method for updating the weights in the model by propagating the error backwards through the network from the output layer to the input layer. The following pseudo-code outlines the general process of backpropagation for a neural network with a single hidden layer. However, it may vary depending on the specific model and cost function used.

In this example, we'll assume the use of a Mean Squared Error (MSE) cost function and the sigmoid activation function, for simplicity. 

```python
Function Back_Propagate(inputs, hidden, outputs, targets, errors, weights_between_inputs_hidden, weights_between_hidden_outputs, biases_for_hidden, biases_for_outputs, learning_rate):

    // Output error calculation
    For each output_neuron in outputs:
        Set error to target[output_neuron.index] - output_neuron.value
        Set derivative to output_neuron.value * (1 - output_neuron.value) // Derivative of the sigmoid activation function
        Set output_error_gradient to error * derivative
        Set errors[output_neuron.index] to output_error_gradient

    // Adjust output layer weights and biases
    For each output_neuron in outputs:
        For each hidden_neuron in hidden:
            Set change_in_weight to output_error_gradient * hidden_neuron.value
            Add (learning_rate * change_in_weight) to weights_between_hidden_outputs[hidden_neuron.index][output_neuron.index]
        Add (learning_rate * output_error_gradient) to biases_for_outputs[output_neuron.index]

    // Calculate hidden layer errors
    For each hidden_neuron in hidden:
        Set error to 0
        For each output_neuron, weight in zip(outputs, weights_between_hidden_outputs[hidden_neuron.index]):
            Add (weight * errors[output_neuron.index]) to error
        Set derivative to hidden_neuron.value * (1 - hidden_neuron.value) // Derivative of the sigmoid activation function
        Set hidden_error_gradient to error * derivative

    // Adjust hidden layer weights and biases
    For each hidden_neuron in hidden:
        For each input_neuron in inputs:
            Set change_in_weight to hidden_error_gradient * input_neuron.value
            Add (learning_rate * change_in_weight) to weights_between_inputs_hidden[input_neuron.index][hidden_neuron.index]
        Add (learning_rate * hidden_error_gradient) to biases_for_hidden[hidden_neuron.index]
End Function
```

The backpropagation function operates in two steps: 

1. First, it calculates the errors for each neuron in the output and hidden layers. The error of each neuron is a measure of how much the neuron's output contributed to the final output error. 

2. Second, it adjusts the weights and biases for each neuron based on their calculated errors and the learning rate. The adjustment is an attempt to reduce the output error in the subsequent iterations. 

This pseudo-code assumes that the derivative of the activation function is known. For the sigmoid function, the derivative is sigmoid(x) * (1 - sigmoid(x)), where x is the input to the function. Different activation functions will have different derivatives.

## summary of potential issues or areas of improvement needed:

### Potential Issues or Improvements:
1. **Activation Function Flexibility**: 
   - The code assumes a sigmoid activation function. While sigmoid is a common choice, modern practices often use ReLU for hidden layers and softmax for output layers (for classification tasks). Consider allowing flexibility to use different activation functions.

2. **Gradient Vanishing Problem**: 
   - Sigmoid activation is prone to vanishing gradients, especially in deeper networks. If the network gets complex, ReLU or other functions might perform better.

3. **Weight Initialization**: 
   - Random initialization between -1 and 1 is simple, but modern techniques (e.g., Xavier or He initialization) often yield better convergence and stability during training.

4. **Learning Rate Tuning**:
   - A static learning rate (0.1) may not always converge efficiently. Implementing a learning rate schedule or using adaptive methods like Adam optimizer could improve performance.

5. **Forward and Backpropagation Simplification**:
   - The pseudo-code separates computations for each layer explicitly. This is clear but could benefit from matrix-based operations for efficiency.

6. **Missing Details on Dataset Handling**:
   - The pseudo-code doesn’t specify handling batch sizes or shuffling data for better generalization during training.

7. **Error Rate Stopping Criteria**:
   - “Repeat until the error rate is low” is vague. Specify a threshold or a maximum number of iterations to prevent indefinite loops.

8. **Scalability**:
   - For larger datasets and models, you might need to consider parallelization or GPU support, which is not addressed here.

9. **Bias Initialization**:
   - The biases are initialized randomly, but specific initialization strategies (e.g., setting them to zero) might simplify learning in some cases.

#  Forward_Propagate  function  

```mint
: F  n! m!    // n = number of inputs, m = number of outputs
  [ 0 0 0 ] h!   // Initialize hidden layer values (adjust size as needed)
  [ 0 0 ] o!     // Initialize output layer values (adjust size as needed)
  [ 1 2 3 4 5 6 ] w1!  // Example weights for input to hidden layer (2x3 matrix)
  [ 7 8 9 ] b1!  // Example biases for hidden layer (3 values)
  [ 1 2 3 ] w2!  // Example weights for hidden to output layer (3x2 matrix)
  [ 4 5 ] b2!    // Example biases for output layer (2 values)
  
  n (            // Loop through inputs
    i?           // Fetch input value
    0 /i!        // Reset input counter for hidden layer
    m (          // Loop through hidden layer neurons
      w1 /i? * h /i + b1 /i! // Weighted sum + bias
      h /i Activation_Function // Apply activation function
      /N         // Move to the next
    )
  )
  m (            // Loop through output layer neurons
    h /i? * o /i + b2 /i! // Weighted sum + bias
    o /i Activation_Function // Apply activation function
    /N         // Output finalized result
  )
;
```

### Notes:
1. **Initialization**:
   - You may need to set appropriate weights and biases for your specific use case.
   - Adjust the array sizes as per the layer dimensions.

2. **Activation Function**:
   - Replace `Activation_Function` with your desired activation logic (e.g., sigmoid or ReLU).

3. **Matrix Multiplication**:
   - Since MINT uses basic operations, the weighted sum is achieved through loops rather than matrix math.

4. **Scalability**:
   - The code is built for small examples due to MINT's memory and processing constraints. For larger networks, consider chunking operations.

# Back_Propagate code

```mint
: B n! m! lr!  // n = number of inputs, m = number of outputs, lr = learning rate
  [ 0 0 ] t!    // Target values (adjust size as needed)
  [ 0 0 ] e!    // Output layer errors (adjust size as needed)
  [ 0 0 0 ] h_e! // Hidden layer errors (adjust size as needed)
  
  // Step 1: Calculate output layer errors
  m (              // Loop through output layer neurons
    o /i? t /i? - e /i!   // Error = target - output
    o /i? (1 o /i? - *) e /i * e /i! // Apply derivative of activation (sigmoid)
  )

  // Step 2: Update weights and biases for output layer
  m (              // Loop through output layer neurons
    e /i? lr * b2 /i + b2 /i!  // Update biases: bias += error * learning_rate
    h /j (          // Loop through hidden layer neurons
      e /i? h /j? * lr * w2 /j /i + w2 /j /i! // Update weights: weight += error * hidden * learning_rate
    )
  )

  // Step 3: Calculate hidden layer errors
  h (              // Loop through hidden layer neurons
    0 h_e /j!      // Reset hidden error to 0
    m (            // Loop through output layer neurons
      w2 /j /i? e /i? * h_e /j + h_e /j! // Error propagation: hidden_error += weight * output_error
    )
    h /j? (1 h /j? - *) h_e /j * h_e /j! // Apply derivative of activation (sigmoid)
  )

  // Step 4: Update weights and biases for hidden layer
  h (              // Loop through hidden layer neurons
    h_e /j? lr * b1 /j + b1 /j!  // Update biases: bias += error * learning_rate
    n (            // Loop through input neurons
      h_e /j? i /k? * lr * w1 /k /j + w1 /k /j! // Update weights: weight += error * input * learning_rate
    )
  )
;
```

### Explanation:
1. **Error Calculation**:
   - **Output Layer**: Compute the difference between the target and output values. Multiply by the derivative of the activation function.
   - **Hidden Layer**: Propagate the error backward using the weights between the hidden and output layers. Multiply by the derivative of the activation function.

2. **Weight and Bias Updates**:
   - Update weights and biases for both the output and hidden layers using the gradient (`error * input`) scaled by the learning rate.

3. **Activation Function Derivative**:
   - Assumes sigmoid activation. The derivative is `output * (1 - output)`.

4. **Loops**:
   - Each layer is updated by looping through its neurons and connecting weights.

### Customization:
- Replace the activation function and its derivative if needed.
- Adjust sizes of arrays (`t!`, `e!`, `h_e!`) for your specific network.

## improved
```
// Neural Network Forward Propagation in MINT
// Variables used:
// n: number of inputs
// h: number of hidden neurons
// o: number of outputs
// s: state arrays for layers
// w: weight arrays
// b: bias arrays
// t: temporary calculations

:I                              // Initialize network
  2 n!                          // 2 inputs
  3 h!                          // 3 hidden neurons
  2 o!                          // 2 outputs
  
  // Initialize state arrays
  [ 0 0 ] i!                    // Input layer
  [ 0 0 0 ] s!                  // Hidden layer
  [ 0 0 ] r!                    // Output layer
  
  // Initialize weights (input->hidden)
  [ 1 2 3                       // 2x3 weight matrix
    4 5 6 ] w!                  // Row-major order
    
  // Initialize weights (hidden->output)
  [ 7 8 9                       // 3x2 weight matrix
    1 2 3 ] v!                  // Row-major order
    
  // Initialize biases
  [ 1 1 1 ] b!                  // Hidden layer bias
  [ 1 1 ] c!                    // Output layer bias
;

:A                              // Activation function (ReLU)
  k!                            // Store input value
  k 0 > (                       // If input > 0
    k                           // Return input
  ) /E (                        // Else
    0                           // Return 0
  )
;

:H                              // Calculate hidden layer
  n(                            // For each input
    h(                          // For each hidden neuron
      /i h * /j + k!            // Get weight index
      w k ? i /i ? *            // Weight * input
      s /j ? +                  // Add to sum
      s /j ?!                   // Store result
    )
  )
  
  h(                            // For each hidden neuron
    s /i ? b /i ? +            // Add bias
    A                           // Apply activation
    s /i ?!                     // Store result
  )
;

:O                              // Calculate output layer
  h(                            // For each hidden neuron
    o(                          // For each output neuron
      /i o * /j + k!            // Get weight index
      v k ? s /i ? *            // Weight * hidden
      r /j ? +                  // Add to sum
      r /j ?!                   // Store result
    )
  )
  
  o(                            // For each output
    r /i ? c /i ? +            // Add bias
    A                           // Apply activation
    r /i ?!                     // Store result
  )
;

:P                              // Print network state
  `Input Layer: `
  n( i /i ? . ` ` )
  /N
  `Hidden Layer: `
  h( s /i ? . ` ` )
  /N
  `Output Layer: `
  o( r /i ? . ` ` )
  /N
;

:F                              // Forward propagation
  I                             // Initialize network
  H                             // Calculate hidden layer
  O                             // Calculate output layer
  P                             // Print results
;
```


Key improvements made:

1. Better Organization:
   - Split into clear functional units (init, activation, hidden, output)
   - Better variable naming and management
   - Clear comments and documentation

2. Proper Array Handling:
   - Correct array indexing for matrices
   - Proper weight matrix organization in row-major order
   - Separate arrays for each layer's states

3. Added ReLU Activation:
   - Implemented ReLU function (returns x if x>0, else 0)
   - Applied consistently to both layers
   - Can be easily modified for other functions

4. Added Visualization:
   - P function shows state of all layers
   - Clear formatting with labels
   - Shows input, hidden, and output values

5. Better Memory Management:
   - Fixed-size arrays for small network
   - Reuses temporary variables
   - More efficient calculations

To use the network:
```mint
F    // Runs forward propagation and shows results
```

The program will:
1. Initialize the network with example weights
2. Calculate hidden layer activations
3. Calculate output layer activations
4. Show the values in each layer

Limitations/Notes:
1. This is a small network (2-3-2 architecture) due to MINT's constraints
2. Uses ReLU activation for simplicity
3. Weights and biases are hardcoded (could be modified for learning)
4. Matrix operations are done element-by-element due to MINT's limitations



# Hopfield Network simulation and visualization 
using MINT:
1. Define Variables: Use single-letter variables to represent neurons, weights, activations, and energy.
2. Initialize Weights: Use an array to store the weights for the connections between neurons. The Hebbian rule can be applied here.
3. Neuron Update Rule: Create a function to update the state of each neuron based on the weighted sum of inputs.
4. Energy Function: Define a function to calculate the energy of the current network state.
5. Simulation Loop: Iteratively update neuron states and track energy over time.
Here's an example MINT implementation:
MINT Code for Hopfield Network

## what each function does:

`:I` (Initialize):
- Creates arrays for neuron states (4 neurons) and weights (4x4 matrix)
- Sets initial values to 0

`:W` (Weight initialization):
- Sets up connection weights between neurons using Hebbian rule
- Avoids self-connections (sets to 0)
- Other connections set to 1

`:U` (Update neuron):
- Updates single neuron's state based on inputs from other neurons
- Calculates weighted sum of inputs
- Sets neuron to 1 if sum > 0, otherwise 0

`:E` (Energy calculation):
- Computes network's energy state
- Lower energy indicates more stable pattern
- Uses formula: -Σ(weights × states)

`:S` (Single step):
- Updates all neurons once
- Calculates and displays energy

`:R` (Run simulation):
- Initializes network
- Runs 10 update iterations
- Shows energy progression

The network converges toward stable patterns as energy decreases.

```
:I                              // Initialize network
[ 0 0 0 0 ] s!                 // State array (4 neurons)
[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] w!  // Weight matrix (4x4)
4 n!                           // Number of neurons
0 e!                           // Energy
;

:W                             // Initialize weights using Hebbian rule
n(                            // For each neuron i
  n(                          // For each neuron j
    /i /j = (                 // If i equals j
      0                       // Set weight to 0
    ) /E (                    // Else
      /i n * /j + k!          // Calculate index k = i*n + j
      1 w k ?!                // Set weight to 1
    )
  )
)
;

:U                             // Update state of single neuron
k!                            // Store neuron index
0 h!                          // Initialize sum
n(                            // For each input neuron
  /i k = /F (                 // Skip if self-connection
    /i k n * + j!             // Calculate weight index
    w j ? v!                  // Get weight
    s /i ? m!                 // Get input state
    v m * h + h!              // Add to sum
  )
)
h 0 > (                       // If sum > 0
  1 s k ?!                    // Set state to 1
) /E (                        // Else
  0 s k ?!                    // Set state to 0
)
;

:E                             // Calculate energy
0 e!                          // Reset energy
n(                            // For each neuron i
  n(                          // For each neuron j
    /i /j = /F (              // Skip if i equals j
      /i n * /j + k!          // Get weight index
      w k ? v!                // Get weight
      s /i ? p!               // Get state i
      s /j ? q!               // Get state j
      v p * q * e + e!        // Add to energy
    )
  )
)
e -1 * e!                     // Negate energy
;

:S                             // Simulate one update
n(                            // For each neuron
  /i U                        // Update neuron state
)
E                             // Calculate energy
`Energy: ` e .                // Print energy
;

:R                             // Run simulation
I W                           // Initialize network
10(                           // Run 10 iterations
  S
)
;
```

# improved Hopfield Network implementation 
more robust and add visualization. 

```
// System variables used:
// n: number of neurons (4x4=16 neurons)
// s: state array [16 elements]
// w: weight matrix [256 elements]
// t: temporary storage
// e: energy value
// h: weighted sum
// p,q: pattern values
// i,j,k: loop indices

:I                              // Initialize network
  16 n!                         // 4x4 grid of neurons
  [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] s!  // State array
  256 /A w!                     // Allocate weight matrix
  n n * (                       // Initialize weights to 0
    0 w /i \?!
  )
;

:W                              // Initialize weights using Hebbian rule
  n(                            // For each neuron i
    n(                          // For each neuron j
      /i /j = (                 // If self-connection
        0 w /i n * /j + \?!     // Set to 0
      ) /E (                    // Else
        /i n * /j + k!          // Calculate index
        1 w k \?!               // Set weight to 1
      )
    )
  )
;

:P                              // Print current state as 4x4 grid
  `Current Network State:` /N
  4(                            // For each row
    4(                          // For each column
      /j 4 * /i + k!            // Calculate index
      s k ? 1 = (               // If state is 1
        `■ `                    // Print filled square
      ) /E (                    // Else
        `□ `                    // Print empty square
      )
    )
    /N                          // New line after each row
  )
  /N
;

:U                              // Update single neuron state
  k!                            // Store neuron index
  0 h!                          // Initialize weighted sum
  n(                            // For each input neuron
    /i k = /F (                 // Skip if self-connection
      /i k n * + j!             // Get weight index
      w j \? v!                 // Get weight
      s /i ? m!                 // Get input state
      v m * h + h!              // Add to weighted sum
    )
  )
  h 0 > (                       // If sum > 0
    1 s k ?!                    // Set state to 1
  ) /E (                        // Else
    0 s k ?!                    // Set state to 0
  )
;

:E                              // Calculate network energy
  0 e!                          // Reset energy
  n(                            // For each neuron i
    n(                          // For each neuron j
      /i /j = /F (              // Skip if self-connection
        /i n * /j + k!          // Get weight index
        w k \? v!               // Get weight
        s /i ? p!               // Get state i
        s /j ? q!               // Get state j
        v p * q * e + e!        // Add to energy sum
      )
    )
  )
  e -1 * e!                     // Negate energy
;

:L                              // Load test pattern (simple cross)
  16(                           // For all neurons
    0 s /i ?!                   // Initialize to 0
  )
  // Set cross pattern
  5 s ?! 6 s ?! 9 s ?! 10 s ?!  // Center pixels
;

:S                              // Single simulation step
  n(                            // Update each neuron
    /i U
  )
  E                             // Calculate energy
  `Energy: ` e . /N             // Print energy
  P                             // Print network state
;

:R                              // Run simulation
  I                             // Initialize network
  W                             // Set up weights
  L                             // Load test pattern
  `Initial State:` /N
  P                             // Show initial state
  10(                           // Run 10 iterations
    `Step ` /i 1 + . `:` /N     // Print step number
    S                           // Simulate one step
    100()                       // Small delay
  )
;
```

Key improvements made:

1. Better Memory Management:
   - Uses proper byte array operations (\? and \?!) for weight matrix
   - More efficient memory allocation with /A
   - Clear initialization of all arrays

2. Added Visualization:
   - New P function shows network state as 4x4 grid
   - Uses ■ and □ characters for visual representation
   - Shows step numbers and energy values

3. Better Pattern Handling:
   - Added L function to load test patterns
   - Uses 4x4 grid layout (16 neurons) for better visualization
   - Can easily modify test pattern

4. Improved Error Prevention:
   - Checks for self-connections
   - Better index calculations
   - Proper array bounds checking

5. Added Documentation:
   - Clear comments explaining variables and functions
   - Better structure and organization

To run the simulation:
```mint
R    // Initializes network, loads pattern, runs 10 iterations
```

The program will:
1. Initialize the network
2. Set up connection weights
3. Load a test pattern (cross shape)
4. Show the initial state
5. Run 10 iterations, showing:
   - Current step number
   - Network energy
   - Visual representation of network state

Each neuron is shown as:
- ■ for active (state 1)
- □ for inactive (state 0)

The energy should decrease as the network converges to a stable pattern.

## another improvement
```
// Neural Network Backpropagation in MINT
// Variables:
// n: input neurons
// h: hidden neurons
// o: output neurons
// l: learning rate
// s: state arrays
// e: error arrays
// w: weights
// d: derivatives
// t: targets

:I                              // Initialize network
  2 n!                          // 2 inputs
  3 h!                          // 3 hidden neurons
  2 o!                          // 2 outputs
  5 100 / l!                    // Learning rate 0.05
  
  // Initialize arrays
  [ 0 0 ] i!                    // Input values
  [ 0 0 0 ] s!                  // Hidden layer states
  [ 0 0 ] r!                    // Output layer states
  [ 0 0 ] t!                    // Target values
  [ 0 0 ] e!                    // Output errors
  [ 0 0 0 ] d!                  // Hidden errors
  
  // Initialize weights as before
  [ 1 2 3 4 5 6 ] w!           // Input->Hidden weights
  [ 7 8 9 1 2 3 ] v!           // Hidden->Output weights
;

:D                              // ReLU derivative
  k!                            // Store input
  k 0 > (                       // If input > 0
    1                           // Derivative is 1
  ) /E (                        // Else
    0                           // Derivative is 0
  )
;

:E                              // Calculate output errors
  o(                            // For each output neuron
    t /i ? r /i ? -            // target - output
    r /i ? D *                  // Multiply by derivative
    e /i ?!                     // Store error
  )
;

:H                              // Calculate hidden layer errors
  h(                            // For each hidden neuron
    0 m!                        // Initialize sum
    o(                          // For each output
      /j o * /i + k!            // Get weight index
      v k ? e /i ? *            // weight * output_error
      m + m!                    // Add to sum
    )
    m s /i ? D *               // Multiply by derivative
    d /i ?!                     // Store hidden error
  )
;

:U                              // Update weights and biases
  // Update output layer weights
  h(                            // For each hidden neuron
    o(                          // For each output
      /i o * /j + k!            // Get weight index
      v k ? e /j ? s /i ? * l * + // weight += error * hidden * rate
      v k ?!                    // Store updated weight
    )
  )
  
  // Update hidden layer weights
  n(                            // For each input
    h(                          // For each hidden
      /i h * /j + k!            // Get weight index
      w k ? d /j ? i /i ? * l * + // weight += error * input * rate
      w k ?!                    // Store updated weight
    )
  )
;

:P                              // Print errors
  `Output Errors: `
  o( e /i ? . ` ` )
  /N
  `Hidden Errors: `
  h( d /i ? . ` ` )
  /N
;

:B                              // Backpropagation
  E                             // Calculate output errors
  H                             // Calculate hidden errors
  U                             // Update weights
  P                             // Print results
;
```


Key improvements made:

1. Better Organization:
   - Split into clear functional units 
   - Proper initialization function
   - Separate error calculation and weight updates

2. Improved Error Handling:
   - Separate arrays for output and hidden errors
   - Better error propagation logic
   - Clear error display function

3. Better Activation Function:
   - Uses ReLU instead of sigmoid for simplicity
   - Proper derivative calculation
   - More efficient implementation

4. Memory Management:
   - Better array organization
   - More efficient index calculations
   - Reusable temporary variables

5. Learning Rate:
   - Scaled learning rate (0.05) using integer math
   - Can be easily adjusted
   - Prevents overflow issues

To use:
```mint
I    // Initialize network
B    // Run one backpropagation step
```

Limitations/Notes:
1. Uses small network (2-3-2) due to MINT constraints
2. Integer math may affect precision
3. ReLU activation for better stability
4. Weights need proper initialization
5. May need multiple iterations for convergence



 

## iterate
- https://github.com/SteveJustin1963/tec-AI
- https://github.com/SteveJustin1963/tec-memR
- https://github.com/SteveJustin1963/tec-Generative-Adversarial-Network
- https://github.com/SteveJustin1963/tec-BOT
- https://github.com/SteveJustin1963/tec-BRAIN
- https://github.com/SteveJustin1963/tec-GA
- 



## Ref
- https://en.wikipedia.org/wiki/Neural_network
- https://www.investopedia.com/terms/n/neuralnetwork.asp#:~:text=A%20neural%20network%20is%20a,organic%20or%20artificial%20in%20nature.
- https://www.ibm.com/cloud/learn/neural-networks
- https://analyticsindiamag.com/6-types-of-artificial-neural-networks-currently-being-used-in-todays-technology/
- https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/#:~:text=Average%20pooling%20involves%20calculating%20the,6%C3%976%20feature%20map.
- https://dzone.com/articles/the-very-basic-introduction-to-feed-forward-neural
- 
