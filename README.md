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

Please adapt this pseudo-code to your specific model and circumstances.

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
