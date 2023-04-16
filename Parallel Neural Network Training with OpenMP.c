/***********************************
This code implements a neural network training process using backpropagation algorithm. 
The neural network consists of one hidden layer with 128 nodes and an output layer with 10 nodes. 
The input layer has 784 nodes. The code initializes the network's weights and biases 
randomly using the rand_r() function. 
The code uses the rectified linear unit (ReLU) activation function for the hidden layer 
and the sigmoid function for the output layer. 
The training process uses stochastic gradient descent with a fixed learning rate of 0.3. 
The code uses OpenMP parallelization to speed up the training process. 
The maximum number of training iterations is set to 10000 and the minimum error is set to 0.001 as stopping criteria.


Neural networks are widely used in machine learning for various tasks such as image classification, 
speech recognition, and natural language processing. The training of neural networks is a 
computationally intensive process, especially for large datasets and complex architectures. 
The use of parallel processing techniques such as OpenMP can significantly speed up the training process 
and make it feasible to train neural networks on large datasets. 
This code provides a practical implementation of parallel neural network training with OpenMP 
that can be used as a basis for building more advanced neural network models for real-world applications.
sj
***********************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

typedef struct {
    double *w1;
    double *b1;
    double *w2;
    double *b2;
} NeuralNetwork;

NeuralNetwork *init_network() {
    NeuralNetwork *nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    nn->w1 = (double*) malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    nn->b1 = (double*) calloc(HIDDEN_SIZE, sizeof(double));
    nn->w2 = (double*) malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    nn->b2 = (double*) calloc(OUTPUT_SIZE, sizeof(double));
    int i, j;
    #pragma omp parallel for
    for (i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->w1[i] = ((double)rand_r(&i) / RAND_MAX * 2 - 1);
    }
    for (i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        nn->w2[i] = ((double)rand_r(&i) / RAND_MAX * 2 - 1);
    }
    return nn;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

void forward(NeuralNetwork *nn, double *input, double *output) {
    int i, j;
    double *a1 = (double*) malloc(HIDDEN_SIZE * sizeof(double));
    #pragma omp parallel for
    for (i = 0; i < HIDDEN_SIZE; i++) {
        a1[i] = nn->b1[i];
        for (j = 0; j < INPUT_SIZE; j++) {
            a1[i] += nn->w1[j * HIDDEN_SIZE + i] * input[j];
        }
        a1[i] = relu(a1[i]);
    }
    #pragma omp parallel for
    for (i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = nn->b2[i];
        for (j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += nn->w2[j * OUTPUT_SIZE + i] * a1[j];
        }
        output[i] = 1 / (1 + exp(-output[i]));
    }
    free(a1);
}

void train(NeuralNetwork *nn, double *input, double *target, double learning_rate, double *error) {
    int i, j;
    double output[OUTPUT_SIZE];
    double *a1 = (double*) malloc(HIDDEN_SIZE * sizeof(double));
    double delta2[OUTPUT_SIZE];
    double delta1[HIDDEN_SIZE];
    *error = 0;
    forward(nn, input, output);
    #pragma omp parallel for reduction(+:*error)
    for (i = 0; i < OUTPUT_SIZE; i++) {
        *error += 0.5 * pow((target[i] - output[i]), 2);
        delta2[i] = output[i] - target[i];
    }
    #pragma omp parallel for
    for (i = 0; i < HIDDEN_SIZE; i++) {
        a1[i] = nn->b1[i];
        for (j = 0; j < INPUT_SIZE; j++) {
            a1[i] += nn->w1[j * HIDDEN_SIZE + i] * input[j];
}
a1[i] = relu(a1[i]);
}
#pragma omp parallel for
for (i = 0; i < HIDDEN_SIZE; i++) {
delta1[i] = 0;
for (j = 0; j < OUTPUT_SIZE; j++) {
delta1[i] += nn->w2[i * OUTPUT_SIZE + j] * delta2[j];
}
delta1[i] *= a1[i] > 0 ? 1 : 0;
}
#pragma omp parallel for collapse(2)
for (i = 0; i < INPUT_SIZE; i++) {
for (j = 0; j < HIDDEN_SIZE; j++) {
nn->w1[i * HIDDEN_SIZE + j] -= learning_rate * input[i] * delta1[j];
}
}
#pragma omp parallel for
for (i = 0; i < HIDDEN_SIZE; i++) {
nn->b1[i] -= learning_rate * delta1[i];
}
#pragma omp parallel for collapse(2)
for (i = 0; i < HIDDEN_SIZE; i++) {
for (j = 0; j < OUTPUT_SIZE; j++) {
nn->w2[i * OUTPUT_SIZE + j] -= learning_rate * a1[i] * delta2[j];
}
}
#pragma omp parallel for
for (i = 0; i < OUTPUT_SIZE; i++) {
nn->b2[i] -= learning_rate * delta2[i];
}
free(a1);
}

int main() {
int i, iteration;
double input[INPUT_SIZE];
double target[OUTPUT_SIZE];
double min_error = 0.001;
int max_iterations = 10000;
double error;
NeuralNetwork *nn = init_network();
for (i = 0; i < INPUT_SIZE; i++) {
input[i] = 0;
}
target[0] = 1;
iteration = 0;
while (1) {
train(nn, input, target, 0.3, &error);
if (iteration >= max_iterations || error < min_error) {
break;
}
iteration++;
}
free(nn->w1);
free(nn->b1);
free(nn->w2);
free(nn->b2);
free(nn);
return 0;
}
