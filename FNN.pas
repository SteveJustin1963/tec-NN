
 

program neural_network;
 
const
    n_inputs = 2;
    n_outputs = 1;
    n_hidden = 2;
    lr = 0.1;
 
var
    weights_ih : array[1..n_inputs, 1..n_hidden] of real;
    weights_ho : array[1..n_hidden, 1..n_outputs] of real;
    bias_h : array[1..n_hidden] of real;
    bias_o : array[1..n_outputs] of real;
    inputs : array[1..n_inputs] of real;
    outputs : array[1..n_outputs] of real;
    targets : array[1..n_outputs] of real;
    errors : array[1..n_outputs] of real;
    hidden : array[1..n_hidden] of real;
    i : integer;
 
begin
 
    { Initialize the weights and biases to random values }
 
    randomize;
 
\\ this line means to randomly generate a set of weights and biases and store them in the the array. 
\\ Random generates a random number between 0 and 1
\\ So by doing random - 0.5 it will be between -0.5 and 0.5
\\ Then by doing * 2 it will be between -1 and 1


    for i := 1 to n_inputs do
        weights_ih[i,:] := [(random - 0.5) * 2, (random - 0.5) * 2];
 
    for i := 1 to n_hidden do
        weights_ho[i,:] := [(random - 0.5) * 2];
 
\\ so first the variable bias_h is declared as a list of two random numbers between -1 and 1
then the variable bias_o is declared as a list of one random number between -1 and 1


    bias_h := [(random - 0.5) * 2, (random - 0.5) * 2];
    bias_o := [(random - 0.5) * 2];
 
    { For each training example... }
 \\

    for i := 1 to n_inputs do
    begin
 
 \\       { Calculate the output of the network }
 
        forward_propagate(inputs[i], hidden, outputs, weights_ih, weights_ho, bias_h, bias_o);
 
     \\   { Adjust the weights and biases according to the perceptron learning rule }
 
        back_propagate(inputs[i], hidden, outputs, targets[i], errors, weights_ih, weights_ho, bias_h, bias_o, lr);
 
    end;
 
end.




 
