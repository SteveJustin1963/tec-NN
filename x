 fortran 95

 

program neural_network

implicit none

integer, parameter :: n_inputs = 2, n_outputs = 1, n_hidden = 2
real, dimension(n_inputs, n_hidden) :: weights_ih
real, dimension(n_hidden, n_outputs) :: weights_ho
real, dimension(n_hidden) :: bias_h
real, dimension(n_outputs) :: bias_o
real, dimension(n_inputs) :: inputs
real, dimension(n_outputs) :: outputs
real, dimension(n_outputs) :: targets
real, dimension(n_outputs) :: errors
real, dimension(n_hidden) :: hidden

real, parameter :: lr = 0.1

integer :: i

! Initialize the weights and biases to random values

call random_seed()

do i = 1, n_inputs
    weights_ih(i, :) = (/ (rand() - 0.5) * 2, (rand() - 0.5) * 2 /)
end do

do i = 1, n_hidden
    weights_ho(i, :) = (/ (rand() - 0.5) * 2 /)
end do

bias_h = (/ (rand() - 0.5) * 2, (rand() - 0.5) * 2 /)
bias_o = (/ (rand() - 0.5) * 2 /)

! For each training example:

do i = 1, n_inputs

    ! Calculate the output of the network

    call forward_propagate(inputs(i), hidden, outputs, weights_ih, weights_ho, bias_h, bias_o)

    ! Adjust the weights and biases according to the perceptron learning rule

    call back_propagate(inputs(i), hidden, outputs, targets(i), errors, weights_ih, weights_ho, bias_h, bias_o, lr)

end do

end program neural_network
