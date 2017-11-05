# mnist_torch

This is a torch project using a simple neural network on the MNIST dataset. Currently, I'm at 94-95% accuracy classifying the hand-written digits. I have implemented mini-batch updates (mini-batch size in code is 10) and run on a Google Cloud Platform Ubuntu 14.04 Nvidia Tesla K80 instance. Version 1 is a straightforward SGD; Version 2 utilizes the optim.adagrad optimization algorithm. 

The process is structured so that the input tensor is shuffled prior to each epoch - the data in the ith batch is different for each epoch. After all batch updates for an epoch are completed, the accuracy is computed against the validation set. Once the "wait" criteria or max number of epochs is reached, the function terminates and returns accuracy against the test set.

I also generate ROC curves, using gnuplot, and the area under the curve for each class in the first version.

Next up is a convolutional neural net for MNIST!
