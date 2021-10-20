from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """



    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)



    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two-layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2'] #shapes 10,3 -- 3
        N, D = X.shape

        # Compute the forward pass
        scores = 0.
        
        #############################################################################
        # TODO: Perform the forward pass, computing the class probabilities for the #
        # input. Store the result in the scores variable, which should be an array  #
        # of shape (N, C).                                                          #
        #############################################################################
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        

        #relu function by hand
        #def ReLU(x):
            #for i in range(len(x)):
                #for j in range(len(x[0])):
                    #if x[i,j] < 0:
                        #x[i,j] = 0
            #return x

        # matrix version (simply taking element-wise maximum between a matrix of
        # same dim of W with all zeros and the W matrix)
        def ReLU(x):
            #print(x)
            return np.maximum(x, np.zeros((x.shape[0], x.shape[1])))
        
        #transpose the matrices to exploit moltiplication
        a1 = np.transpose(X)
        W1_t = np.transpose(W1)
        W2_t = np.transpose(W2)
        
        
        #first layer
        z2 = np.dot(W1_t,a1)
        
        # add biases
        #for i in range(len(z2)):
            #z2[i] += b1[i]

        # matrix version (row-wise sum)
        
        z2 = z2 + b1[:, None]
        
        #relu
        
        a2 = ReLU(z2)

        
        #second layer + bias + softmax
        z3 = np.dot(W2_t,a2)
        
        #for i in range(len(z3)):
            #z3[i] += b2[i]
            
        # matrix version (row-wise sum)
        z3 = z3 + b2[:, None]
        
        a3 = np.exp(z3) / np.sum(np.exp(z3), axis=0)   #softmax
        
        #transpose in the end
        scores = np.transpose(a3)
        
        
        
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # If the targets are not given then jump out, we're done
        if y is None:
            return scores


        # Compute the loss
        loss = 0.
        
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        
        # Implement the loss for the softmax output layer
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        
        i = np.arange(0,N,1)
        
        
        # matrix version...HERE WAS THE ERRORRRRRRRR
        loss = np.mean(-np.log(scores[i,y]), axis = 0)
        
        # regularizer
        W1_norm = (np.linalg.norm(W1))**2
        W2_norm = (np.linalg.norm(W2))**2
        
        
        # matrix version 
        loss = loss + reg * ( W1_norm + W2_norm )
        
        
        
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}

        ##############################################################################
        # TODO: Implement the backward pass, computing the derivatives of the weights#
        # and biases. Store the results in the grads dictionary. For example,        #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size  #
        ##############################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        
        
        delta = np.zeros((N,len(np.unique(y))))
        for i in range(len(scores)):
          delta[i][y[i]] = 1
          #print(i)


      
        grad_z3 =  1/N *(scores-delta)
        #grads['b2'] = grad_z3

        dz3_db2 =  np.ones((1,N))
        grads['b2'] = np.dot(dz3_db2, grad_z3) 
        
        grads['W2'] = np.dot( a2, grad_z3) +2*reg* W2 
           

        grads_a2 = np.dot(W2,np.transpose(grad_z3))
        
        step= np.sign(a2)

        step_t = np.transpose(step)
     
        grads_z2 = np.multiply(grads_a2, step)
        
        grads['b1'] = np.dot( np.ones((1,N)), np.transpose(grads_z2) )


        
        grads['W1'] = np.dot(a1, np.transpose(grads_z2))  +2*reg* W1 
        
        
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads



    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        
        num_train = X.shape[0]
        iterations_per_epoch = max( int(num_train // batch_size), 1)


        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = X
            y_batch = y

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
<<<<<<< HEAD
            random_idx = np.random.choice(num_train, size=batch_size)
            
            X_batch = X[random_idx, :]
            y_batch = y[random_idx]
=======
            # associating the right label to each X_i
            y = np.reshape(y, (5,1))
            data = np.concatenate((X, y), axis=1)
            
            # random shuffle and then take subsample
            np.random.shuffle(data)
            
            sub = data[:3,:]
            # re-splitting to get the sub sample of X and y
            X_batch = sub[:,:4]
            y_batch = np.reshape(sub[:,4:], (3,)).astype(int)
            
            pass
>>>>>>> c2af667c9884b1f7f285c214f45d1c89b94d535f
        
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            hidden_s = self.params['W1'].shape[1]
            output_s = self.params['W2'].shape[1]
            
            
            
            self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
            self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
<<<<<<< HEAD
            self.params['b1'] = np.reshape(self.params['b1'] - learning_rate * grads['b1'], (hidden_s,))
            self.params['b2'] = np.reshape(self.params['b2'] - learning_rate * grads['b2'], (output_s,))
=======
            self.params['b1'] = np.reshape(self.params['b1'] - learning_rate * grads['b1'], (10,))
            self.params['b2'] = np.reshape(self.params['b2'] - learning_rate * grads['b2'], (3,))
>>>>>>> c2af667c9884b1f7f285c214f45d1c89b94d535f
            



            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # At every epoch check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }



    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        def ReLU(x):
            #print(x)
            return np.maximum(x, np.zeros((x.shape[0], x.shape[1])))
        
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2'] #shapes 10,3 -- 3
        N, D = X.shape
        
        a1 = np.transpose(X)
        W1_t = np.transpose(W1)
        W2_t = np.transpose(W2)
        
        
        #first layer
        z2 = np.dot(W1_t,a1)
    

        # matrix version (row-wise sum)
        z2 = z2 + b1[:, None]
        
        #relu
        a2 = ReLU(z2)

        
        #second layer + bias + softmax
        z3 = np.dot(W2_t,a2)

        # matrix version bias (row-wise sum)
        z3 = z3 + b2[:, None]
        
        a3 = np.exp(z3) / np.sum(np.exp(z3), axis=0)   #softmax
        
        #transpose in the end
        scores = np.transpose(a3)
        
        
        y_pred = np.zeros(N)
        
        
        for i in range(N):
            c = np.argmax(scores[i])
            y_pred[i] = c
              
        #print(y_pred)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred


