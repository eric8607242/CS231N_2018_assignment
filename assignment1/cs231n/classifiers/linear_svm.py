import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
	  
	  # compute each score except correst_class_scores to minus the correst_class_scores 
	  # and check sign to determine the loss value
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i,:]
        dW[:,j] += X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  delta = 1
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  yi_scores = scores[np.arange(scores.shape[0]), y] 
  margins = np.maximum(0, scores - np.matrix(yi_scores).T + delta)
  margins[np.arange(num_train), y] = 0
  loss = np.mean(np.sum(margins, axis = 1))

  loss += reg * np.sum(W*W) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # get the element that loss is bigger than 0, and set to 1 else set 0
  binary = margins
  binary[margins > 0] = 1
  # count the number of the scores bigger than the correct_class_score
  # and that the correct_class minus the number that we computed
  row_sum = np.sum(binary, axis = 1)
  binary[np.arange(num_train), y] = -row_sum.T
  # the dW is the dot of X.T with binary. The reason between the X.T is when we compute the gradient
  # we need to compute the num_train times and plus all of them to get the total gradient. 
  # For example, the first row of dW is all composed by the element of first col of X because 
  # of the gradient is composed by the first element between the different examples
  dW = np.dot(X.T, binary)
  
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return loss, dW
