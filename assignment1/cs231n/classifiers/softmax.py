import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    
	
    loss_i = -correct_class_score + np.log(np.sum(np.exp(scores)))  
    loss += loss_i
    
    for j in range(num_classes):
      score_loss = np.exp(scores[j])/np.sum(np.exp(scores))
      if j == y[i]:
        dW[:,y[i]] += (-1+score_loss)*X[i]
      else:
        dW[:,j] += score_loss*X[i]
  dW /= num_train
  dW += 2 * reg * W

  loss /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = X.dot(W)
  scores -= np.matrix(np.max(scores, axis=1)).T

  correct_class_score = scores[np.arange(scores.shape[0]), y]

  loss_i = -correct_class_score + np.log(np.sum(np.exp(scores), axis=1))
  loss = np.sum(loss_i)
  loss /= num_train
  loss += reg * np.sum(W * W)

  score_loss = np.exp(scores)/np.matrix(np.sum(np.exp(scores), axis=1)).T
  score_loss[np.arange(num_train), y] += -1

  dW = np.dot(X.T, score_loss)
  dW /= num_train
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

