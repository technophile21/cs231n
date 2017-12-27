import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  loss_i = np.zeros(num_train)
  for i in range(num_train):
    scores = X[i].dot(W)
    #print("Score: ", scores)
    scores -= np.max(scores)  #numerical stability
    #print("Stabilized score: ", scores)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    loss_i[i] = -np.log(exp_scores[y[i]] / sum_exp_scores)
    for j in range(num_class):
      softmax_score = exp_scores[j] / sum_exp_scores
      if(j == y[i]):
        dW[:, j] += (softmax_score - 1) * X[i]
      else:
        dW[:, j] += softmax_score * X[i]
  
  #print("Loss: ", loss_i)
  #print("Loss sum: ", np.sum(loss_i))
  loss = np.sum(loss_i) / num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  sample_indices = np.random.choice(X.shape[0], 20)
  #X = X[sample_indices, :]
  #y = y[sample_indices]

  scores = X.dot(W)
  scores = scores - np.reshape(np.max(scores, axis=1), (-1, 1))  #numerical stability
  #print("Stabilized  scores", scores)
  exp_scores = np.exp(scores)
  #print("Exp scores", exp_scores)
  prob_scores = exp_scores / np.reshape(np.sum(exp_scores, axis=1), (-1, 1))
  #print("Prob scores", prob_scores)
  class_scores = prob_scores[np.arange(prob_scores.shape[0]), y]
  #print("Class scores", class_scores)
  cross_entropy_loss = -np.log(class_scores)

  loss = np.sum(cross_entropy_loss)
  loss /= num_train
  loss += reg * np.sum(W * W)

  #gradient
  prob_scores[np.arange(prob_scores.shape[0]), y] -= 1
  dW = X.T.dot(prob_scores)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_for_scores(scores, y):
  """
  Compute softmax loss, gradient for given score matrix (no regularization)

  score => N x C where N is number of smaples and C is number of classes

  y => N, vector of correct labels

  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = scores.shape[0]

  scores = scores - np.reshape(np.max(scores, axis=1), (-1, 1))  #numerical stability
  #print("Stabilized  scores", scores)
  exp_scores = np.exp(scores)
  #print("Exp scores", exp_scores)
  prob_scores = exp_scores / np.reshape(np.sum(exp_scores, axis=1), (-1, 1))
  #print("Prob scores", prob_scores)
  class_scores = prob_scores[np.arange(prob_scores.shape[0]), y]
  #print("Class scores", class_scores)
  cross_entropy_loss = -np.log(class_scores)

  loss = np.sum(cross_entropy_loss)
  loss /= num_train

  prob_scores[np.arange(prob_scores.shape[0]), y] -= 1
  dW = prob_scores
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW