import numpy as np 
import tensorflow as tf 

def CrossEntropyLoss(logits, labels):
    """
    Calculates the softmax cross entropy loss for classification
    predictions.
    """
    labels = tf.cast(labels, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels)
    return loss

def advreg_loss(loss, regloss="mean_gain", probs=None, 
                        _lambda=0, l2loss=0, per_example=False):
    """
    Args:
    -----
    regloss: "mean_gain" or "mse" (mean squared error)
    """
    if per_example:
         if _lambda > 0:
            regterm = probs - 0.5
            return tf.add(loss, tf.multiply(_lambda, regterm), l2loss)
         else:
            return loss
    else:
        loss = tf.reduce_mean(loss)
        if _lambda > 0:
            if regloss=="mean_gain":
                regterm = tf.reduce_mean(probs) - 0.5
            elif regloss=="mse":
                regterm = tf.reduce_mean(tf.square(tf.subtract(probs, 1.0)))
            return tf.add(loss, tf.multiply(_lambda, regterm), l2loss)
        else:
            return loss

def mse(true, predicted):
      """
      Computes loss of the attack model on given batch
      Args:
      ----
      """
      loss = tf.losses.mean_squared_error(true, predicted)
      return loss

def inference_mse(mem_probs, nonmem_probs):
  """
  """
  members = tf.losses.mean_squared_error(tf.ones(
                          mem_probs.shape), mem_probs)
  nonmembers = tf.losses.mean_squared_error(tf.zeros(
                          nonmem_probs.shape), nonmem_probs)
  empirical = tf.add(members, nonmembers)
  
  # Plotting the inference gain (uses a slightly different formula)
  plot = 0.5 * tf.add(tf.reduce_mean(tf.square(mem_probs)), 
                      tf.reduce_mean(tf.square(1 - nonmem_probs)))
  return plot, empirical