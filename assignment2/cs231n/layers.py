from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    shaped_x = np.reshape(x, (x.shape[0], -1))

    out = shaped_x.dot(w)+b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_shape = x.shape
    x = np.reshape(x, (x.shape[0], -1))
    dx = dout.dot(w.T)
    dx = np.reshape(dx, x_shape)

    dw = x.T.dot(dout)
    # plus each row to one row
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # because relu is mask the input x to generate output dout ,so in backprop
    # we need to use the mask that x used to dout
    # why we can not use dout>0 here(x sould be the same with out)
    # because here dout is not forward from x but random.
    drelu = np.zeros(dout.shape)
    drelu[x > 0] = 1
    dx = dout * drelu
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_variance = np.var(x, axis=0)
        standard_d = np.sqrt(sample_variance+eps)
        

        normalization = (x-sample_mean)/standard_d
        out = normalization*gamma + beta

        running_mean = momentum * running_mean + (1-momentum) * sample_mean
        running_var = momentum * running_var + (1-momentum) * sample_variance

        cache = {
              'normalization':normalization,
              'standard_d':standard_d,
              'var':sample_variance,
              'eps':eps,
              'x_mean':x-sample_mean,
              'gamma':gamma
            }
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        running_standard_d = np.sqrt(running_var+eps)
        
        x = (x-running_mean)/running_standard_d
        out = x*gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    N = dout.shape[0]

    # (beta) out = x*gamma + beta
    dbeta = np.sum(dout, axis=0)
    dbeta_layer = dout
    
    # (gamma) out = normalization*gamma + beta
    dgamma = np.sum(cache['normalization']*dbeta_layer, axis=0)
    dgamma_x = dbeta_layer*cache['gamma']

    # (1/sta_d) normalization = (x-mean)/sta_d
    distan_d = np.sum(dgamma_x*cache['x_mean'], axis=0)
    dx_mean = dgamma_x/cache['standard_d']

    # (sta_d) 1/sta_d
    dstan_d = distan_d * (-1/(cache['standard_d']**2))
    # (var+eps) sta_d = sqrt(var+eps)
    dvar = 0.5*(1/np.sqrt(cache['var']+cache['eps']))*dstan_d
    
    # (((x-mean)^2)/n) var = sum((x-mean)^2)/n
    dvar_sum = (np.ones_like(dout)*dvar)/N
    dx_mean_pow = 2*cache['x_mean']*dvar_sum

    # (x-mean) used in two place
    # 1) variance -> (x-mean)^2
    # 2) numerator -> (x-mean)
    dmean = -1*(np.sum(dx_mean+dx_mean_pow, axis=0))
    dx = (dx_mean + dx_mean_pow)

    # (x) mean = sum(x)/n
    dx += np.ones_like(dout)*dmean/N 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x_mean = cache['x_mean']
    var = cache['var']
    gamma = cache['gamma']
    normalization = cache['normalization']
    standard_d = 1/cache['standard_d'] #1/sqrt(var+eps)
    N = dout.shape[0]

    dgamma = np.sum(dout*normalization, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx = (1/N)*gamma*(standard_d)*(N*dout-np.sum(dout, axis=0)-(x_mean)*(standard_d**2)*np.sum(dout*x_mean, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    x = x.T

    sample_mean = np.mean(x, axis=0)
    sample_variance = np.var(x, axis=0)
    standard_d = np.sqrt(sample_variance+eps)

    normalization = (x-sample_mean)/standard_d
    normalization = normalization.T
    out = normalization*gamma + beta

    cache = {
          'normalization':normalization,
          'standard_d':standard_d,
          'var':sample_variance,
          'eps':eps,
          'x_mean':x-sample_mean,
          'gamma':gamma
        }
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x_mean = cache['x_mean']
    var = cache['var']
    gamma = cache['gamma']
    normalization = cache['normalization']
    standard_d = 1/cache['standard_d']
    N = dout.shape[1] # because use dimension to scale and shift, use bn code but transpose

    dgamma = np.sum(dout*normalization, axis=0)
    dbeta = np.sum(dout, axis=0)

    dnormalization = dout * gamma
    dnormalization = dnormalization.T
    normalization = normalization.T

    dx = (1/N)*standard_d*(N*dnormalization-np.sum(dnormalization, axis=0)-(x_mean)*(standard_d**2)*np.sum(dnormalization*x_mean, axis=0))
    dx = dx.T

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        drop_mask = (np.random.rand(*x.shape) <= p)*1 # muse <= p
        inverted_mask = 1/(1-p)
        mask = drop_mask * inverted_mask
        
        out = x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask # ddrop/x = mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and

    Input:
    - x: Input data of shape (N, C, H, W) 
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']

    N, C, x_H, x_W = x.shape
    F, C, weight_H, weight_W = w.shape # F is filters nums
   
    out_H = int(1 + (x_H+2*pad-weight_H)/stride)
    out_W = int(1 + (x_W+2*pad-weight_W)/stride)

    out = np.zeros((N, F, out_H, out_W))

    pad_array = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for n in range(N):
        for f in range(F):
            for n_h in range(int(out_H)):
                for n_w in range(int(out_W)):
                    region = pad_array[n, :, n_h*stride:n_h*stride+weight_H, n_w*stride:n_w*stride+weight_W]
                    out[n][f][n_h][n_w] = np.sum(region*w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    pad = conv_param['pad']
    stride = conv_param['stride']

    N, C, x_H, x_W = x.shape
    F, C, weight_H, weight_W = w.shape

    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    N, F, out_H, out_W = dout.shape
    pad_array = np.pad(x, ((0 ,0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dpad_array = np.zeros(pad_array.shape)

    # db. Each filter of dout is contributed by different filter of bias.
    # So db of each filter is sumed up each filter of dout.
    # Ex. out = w*x+b => db = 1. In chain rule, dout * db = dout * 1(sum) = db/L
    for f in range(F):
        db[f] = np.sum(dout[:,f])

    # dw. Each filter of out is composed of each filter of weight. Every weight element
    # influenced the apperance of out.So we can sum up every value that composed by out 
    # element * a region of x which is as big as the size of weight to get the each 
    # filter of dw.
    # Ex. out = w*x+b => dw/out = x. In chain rule, dout * x = dw/L
    for n in range(N):
        for f in range(F):
            for n_h in range(out_H):
                for n_w in range(out_W):
                    region = pad_array[n, :, n_h*stride:n_h*stride+weight_H, n_w*stride:n_w*stride+weight_W]
                    dw[f] += region * dout[n, f, n_h, n_w]
                    
    # dx. X multiply each filter of weight to get the out. A small region of x with 
    # weight get a element of out.So we let a element of out to multiply by weight,
    # we can get a small region of dx which composed the element of out.But each 
    # element of x can compose many element of out, so we need to sum up the each 
    # small region to make the total dx.
    # Ex. out = w*x+b => dx/out = weight. In chain rule, dout * weight = dx/L
                    dpad_array[n, :, n_h*stride:n_h*stride+weight_H, n_w*stride:n_w*stride+weight_W] += w[f] * dout[n, f, n_h, n_w]
                    
    dx = dpad_array[:, :, pad:pad+x_H, pad:pad+x_W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape

    out_H = 1 + int((H-pool_h)/stride)
    out_W = 1 + int((W-pool_w)/stride)

    out = np.zeros([N, C, out_H, out_W])

    # pooling. we have to get the max value in each filter of pooling region. First,
    # get the pooling region which shape is (C, pool_h, pool_w), and get the max
    # value in each filter.So we use the np.max with axis (1,2) which is get the
    # max value in both poo_h and pool_w region but not cross filter.
    for n in range(N):
        for n_h in range(out_H):
            for n_w in range(out_W):
                region = x[n, :, n_h*stride:n_h*stride+pool_h, n_w*stride:n_w*stride+pool_w]
                out[n, :, n_h, n_w] =  np.amax(region, axis=(1, 2))
                

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    N, C, out_H, out_W = dout.shape

    dx = np.zeros(x.shape)

    # dpooling. Out is composed by the max value of the small size of x. The gradient
    # of max is let the max value exist and other are set to 0.Use argmax to get the
    # index of max value in the small region of x, and use unravel to change the value 
    # to coordinate axis.Set the same position in dx to the value of the dout
    # (chain rule => 1*dout).
    for n in range(N):
        for c in range(C):
            for n_h in range(out_H):
                for n_w in range(out_W):
                    region = x[n, c, n_h*stride:n_h*stride+pool_h, n_w*stride:n_w*stride+pool_w]
                    # we have different usage with the pooling forward because argmax can't use
                    # tuple to axis.So we have to set the value to each channel
                    max_index = np.argmax(region)
                    index = np.unravel_index(max_index, (pool_h, pool_w)) 

                    dx[n, c, n_h*stride+index[0], n_w*stride+index[1]] = dout[n, c, n_h, n_w]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # spatial batchnorm is calculate mean and variance for each channel in all N,W,H.
    # general batchnorm is (N, D) ,and calculated norm to (D)
    # spatial batchnorm is (N, C, H, W) ,and calculated norm to (C)
    N, C, H, W = x.shape
    
    x_trans = np.transpose(x, (0, 2, 3, 1))
    x_reshape = np.reshape(x_trans, (-1, C))

    out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)

    out = np.reshape(out, (N, H, W, C))
    out = np.transpose(out, (0, 3, 1, 2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape

    dout = np.transpose(dout, (0, 2, 3, 1))
    dout = np.reshape(dout, (-1, C))
    
    # the shape of dx is same as dout. So we have to transpose and reshape back
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = np.reshape(dx, (N, H, W, C))
    dx = np.transpose(dx, (0, 3, 1, 2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N, C, H, W = x.shape

    x = np.reshape(x, (N, G, C//G, H, W))

    sample_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    sample_variance = np.var(x, axis=(2, 3, 4), keepdims=True)
    standard_d = np.sqrt(sample_variance + eps)

    normalization = (x-sample_mean)/standard_d
    normalization = np.reshape(normalization, (N, C, H, W))
    
    out = normalization*gamma + beta

    cache = {
          'normalization':normalization,
          'standard_d':standard_d,
          'var':sample_variance,
          'eps':eps,
          'x_mean':x-sample_mean,
          'gamma':gamma,
          'G':G
        }
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x_mean = cache['x_mean']
    var = cache['var']
    gamma = cache['gamma']
    normalization = cache['normalization']
    standard_d = 1/cache['standard_d']
    G = cache['G']

    N, C, H, W = dout.shape
    group = C//G*H*W

    dgamma = np.sum(dout*normalization, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    dnormalization = dout * gamma
    dnormalization = np.reshape(dnormalization, (N, G, C//G, H, W))
    normalization = np.reshape(normalization, (N, G, C//G, H, W))

    dx = (1/group)*standard_d*(group*dnormalization-np.sum(dnormalization, axis=(2, 3, 4), keepdims=True)-(x_mean)*(standard_d**2)*np.sum(dnormalization*x_mean, axis=(2, 3, 4), keepdims=True))
    dx = np.reshape(dx, (N, C, H, W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    # margins is the scores diff between error score and correct score ,if correct score
    # is bigger than error score than margin is 0 
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    # it is just gradient of dx(loss) if u need dw, u should mulitply dx(loss) and x(input)
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    # if the loss is bigger than 0, the Wj is influence the loss, so we need to minus the margin
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    probs = np.exp(log_probs)
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
