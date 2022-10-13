import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def unpack_part(fmt, data):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, data[:size]), data[size:]

def read_idx_file(filename):
    with gzip.open(filename, mode='rb') as fileobj:
        data = fileobj.read()

        (zero1, zero2, type_id, dims), data = unpack_part('>bbbb', data)
        if zero1 != 0 or zero2 != 0:
            raise Exception("Invalid file format")

        types = {
            int('0x08', base=16): 'B',
            int('0x09', base=16): 'b',
            int('0x0B', base=16): 'h',
            int('0x0C', base=16): 'i',
            int('0x0D', base=16): 'f',
            int('0x0E', base=16): 'd'
        }
        type_code = types[type_id]

        dim_sizes, data = unpack_part('>' + ('i' * dims), data)
        num_examples = dim_sizes[0]
        input_dim = int(np.prod(dim_sizes[1:]))

        X, data = unpack_part('>' + (type_code * (num_examples * input_dim)), data)
        if data:
            raise Exception("invalid file format")

        new_shape = (num_examples, input_dim) if input_dim > 1 else num_examples
        return np.array(X).reshape(new_shape, order='C')

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    images = read_idx_file(image_filename).astype('float32')
    images = (images - images.min()) / (images.max() - images.min())
    labels = read_idx_file(label_filename).astype('uint8')
    return images, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """

    ## Reference code from HW0:
    # return (np.log(np.exp(Z).sum(axis=1)) - Z[np.arange(Z.shape[0]), y]).mean()
    ## Or:
    # z2 = np.log(np.exp(Z).sum(axis=1))
    # y2 = (Z * one_hot(y)).sum(axis=1)
    # return (z2 - y2).mean()

    ### BEGIN YOUR SOLUTION
    z2 = ndl.log(ndl.summation(ndl.exp(Z), axes=(1)))
    y2 = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1))
    res = ndl.sub(z2, y2)
    res = ndl.divide_scalar(ndl.summation(res), res.shape[0])
    return res
    ### END YOUR SOLUTION

def one_hot(indexes, dims=None):
    dims = dims or indexes.max()+1
    return np.eye(dims)[indexes]

def normalize_rows(values):
    return ndl.divide(values, ndl.broadcast_to(ndl.summation(values, axes=(1,)), values.shape))

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ## Reference code from HW0:
    # def normalize_rows(values):
    #     return values / values.sum(axis=1)[:, np.newaxis]
    # num_classes = y.max() + 1
    # for i in range(0, X.shape[0], batch):
    #     # print(f"processing minibatch [{i}, {i+batch})...")
    #     minibatch_X = X[i:i+batch]
    #     minibatch_y = y[i:i+batch]
    #     Z1 = np.maximum(0, np.matmul(minibatch_X,W1))
    #     G2 = normalize_rows(np.exp(np.matmul(Z1,W2))) - one_hot(minibatch_y, dims=num_classes)
    #     G1 = np.matmul(G2,W2.T) * (Z1 > 0).astype('int')
    #     W1_grad = np.matmul(minibatch_X.T, G1)
    #     W2_grad = np.matmul(Z1.T, G2)
    #     W1 -= (lr / batch) * W1_grad
    #     W2 -= (lr / batch) * W2_grad

    ### BEGIN YOUR SOLUTION
    num_classes = y.max() + 1
    for i in range(0, X.shape[0], batch):
        # print(f"DEBUG: processing minibatch [{i}, {i+batch})...")
        minibatch_X = ndl.Tensor(X[i:i+batch])
        minibatch_y = ndl.Tensor(one_hot(y[i:i+batch], dims=num_classes))
        Z = ndl.relu(minibatch_X @ W1) @ W2
        loss = softmax_loss(Z, minibatch_y)
        # print("DEBUG: calculating gradients...")
        loss.backward()
        # print("DEBUG: calculating gradients: done")

        ## Since softmax_loss is already returning the average loss over a
        # batch of size m, we don't need to divide the gradients to batch:
        # W1 = ndl.sub(W1, ndl.mul_scalar(W1.grad, (lr / batch))).detach()
        # W2 = ndl.sub(W2, ndl.mul_scalar(W2.grad, (lr / batch))).detach()
        W1 = ndl.sub(W1, ndl.mul_scalar(W1.grad, lr)).detach()
        W2 = ndl.sub(W2, ndl.mul_scalar(W2.grad, lr)).detach()

    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
