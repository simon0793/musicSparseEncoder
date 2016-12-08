import numpy as np

def subtractMeanPerImage(data):
    return np.stack([data[:, idx] - np.mean(data[:, idx]) for idx in range(data.shape[1])], axis=1)

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))

def genLines(numInputs, numDatapoints, probabilityOn):
    """
    Generate random dataset of images containing lines. Each image has a mean value of 0.
    Inputs:
        numInputs [int] number of pixels for each image, must have integer sqrt()
        numDatapoints [int] number of images to generate
        probabilityOn [float] probability of a line (row or column of 1 pixels) appearing in the image,
            must be between 0.0 (all zeros) and 1.0 (all ones)
    Outputs:
        outImages [np.ndarray] batch of images, each of size
            (numInputs, numDatapoints)
    """
    if probabilityOn < 0.0 or probabilityOn > 1.0:
        assert False, "probabilityOn must be between 0.0 and 1.0"

    # Each image is a square, rasterized into a vector
    outImages = np.zeros((numInputs, numDatapoints))
    numEdgePixels = int(np.sqrt(numInputs))
    for batchIdx in range(numDatapoints):
        outImage = np.zeros((numEdgePixels, numEdgePixels))
        # Construct a set of random rows & columns that will have lines with probablityOn chance
        rowIdx = [0]; colIdx = [0];
        #while not np.any(rowIdx) and not np.any(colIdx): # uncomment to remove blank inputs
        rowIdx = np.where(np.random.uniform(low=0, high=1, size=numEdgePixels) < probabilityOn)
        colIdx = np.where(np.random.uniform(low=0, high=1, size=numEdgePixels) < probabilityOn)
        if np.any(rowIdx):
            outImage[rowIdx, :] = 1
        if np.any(colIdx):
            outImage[:, colIdx] = 1
        outImages[:, batchIdx] = outImage.reshape((numInputs))
    return subtractMeanPerImage(outImages)

def imprintWeights(weights, dataset):
    """
    Imprint weights with samples from the dataset
    Inputs:
        weights [np.ndarray] of shape (numInputs, numOutputs)
        dataset [np.ndarray] batch of images, each of size
            (numInputs, numDataPoints)
    """
    for neuronIndex in range(weights.shape[1]):
        # Choose random datapoint
        image = dataset[:, np.random.randint(0, dataset.shape[1])]
        weights[:, neuronIndex] = image
    return weights

def sangerLearn(dataset,weights,learningRate):
    """
    Weight update with the Sanger rule.
    weights and learningRate should be provided by output of util.initialize()
    
    Parameters
    ----------
    dataset      : dataset, numpy array, either D1 or D2
    weights      : numpy array, weight matrix of linear transformation of input data
    learningRate : float, factor to multiply weight updates
    
    Returns
    -------
    weights      : numpy array, Sanger-updated weight matrix 
                                 of linear transformation of input data
                                 
    NOTE: if you add any additional parameters to this function, you need to
    also add them to the "argumentsForOjaLearn" list variable
    """
    
    output = weights.T @ dataset # compute neuron output for all data
    numOutputs = output.shape[0]
    
    residual = dataset
    dw = np.zeros(weights.shape)
    
    for i in range(numOutputs):
        
        residual = residual - (weights[:,i,None] @ output[None,i,:])
        
        dw[:,i] = residual @ output[i,:].T

    weights += dw*learningRate # update weight vector by dw
    
    return weights

def l2Norm(weights):
    norm = np.sqrt(np.maximum(np.sum(np.square(weights), axis=0, keepdims=True), np.finfo(float).eps))
    return weights / norm

def getRandomPatch(dataset, patchEdgeSize):
    numImages = dataset.shape[1]
    image = dataset[:, np.random.randint(0, numImages)]
    numPixels = image.shape
    image = image.reshape(int(np.sqrt(numPixels)), int(np.sqrt(numPixels)))
    randCol = np.random.randint(0, np.sqrt(numPixels) - patchEdgeSize)
    randRow = np.random.randint(0, np.sqrt(numPixels) - patchEdgeSize)
    patch = image[randRow:randRow+patchEdgeSize, randCol:randCol+patchEdgeSize]
    return patch.reshape(patchEdgeSize*patchEdgeSize) 

def lcaThreshold(u, threshold):
    """
    Threshold matrix of membrane potentials (u) according to the LCA soft thresholding rule:
    
    a = T(u) = u - threshold, u > threshold
               u + threshold, u < -threshold
               0, otherwise

    Parameters
    ----------
    u : np.ndarray of dimensions (numOutputs, batchSize) holding LCA membrane potentials
    threshold : float indicating Sparse Coding lambda value
    
    Returns
    -------
    a : np.ndarray of dimensions (numOutputs, batchSize) holding thresholded potentials
    """
    a = np.abs(u) - threshold
    a[np.where(a<0)] = 0
    a = np.sign(u) * a  
    return a
