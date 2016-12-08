import math
import random
import numpy as np
import matplotlib.pyplot as plt

'''
Below code provided by Kunal Marwaha on piazza: https://piazza.com/class/ijltj19y6lv59e?cid=41
'''
#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

#montage_images.m, converted
def montage_images(images):
    num_images=min(1000,np.size(images,2))
    numrows=math.floor(math.sqrt(num_images))
    numcols=math.ceil(num_images/numrows)
    img=np.zeros((numrows*28,numcols*28));
    for k in range(num_images):
        r = k % numrows
        c = k // numrows
        img[r*28:(r+1)*28,c*28:(c+1)*28]=images[:,:,k];
    return img
'''
Above code provided by Kunal Marwaha on piazza: https://piazza.com/class/ijltj19y6lv59e?cid=41
'''

# Get a random n samples from provided list
def get_n_samples(data, n):
    if n > len(data):
        print("ERROR: Invalid number of samples")
        return -1
    return [data[x] for x in random.sample(xrange(len(data)), n)]

#Assumes input is of size (x,y,z) where we have z distinct matrixes fo dim x,y
# and we will return a single matrix of size (z, x*y)
def vectorize(data):
    x, y, z = data.shape
    return np.reshape(data, (x*y, z)).swapaxes(0,1)

#Assumes input is of size (z,y,x) where we have z distinct matrixes fo dim x,y
# and we will return a single matrix of size (z, x*y)
def vectorize2(data):
    z, y, x = data.shape
    data = data.swapaxes(1,2)
    return np.reshape(data, (z, x*y))


# Modified source code at http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# and posted on piazza at https://piazza.com/class/ijltj19y6lv59e?cid=42
def plot_confusion_matrix(cm, targets, title='Confusion matrix', cmap=plt.cm.afmhot):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(targets))
    plt.xticks(tick_marks, targets)
    plt.yticks(tick_marks, targets)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')