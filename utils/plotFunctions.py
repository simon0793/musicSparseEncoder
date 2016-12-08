import matplotlib.pyplot as plt
import numpy as np

def plotFoldiak(data, title="", prevFig=None):
  (activity, thresholds, corr, latWeights) = data
  if prevFig is not None:
    (fig, rects, axisImage) = prevFig
    for rect, h in zip(rects[0], activity):
      rect.set_height(h)
    for rect, h in zip(rects[1], thresholds):
      rect.set_height(h)
    axisImage[0].set_data(corr)
    axisImage[1].set_data(latWeights)
  else:
    fig, subAxes = plt.subplots(nrows=2, ncols=2)
    rects = [None]*2
    axisImage = [None]*2
    rects[0] = subAxes[0,0].bar(np.arange(len(activity)), activity)
    subAxes[0,0].set_title("Activity")
    rects[1] = subAxes[0,1].bar(np.arange(len(thresholds)), thresholds)
    subAxes[0,1].set_ylim([0, 2])
    subAxes[0,1].set_title("Threshold")
    axisImage[0] = subAxes[1,0].imshow(corr, cmap="Greys", interpolation="nearest")
    subAxes[1,0].set_title("Output Correlations")
    subAxes[1,0].tick_params(axis="both", bottom="off", top="off", left="off", right="off")
    axisImage[1] = subAxes[1,1].imshow(latWeights, cmap="Greys", interpolation="nearest")
    subAxes[1,1].set_title("Lateral Weights")
    subAxes[1,1].tick_params(axis="both", bottom="off", top="off", left="off", right="off")
  fig.suptitle(title)
  if prevFig is not None:
    fig.canvas.draw()
  else:
    fig.show()
  return (fig, rects, axisImage)


def plotDataTiled(data, title="", prevFig=None):
  """
    Create a matplotlib plot that displays data as subplots
    Inputs:
      data [np.ndarray] 2-D array of dims (numPixels, batchSize)
        It is assumed that numPixels has an even square root
      title [str] optional string to set as the figure title
  """
  if len(data.shape) == 2:
    (numPixels, batchSize) = data.shape
  elif len(data.shape) == 1:
    numPixels = data.shape
    batchSize = 1
    data = data[:,None]
  else:
    assert 0, ("len(data.shape) should = 2 or 1")
  assert numPixels%np.sqrt(numPixels) == 0, (
        "data.shape[0]  must have an even square root")
  pixelEdgeSize = int(np.sqrt(numPixels))
  subplotEdgeSize = int(np.ceil(np.sqrt(batchSize)))
  if prevFig is not None:
    (fig, subAxes, axisImage) = prevFig
  else:
    fig, subAxes = plt.subplots(nrows=subplotEdgeSize, ncols=subplotEdgeSize)
    axisImage = [None]*len(fig.axes)
  for axisIndex, axis in enumerate(fig.axes):
    if axisIndex < batchSize:
      image = data[:,axisIndex].reshape(pixelEdgeSize, pixelEdgeSize)
      if prevFig is not None:
        axisImage[axisIndex].set_data(image)
      else:
        axisImage[axisIndex] = axis.imshow(image, cmap="Greys",
          interpolation="nearest")
        axis.tick_params(
          axis="both",
          bottom="off",
          top="off",
          left="off",
          right="off")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    else:
      for spine in axis.spines.values():
        spine.set_visible(False)
      axis.tick_params(
        axis="both",
        bottom="off",
        top="off",
        left="off",
        right="off")
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
  fig.suptitle(title)
  if prevFig is not None:
    fig.canvas.draw()
  else:
    fig.show()
  return (fig, subAxes, axisImage)

def makeSubplots(dataList, labelList, title=""):
  assert len(dataList) == len(labelList), (
    "The lengths of dataLst and labelList must be equal.")
  numSteps = np.arange(dataList[0].size)[-1]
  fig, subAxes = plt.subplots(2, 2)
  subAxes[0, 0].scatter(np.arange(dataList[0].size), dataList[0], alpha=0.2)
  subAxes[0, 0].set_xlabel("Time Step")
  subAxes[0, 0].set_ylabel(labelList[0])
  subAxes[0, 0].set_xlim(0, numSteps)
  subAxes[0, 1].scatter(np.arange(dataList[1].size), dataList[1], alpha=0.2)
  subAxes[0, 1].set_xlabel("Time Step")
  subAxes[0, 1].set_ylabel(labelList[1])
  subAxes[0, 1].set_xlim(0, numSteps)
  subAxes[1, 0].scatter(np.arange(dataList[2].size), dataList[2], alpha=0.2)
  subAxes[1, 0].set_xlabel("Time Step")
  subAxes[1, 0].set_ylabel(labelList[2])
  subAxes[1, 0].set_xlim(0, numSteps)
  subAxes[1, 1].axis("off")
  fig.suptitle(title)
  fig.show()
