# Author "Keerthi Raj Nagaraja"

# Importing libraries
from math import*
from numpy import*
import numpy as np
#from sympy import Symbol,cos,sin
from operator import*
from numpy.linalg import*
from sklearn import *
from collections import defaultdict
import common
import time
import ctypes
from matplotlib import pyplot as plt
# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

filename='F:\HAR\UCI HAR Dataset\'
# Dataset Used (Change as per your computers path)
#--------------------------------------------------------------------------------------------#

X_train=common.parseFile(filename +'train/X_train.txt')				# Read X Train 
Y_train=(common.parseFile(filename +'train/y_train.txt')).flatten()		# Read Y Train and flatten it to 1D array
X_test=common.parseFile(filename +'test/X_test.txt')				# Read X Test
Y_test=(common.parseFile(filename +'test/y_test.txt')).flatten()			# Read Y test and flatten it to 1D array

print ("Training Examples",len(X_train),len(Y_train))					                # Printing Lengths of Train and Test Data
print ("Testing Examples",len(X_test), len(Y_test))

X_dynamic_train, Y_dynamic_train=common.getDataSubset(X_train, Y_train, [1,2,3])	# Get Train sub data for [1,2,3]
X_nondynamic_train, Y_nondynamic_train=common.getDataSubset(X_train, Y_train, [4,5,6])  # Get Train sub data for [4,5,6]

X_dynamic_test, Y_dynamic_test=common.getDataSubset(X_test, Y_test, [1,2,3])		# Get Test sub data for [1,2,3]
X_nondynamic_test, Y_nondynamic_test=common.getDataSubset(X_test, Y_test, [4,5,6])	# Get Test sub data for [4,5,6]

X_nondynamic_train=common.getPowerK(X_nondynamic_train, [1,2])				# Convert X Train to X+X^2
X_nondynamic_test=common.getPowerK(X_nondynamic_test, [1,2])				# Convert X Test to X+X^2
#X_nondynamic_train_6, Y_nondynamic_train_6=common.getDataSubset(X_train, Y_train, [6]) # Used earlier to get Sub data for just 6th label
#X_nondynamic_test_6, Y_nondynamic_test_6=common.getDataSubset(X_test, Y_test, [6])
#Y_nondynamic_train_sublabels=common.convertLabel(Y_nondynamic_train, [4,5], [6])	# Used earlier to convert [4,5] to [1] and [6] to [0]


print ("Dymnamic Activites Examples",len(X_dynamic_train), len(Y_dynamic_train), Y_dynamic_train)		# Printing lenghts and Labels extracted for verification
print ("Non Dynamic Activites Examples",len(X_nondynamic_train), len(Y_nondynamic_train), Y_nondynamic_train)


sample_weights=common.getSampleWeights( X_nondynamic_train, Y_nondynamic_train , [4,5,6])	# Get sample weights for non-dynamic Data
###############################################################################################

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours( ax,clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(X_dynamic_test)
    
    out=ax.contourf(xx, yy, Z, **params)
    return out

################################################################################################
#Code used for Dynamic Data - Commented for now
print("Dynamic Activites")
clf = svm.LinearSVC(multi_class='crammer_singer')
clf.fit(X_dynamic_train, Y_dynamic_train)

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
xx, yy = make_meshgrid(X_dynamic_train, Y_dynamic_train)
plot_contours(ax,clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
scatter(X_dynamic_train, Y_dynamic_train, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
#ax.set_xlabel('Sepal length')
#ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())

Y_predict_dynamic=clf.predict(X_dynamic_test)
print (type(Y_predict_dynamic), size(Y_predict_dynamic), Y_predict_dynamic)
prec, rec, f_score=common.checkAccuracy(Y_dynamic_test, Y_predict_dynamic, [1,2,3])
print ("precesion=", prec)
print ("recall=",rec)
print (f_score)
print ("Confusion Matrix for Dynamic Activites")
print(common.createConfusionMatrix(Y_predict_dynamic, Y_dynamic_test, [1,2,3]))
#print clf.n_support_'''
################################################################################################

# SVM Code for Linear Kernel with sample weights for non-dynamic classes [4,5,6]
print("Non-Dynamic Activites")
clf = svm.SVC(kernel='linear')
clf.fit(X_nondynamic_train, Y_nondynamic_train, sample_weight=sample_weights) 		# Fit SVM using sample weights
Y_predict_nondynamic=clf.predict(X_nondynamic_test)					# Predict Labels for test data
print (type(Y_predict_nondynamic), size(Y_predict_nondynamic), Y_predict_nondynamic)
prec,rec,f_score=common.checkAccuracy(Y_nondynamic_test, Y_predict_nondynamic, [4,5,6])# Check accuracy
print ("precision=",prec)										# Print Precision, Recall and f-score
print ("recall",rec)
print (f_score)
print ("Confusion matrix for Non-Dynamic Activites")
print(common.createConfusionMatrix(Y_predict_nondynamic, Y_nondynamic_test, [4,5,6]))	# Print Confusion Matrix for the same'''
