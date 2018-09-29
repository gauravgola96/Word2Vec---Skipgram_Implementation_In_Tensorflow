import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import bz2
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk # standard preprocessing
import operator # sorting items in dictionary by value
##nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil


def read_data_without_preprocess(filename):
  """Extract the first file enclosed in a zip file as a list of words"""

  with bz2.BZ2File(filename) as f:
    data = []
    file_string = f.read().decode('utf-8')
    file_string = nltk.word_tokenize(file_string)
    data.extend(file_string)
  return data
  
    

def read_data_with_preprocess(filename):
  """
  Extract the first file enclosed in a zip file as a list of words
  and pre-processes it using the nltk python library
  """

  with bz2.BZ2File(filename) as f:

    data = []
    file_size = os.stat(filename).st_size
    chunk_size = 1024 * 1024 # reading 1 MB at a time as the dataset is moderately large
    print('Reading data...')
    for i in range(ceil(file_size//chunk_size)+1):
        bytes_to_read = min(chunk_size,file_size-(i*chunk_size))
        file_string = f.read(bytes_to_read).decode('utf-8')
        file_string = file_string.lower()
        # tokenizes a string to words residing in a list
        file_string = nltk.word_tokenize(file_string)
        data.extend(file_string)
  return data
