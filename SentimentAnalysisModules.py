# pjoshi, 20.10.2020
# python module containing required classes and functions for the deep learning framework for sentiment analysis

# Import necessary modules

# os package to work with paths, python built in package
import os

# numpy package for array operations
import numpy as np

# pandas package for data processing
#   python -m pip install pandas
import pandas as pd

# sklearn package for machine learning / data mining
from sklearn.linear_model import LogisticRegression, LinearRegression

# keras package for rapid prototyping (uses tensorflow backend)
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

# matplotlib package for plotting
import matplotlib.pyplot as plt
# using the 'ggplot' parameter requires the package 'tornado'
plt.style.use('ggplot')

# Baseline models

# Logistic regression model for classification problems
def LogisticBaseline(X_train, X_test, Y_train, Y_test):
    '''
    function to test the classification with a simple logistic regression as the classifier
    '''

    # Begin with logistic regression for Baseline model
    classifier = LogisticRegression()
    classifier.fit( X_train, Y_train)
    score = classifier.score(X_test, Y_test)
    print("Accuracy:", score)

# Linear regression model for training set with continuous-value labels
def LinearBaseline(X_train, X_test, Y_train, Y_test):
    '''
    function to test the classification with a simple Linear regression as the classifier.
    '''

    # Begin with logistic regression for Baseline model
    classifier = LinearRegression()
    classifier.fit( X_train, Y_train)
    score = classifier.score(X_test, Y_test)
    print("Accuracy:", score)



def PlotKerasHistory(history):
    '''
    function to plot the evolution of:
    \t training and validation accuracy
    \t training and validation loss \n
    \t input : keras history instance
    \t return: None
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1,len(acc) + 1)

    plt.figure( figsize=(12,5) )
    plt.subplot( 1,2,1)
    plt.plot( x, acc, 'b', label = 'Training acc.')
    plt.plot( x, val_acc, 'r', label = 'Validation acc.')
    plt.title( 'Training and Validation accuracy')
    plt.legend()

    plt.subplot( 1,2,2)
    plt.plot( x, loss, 'b', label = 'Training loss')
    plt.plot( x, val_loss, 'r', label = 'Validation loss')
    plt.title( 'Training and Validation loss')
    plt.legend()
    plt.show()



def SimpleNeuralNetwork(X_train, X_test, Y_train, Y_test):
    '''
    function to execute a simple Neural Network:
    \t a dense layer with 10 nodes + a sigmoid activation layer, and
    \t loss = 'binary_crossentropy', optimizer = 'adam'

    \t inputs:
    \t X_train : numpy array of training feature vectors
    \t X_test  : numpy array of testing feature vectors
    \t Y_train : numpy array of training labels
    \t Y_test  : numpy array of testing labels

    \t returns:
    \t history      : a keras history instance containing the evolution of accuracy and loss
    '''

    # Specify the Neural Network Model by adding the appropriate layers
    model = Sequential()
    model.add(
        layers.Dense(10,
        input_dim = X_train.shape[1], 
        activation = 'relu')
    )

    model.add(
        layers.Dense(1,
        activation = 'sigmoid')
    )

    # compile the model and get a summary of the network architecture
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics= ['accuracy']
    )
    model.summary()

    # feed the training data and fit the parameters
    history = model.fit(X_train, Y_train,
                        epochs = 50,
                        verbose = False,
                        validation_data = (X_test,Y_test),
                        batch_size = 10)

    # Get the evaluation metrics of the model for training set
    _, accuracy = model.evaluate(X_train, Y_train, 
                                    verbose = False)
    print("Training Accuracy: {:.4f}".format(accuracy))

    # Get the evaluation metrics of the model for testing set
    _, accuracy = model.evaluate(X_test, Y_test, 
                                    verbose = False)
    print("Testing Accuracy: {:.4f}".format(accuracy))

    return history



# to use pretrained word embeddings
# Download GloVe embeddings pretrained on 6 billion words from : http://nlp.stanford.edu/data/glove.6B.zip
def GetEmbeddingMatrix(DataPathBase, word_index, embedding_dim):
    '''
    function to obtain the pre-trained weights for the Embedding matrix that result in GloVe embeddings.
    \t inputs:
    \t DataPathBase: path to the directory containing the glove embeddings
    \t word_index : dict relating words to index produced by the tokenizer
    \t embedding_dim: dimension of the output embedding vectors
    \t output:
    \t EmbeddingMatrix : numpy array of size (VocabSize, embedding_dim) containing pre-trained weights for words in training corpus
    '''
    filepath = os.path.join( DataPathBase, 'glove.6B', 'glove.6B.50d.txt' )
    VocabSize = len(word_index) + 1
    EmbeddingMatrix = np.zeros( (VocabSize, embedding_dim) )

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                EmbeddingMatrix[idx] = np.array( vector,
                                                 dtype = np.float32)[:embedding_dim]
    
    # check the obtained embedding matrix
    NonZeroElements = np.count_nonzero( np.count_nonzero( EmbeddingMatrix, axis=1) )
    print("Fraction of non-zero elements in Embedding Matrix: {:.2f}".format( NonZeroElements/VocabSize )) 

    return EmbeddingMatrix




# neural network model for sentiment analysis to use in grid search for best hyperparameters
def GetSentimentAnalysisModel(VocabSize=None, EmbeddingDim=None, EmbeddingMatrix=None, MaxLenTokenizer=None, conv1D_filters=20, conv1D_kernel_size=2 ):
    '''
    function to obtain a Neural Network Model with given parameters.
    \t inputs:
    \t VocabSize : size of the vocabulary
    \t EmbeddingDim : dimension of the output embedding vectors
    \t MaxLenTokenizer : Length of the vectors prepared by tokenizer
    \t conv1D_filters : number of filters for the 1D convolution array
    \t conv1D_kernel_size : size of the filter kernel for the 1D convolution array
    
    \t output:
    \t model instance of keras

    \t TODO!:
    \t The KerasClassifier constructor does not set EmbeddingMatrix! Can one pass such arrays?
    '''
    
    model = Sequential()
    model.add( layers.Embedding(input_dim = VocabSize,
                                output_dim = EmbeddingDim,
                                weights = [EmbeddingMatrix],
                                input_length = MaxLenTokenizer,
                                trainable = True))
    model.add( layers.Conv1D( filters = conv1D_filters,
                            kernel_size = conv1D_kernel_size,
                            activation = 'relu'))
    model.add( layers.GlobalMaxPool1D())
    model.add( layers.Dense( 10,
                            activation='relu'))
    model.add( layers.Dense( 1,
                            activation='sigmoid'))
    model.compile( optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    return model
