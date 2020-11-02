# pjoshi, 20.10.2020

# Necessary imports
# os package to work with paths, python built in package
import os


# pandas package for data processing
#   python -m pip install pandas
import pandas as pd

# sklearn package for machine learning / data mining
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV

# keras package for rapid prototyping (uses tensorflow backend)
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

# imports from custom modules file
from MindMapModules import LogisticBaseline, LinearBaseline, GetEmbeddingMatrix, GetSentimentAnalysisModel, SimpleNeuralNetwork, PlotKerasHistory



# Flag to only use a small dataset during code development
SmallData = 1

# Locate data and specify datasets
DataPathBase = os.path.join('..', 'data', 'sentiment_analysis')
filepath_dict = {'yelp': os.path.join(DataPathBase, 'yelp_labelled.txt'),
                'amazon': os.path.join(DataPathBase,'amazon_cells_labelled.txt',),
                'imdb': os.path.join(DataPathBase,'imdb_labelled.txt')}

# Combine data from multiple datasets
df_list = []
for source, filepath in filepath_dict.items():
    # read each datafile
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    # add a new column containing source name
    df['source'] = source
    # append data from each file to the data list
    df_list.append(df)

df = pd.concat(df_list)
#print(df.iloc[0])

# use a small subset for code development
if SmallData:
    df = df[df['source'] == 'yelp']



# prepare raw features and labels
features_arr = df['sentence'].values
labels_arr   = df['label'].values

# prepare the training and testing datasets
features_train, features_test, Y_train, Y_test = train_test_split(  features_arr, 
                                                                    labels_arr, 
                                                                    test_size=0.25, 
                                                                    random_state=1000)

# Use a Bag of Words (BOW) model to create simple feature vectors from training data
vectorizer = CountVectorizer()
vectorizer.fit(features_train)

# transform into proper feature vectors
X_train = vectorizer.transform(features_train)
X_test = vectorizer.transform(features_test)



print("Check baseline for data:")
LogisticBaseline(X_train, X_test, Y_train, Y_test)






# Begin Deep Learning with Neural Network in keras
'''
# execute the simple neural network
SimpleNNHistory = SimpleNeuralNetwork( X_train, X_test, Y_train, Y_test)
# plot the loss/accuracy history
# Note: This ends the program because WebAgg server needs to close with Ctrl+c
PlotKerasHistory(SimpleNNHistory)
'''






# Now start using embedded vectors

# First use the keras tokenizer to create feature vectors
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts( features_train)

X_train = tokenizer.texts_to_sequences(features_train)
X_test = tokenizer.texts_to_sequences(features_test)

# The size of the vocabulary (total words in training data)
# one extra index (0) is reseved for padding purposes
VocabSize = len(tokenizer.word_index) + 1

# check the tokenizer
print(features_train[2])
print(X_train[2])
# check index of individual words
for word in ['the', 'all', 'happy', 'sad']:
    print('{} : {}'.format(word, tokenizer.word_index[word]))

# to make the length of embedded feature vectors uniform, use padding
MaxLenTokenizer = 100
X_train = pad_sequences( X_train, padding='post', maxlen=MaxLenTokenizer)
X_test = pad_sequences( X_test, padding='post', maxlen=MaxLenTokenizer)


# Specify the length of each embedded vector
EmbeddingDim = 50

# hyperparameters not involved in grid search
Epochs = 20
EmbeddingMatrix = GetEmbeddingMatrix(DataPathBase, tokenizer.word_index, EmbeddingDim)

# Run grid search
print('Running grid search for data: ')

# obtain a keras model instance
model = KerasClassifier( build_fn = GetSentimentAnalysisModel,
                        VocabSize = VocabSize,
                        EmbeddingDim = EmbeddingDim,
                        MaxLenTokenizer = MaxLenTokenizer,
                        EmbeddingMatrix = EmbeddingMatrix,
                        epochs = Epochs,
                        batch_size = 10,
                        verbose = False )

# prepare a dictonary of hyperparameters involved in  grid search
ParameterGrid = dict(
                    conv1D_filters = [32, 64, 128],
                    conv1D_kernel_size = [3, 5, 7] )
                    
# obtain a grid search instance
grid = RandomizedSearchCV(  estimator = model,
                            param_distributions = ParameterGrid,
                            cv = 4,
                            verbose = 1,
                            n_iter = 5 )

# use the grid search parameters on training set
grid_result = grid.fit( X_train, Y_train )

# obtain accuracy of grid search parameters on testing set
test_accuracy = grid.score( X_test, Y_test )

# output the best results and parameters
s = ('Best Accuracy : {:.4f}\n{} \nTest Accuracy : {:.4f}\n\n')
output_string = s.format( grid_result.best_score_,
                          grid_result.best_params_,
                          test_accuracy)
print(output_string)

# Save and evaluate results
prompt = input(f'finished grid search; write to file and proceed? [y/n]')
if prompt.lower() in {'y', 'true', 'yes'}:
    OutputFile = os.path.join(DataPathBase, 'output.txt')
    with open(OutputFile, 'a') as f:
        f.write(output_string)
else:
    quit("Finished without saving!")






'''
# Run grid search separately for each source (yelp, amazon, imdb)
for source, frame in df.groupby('source'):
    print('Running grid search for data set: ', source)

    # prepare raw features and labels
    features_arr = df['sentence'].values
    labels_arr   = df['label'].values

    # prepare the training and testing datasets
    features_train, features_test, Y_train, Y_test = train_test_split(  features_arr, 
                                                                        labels_arr, 
                                                                        test_size=0.25, 
                                                                        random_state=1000)
    # First try the keras tokenizer to create feature vectors
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts( features_train)

    X_train = tokenizer.texts_to_sequences(features_train)
    X_test = tokenizer.texts_to_sequences(features_test)

    # one extra index (0) is reseved for padding purposes
    VocabSize = len(tokenizer.word_index) + 1

    X_train = pad_sequences( X_train, padding='post', maxlen=MaxLenTokenizer)
    X_test = pad_sequences( X_test, padding='post', maxlen=MaxLenTokenizer)

    # prepare a dictonary of parameters for grid search of hyperparameters
    ParameterGrid = dict( VocabSize = [VocabSize],
                        EmbeddingDim = [EmbeddingDim],
                        MaxLenTokenizer = [MaxLenTokenizer],
                        conv1D_filters = [32, 64, 128],
                        conv1D_kernel_size = [3, 5, 7] )
    
    # keras model instance
    model = KerasClassifier( build_fn = GetNNModel,
                             epochs = epochs,
                             batch_size = 10,
                             verbose = False )
    
    # search grid process
    grid = RandomizedSearchCV( estimator = model,
                               param_distributions = ParameterGrid,
                               cv = 4,
                               verbose = 1,
                               n_iter = 5 )

    grid_result = grid.fit( X_train, Y_train )
    test_accuracy = grid.score( X_test, Y_test )

    # Save and evaluate results
    prompt = input(f'finished {source}; write to file and proceed? [y/n]')
    if prompt.lower() not in {'y', 'true', 'yes'}:
        break
    # with open(OutputFile, 'a') as f:
    s = ('Running {} data set\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
    output_string = s.format(
        source,
        grid_result.best_score_,
        grid_result.best_params_,
        test_accuracy)
    print(output_string)
    # f.write(output_string)

'''


'''
model.summary()
history = model.fit( X_train, Y_train,
                     epochs= 20,
                     verbose= False,
                     validation_data= (X_test,Y_test),
                     batch_size= 10)

loss, accuracy = model.evaluate( X_train, Y_train, verbose= False)
print( "Training Accuracy: {:.4f}".format( accuracy ) )
loss, accuracy = model.evaluate( X_test, Y_test, verbose=False)
print( "Testing Accuracy: {:.4f}".format( accuracy ) )
PlotKerasHistory(history)
'''
 




