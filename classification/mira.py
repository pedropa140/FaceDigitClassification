# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import math
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    best_weights = {}
    best_accuracy = float('-inf')

    for grid in Cgrid:
      current_weight = {}
      for key in self.weights:
        current_weight[key] = self.weights[key]
      for i in range(self.max_iterations):
        for training_index in range(len(trainingData)):
          training_weight = trainingData[training_index]
          score_prediction = float('-inf')
          label_prediction = float('-inf')
          for label_index in self.legalLabels:
            trainingCurrentWeight = training_weight * current_weight[label_index]
            if trainingCurrentWeight > score_prediction:
              score_prediction = trainingCurrentWeight
              label_prediction = label_index
          label_index = trainingLabels[training_index]
          if label_prediction != label_index:
            f = training_weight.copy()
            if grid < ((current_weight[label_prediction] - current_weight[label_index]) * f + 1.0) / (2.0 * (f * f)):
              tau = grid
            else:
              tau = ((current_weight[label_prediction] - current_weight[label_index]) * f + 1.0) / (2.0 * (f * f))
            for i in range(len(f)):
              f[i] = f[i] / (1.0 / tau) 
            current_weight[label_prediction] = current_weight[label_prediction] - f
            current_weight[label_index] = current_weight[label_index] + f
      guesses = []
      for datum in validationData:
        vectors = {}
        for l in self.legalLabels:
          vectors[l] = self.weights[l] * datum
        max_value = float('-inf')
        max_key = None
        for key in vectors:
          value = vectors[key]
          if value > max_value:
            max_value = value
            max_key = key
        guesses.append(max_key)
        
      correct_counter = 0
      for i in range(len(validationLabels)):
        if guesses[i] == validationLabels[i]:
          correct_counter = correct_counter + 1
      correct_float = float(correct_counter)
      accuracy_float = correct_float / len(guesses)
      accuracy = float(accuracy_float)

      if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = current_weight

    self.weights = best_weights

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

