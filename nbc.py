#Creates probabilistic model on training data
#Classify test data based on model

#run as python3 nbc.py <training data> <testing data>

import sys
import os
import math

#function to put data from file into working list
def format_data(data_file):
  #make sure file exists
  if not os.path.exists(data_file):
    print(file_path + " does not exit")
    sys.exit(0)

  #open the file
  text_file = open(data_file)

  #create a list of the lines
  data = text_file.readlines()

  #close file
  text_file.close()

  #put data in float form (instead of string)
  for line in range(len(data)):
    new = []

    for val in data[line].split(" "):

      if val != "\n":
        new.append(float(val))

    data[line] = new

  return data

#Class for probabilistic model
class Model(object):
  def __init__(self, data):
    self.data = data
    self.mean_pos = []
    self.mean_neg = []
    self.st_dev_pos = []
    self.st_dev_neg = []
    self.p_pos = 0
    self.p_neg = 0

    self.total_pos = 0
    self.total_neg = 0
    self.total = 0

  #function to get mean for pos and neg
  #for each feature
  #also calcs prob of pos and neg (bonus!)
  def get_mean(self):
    #note total samples
    self.total = len(self.data)

    #some temp arrays to keep track of 
    #running totals for features
    temp_pos = [0] * 57
    temp_neg = [0] * 57

    for sample in self.data:
      #pos training samples
      if sample[0] == 1:
        self.total_pos += 1

        #increment temp pos for each feature
        for feature in range(57):
          #don't forget sample has class as 1st val
          temp_pos[feature] += sample[feature + 1]

      #neg training samples
      else:
        self.total_neg += 1

        #increment temp neg for each feature
        for feature in range(57):
          temp_neg[feature] += sample[feature + 1]

    #now get the mean
    for feature in range(57):
      self.mean_pos.append(temp_pos[feature]/self.total_pos)
      self.mean_neg.append(temp_neg[feature]/self.total_neg)

    #also, get the class's probability of pos and neg
    self.p_pos = self.total_pos/self.total
    self.p_neg = self.total_neg/self.total


  #function to get standard deviation
  #for pos and neg for each feature
  def get_st_dev(self):
    #must calc mean first!!!!

    #temp lists to keep track of
    #running totals for features
    temp_pos = [0] * 57
    temp_neg = [0] * 57

    for sample in self.data:
      #pos training samples
      if sample[0] == 1:

        #increment temp_pos for each feature
        #by the (feature value - feature standard deviation) ^2
        for feature in range(57):
          temp = sample[feature + 1] - self.mean_pos[feature]
          temp_pos[feature] += math.pow(temp, 2)

      #neg training samples
      else:
        for feature in range(57):
          temp = sample[feature + 1] - self.mean_neg[feature]
          temp_neg[feature] += math.pow(temp, 2)

    #take squre root of (value in temp list / total class (+/-))
    for feature in range(57):
      self.st_dev_pos.append(math.sqrt(temp_pos[feature]/self.total_pos))
      self.st_dev_neg.append(math.sqrt(temp_neg[feature]/self.total_neg))

      #if 0, assign minimal standard deviation of .0001
      #to avoid divide-by-zero error
      if self.st_dev_pos[feature] == 0:
        self.st_dev_pos[feature] = .0001
      if self.st_dev_neg[feature] == 0:
        self.st_dev_neg[feature] = .0001


#class for classification samples from file based on model
class Tester(object):
  def __init__(self, data, model):
    self.model = model
    self.data = data
    self.classed = []

    self.total_pos = 0
    self.total_neg = 0
    self.total = 0

    self.true_pos = 0
    self.false_pos = 0
    self.true_neg = 0
    self.false_neg = 0

    self.accuracy = 0
    self.precision = 0
    self.recall = 0

  #run Naive Bayes on test data to determine classification
  def classify(self):

    #sqrt(2 * pi) gets used a lot, so pre-calc that
    x = math.sqrt(2 * math.pi)

    #note the sample size
    self.total = len(self.data)

    for sample in self.data:

      #update totals based on this sample
      if sample[0] == 1:
        self.total_pos += 1
      else:
        self.total_neg += 1

      #note extra place holder for probability of class
      pos = [0] * 58
      neg = [0] * 58
      final_pos = 0
      final_neg = 0
      final_class = 0

      #first index is the probability of +/-
      pos[0] = math.log(self.model.p_pos)
      neg[0] = math.log(self.model.p_neg)

      #build pos/neg calculations for each feature
      for feature in range(1, 58):
        #get some variables that will be used for this feature
        #note mean & standard dev will be off by one since
        #index 0 = feature 1
        mean_pos = self.model.mean_pos[feature - 1]
        mean_neg = self.model.mean_neg[feature - 1]
        dev_pos = self.model.st_dev_pos[feature - 1]
        dev_neg = self.model.st_dev_neg[feature - 1]

        #pos list
        val1 = 1/(x * dev_pos)
        val2 = -1 * math.pow(sample[feature] - mean_pos, 2)
        val3 = 2 * math.pow(dev_pos, 2)
        val4 = val2/val3
        if val4 < -740:
          val4 = -740
        pos[feature] = math.log(val1 * math.exp(val4))
        
        #neg list
        val1 = 1/(x * dev_neg)
        val2 = -1 * math.pow(sample[feature] - mean_neg, 2)
        val3 = 2 * math.pow(dev_neg, 2)
        val4 = val2/val3
        if val4 < -740:
          val4 = -740
        neg[feature] = math.log(val1 * math.exp(val4))

      #add up the lists, compare, max is the class!
      for feature in range(58):
        final_pos += pos[feature]
        final_neg += neg[feature]

      if final_pos > final_neg:
        final_class = 1
      else:
        final_class = -1

      self.classed.append(final_class)

      #accuracy updates
      #increment TP/FP/TN/FN
      if final_class == 1:
        if sample[0] == final_class:
          self.true_pos += 1
        else:
          self.false_pos += 1
      if final_class == -1:
        if sample[0] == final_class:
          self.true_neg += 1
        else:
          self.false_neg += 1

  def finalize_accuracy(self):
    #accuracy
    self.accuracy = (self.true_pos + self.true_neg)/self.total

    #precision
    self.precision = self.true_pos/(self.true_pos + self.false_pos)

    #recall
    self.recall = self.true_pos/(self.true_pos + self.false_neg)

    #output the results
    print("accuracy = ", self.accuracy)
    print("precision = ", self.precision)
    print("recall = ", self.recall)

    #output the confusion matrix
    print("\nconfusion matrix: ")
    print("    ", self.true_pos, "  ", self.false_neg)
    print("    ", self.false_pos, "  ", self.true_neg)  


#get training and testing data
training = format_data(sys.argv[1])
testing = format_data(sys.argv[2])


#create the probabilistic model
model = Model(training)

#get mean of pos class features 
#and neg class features
#this will also calc P(pos) and P(neg) for class
model.get_mean()
#get standard deviation of pos class features
#and neg class features
model.get_st_dev()

#create a classifier for testing
classifier = Tester(testing, model)
classifier.classify()
classifier.finalize_accuracy()

