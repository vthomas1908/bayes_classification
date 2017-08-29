import numpy as np
from sklearn import linear_model, metrics
import sys
import os

#get the training and testing data
class Data(object):
  def __init__(self, data_file):
    self.features = []
    self.target = []
    self.data_file = data_file

  def format_data(self):
    #make sure file exists
    if not os.path.exists(self.data_file):
      print(self.data_file, " does not exist")
      sys.exit(0)

    #open file
    text_file = open(self.data_file)

    #create list of lines
    data = text_file.readlines()

    #close file
    text_file.close()

#    print(data)

    #put data in float form (instead of string)
    for line in range(len(data)):
      new = []

      for val in data[line].split(" "):
        if val != "\n":
          new.append(float(val))

      data[line] = new
#      print(data[line], "\n\n")

    #now get separate the data as needed for sklearn
    for line in data:
      #get the target
      self.target.append(line[0])

      #get the features
      temp_x = line[1:]
      self.features.append(temp_x)

#get the training and testing data sets
training = Data(sys.argv[1])
testing = Data(sys.argv[2])
training.format_data()
testing.format_data()

#create the logistic regression model
model = linear_model.LogisticRegression()

#fit (train) the model on the training data
model.fit(training.features, training.target)
print(model)

#classify based on testing data
classed = model.predict(testing.features)

print(classed)

print(metrics.classification_report(testing.target, classed))
print(metrics.confusion_matrix(testing.target, classed))
