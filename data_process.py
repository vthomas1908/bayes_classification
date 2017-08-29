import os
import sys
import math

#parameters:
#[1]: data file
#[2]: train file
#[3]: test file

#vars
sample_size = 0
total_pos = 0
total_neg = 0
half_pos = 0
half_neg = 0
pos_count = 0
neg_count = 0
train_size = 0
test_size = 0
data = []
data_new = []
training = []
testing = []

#get the files
file_path = sys.argv[1]
write_training_path = sys.argv[2]
write_testing_path = sys.argv[3]

#make sure file exists
if not os.path.exists(file_path):
  print(file_path + " does not exit");
  sys.exit(0)

#open the file
text_file = open(file_path)

#create a list of the lines
list_data = text_file.readlines()

#close file
text_file.close()

#get split the data on ","
for line in list_data:
  data.append(line.split(","))

#convert string data to float and put
#class as first value
for y in range(len(data)):
  temp_line = []
  for x in range(58):
    temp_line.append(float(data[y][x]))
  temp_line.insert(0, temp_line.pop(len(temp_line)-1))

  data_new.append(temp_line)

#count positive and negative classes
#so we can make sure test/train have
#right number of each class
for line in data_new:
  if line[0] == 1:
    total_pos += 1
  else:
    total_neg += 1

#half values shows how many to put in
#test and train files
half_pos = total_pos//2
half_neg = total_neg//2

# split into training and testing sets
for line in data_new:
  if line[0] == 1:
    if pos_count < half_pos:
      testing.append(line)
      pos_count += 1
    else:
     training.append(line)
     pos_count += 1
  else:
    line[0] = -1.0
    if neg_count < half_neg:
      testing.append(line)
      neg_count += 1
    else:
      training.append(line)
      neg_count += 1

# write the training file (input parameter)
text_file = open(write_training_path, "w")

for i in range(len(training)):
  for j in range(len(training[0])):
    text_file.write(str(training[i][j]) + " ")
  text_file.write("\n")

text_file.close()

# write test file (input parameter)
text_file = open(write_testing_path, "w")

for i in range(len(testing)):
  for j in range(len(testing[0])):
    text_file.write(str(testing[i][j]) + " ")
  text_file.write("\n")

text_file.close()

