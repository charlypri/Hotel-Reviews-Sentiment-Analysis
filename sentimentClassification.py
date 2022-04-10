import numpy as np
from numpy import array
import math as ma
import csv 
import random
from tqdm import tqdm

# we declare our main variables that we will need in order to do the model for the logistic regression
weights = np.array([0, 0, 0, 0, 0, 0, 1])
bias = 1
learningrate = 0.1


#############################  TRAINING LOOP  ###################################################################
def trainLinearRegression(weights):
    with open('trainingData.csv', 'r') as f:
        reader = csv.reader(f)
        trainingVectors = np.array(list(reader)) #we convert the list of lists we read into a numpy array
        # print(trainingVectors) 
    
    # we iterate a sufficient amount of times till the values in our weight table converge
    for i in tqdm(range(0, 20000)):
        
        limit = len(trainingVectors)-1
        pos = random.randint(1,limit) #we generate random indexes

        # here we create a vector that only contains the the features of the vectors casted to numerical values
        randomvector = np.array([int(trainingVectors[pos][1]), int(trainingVectors[pos][2]), int(trainingVectors[pos][3]), int(trainingVectors[pos][4]), int(trainingVectors[pos][5]), float(trainingVectors[pos][6]), bias])
    #     print(randomvector)

        # we calculate the rawscore doing the dot product between our current weights and the vector randomly selected
        rawscore = np.dot(weights, randomvector) 
        score = 1/(1+np.exp(-rawscore))
        # print(score)

        correct = int(trainingVectors[pos][7]) #we look at the correct  vector´s label 
        gradient = (score - correct) * randomvector
        weights = weights - learningrate * gradient #updating values for the weights

#############################  TESTING MODEL LOOP  ###################################################################
def testLinearRegression(weights):
    with open('trainingData.csv', 'r') as f:
        reader = csv.reader(f)
        testingVectors = np.array(list(reader)) #we convert the list of lists we read into a numpy array

    goodGuesses = 0
    guesses =  len(testingVectors)
    for i in range(0, len(testingVectors)):
        
        testVector = np.array([int(testingVectors[i][1]), int(testingVectors[i][2]), int(testingVectors[i][3]), int(testingVectors[i][4]), int(testingVectors[i][5]), float(testingVectors[i][6]), bias])
        rawscore = np.dot(weights, testVector) 
        score = 1/(1+np.exp(-rawscore))
        correct = int(testingVectors[i][7])
        if score <= 0.5 :
            score = 0
        else : score = 1
    #   debugging  
    #     print(testingVectors[i][0],correct, score) 
        
        if correct == score:
            goodGuesses+= 1
    # we print the percentage of correct guesses our model has made
    print (goodGuesses / guesses)
            

###################################### Testing against new test data  ###########################
def extraTestLinearRegression():

    final = open("Results.txt",'w+') # here we will write the output of our logistic regression model on the test set
    with open('extraTestSet.csv', 'r') as f:
        reader = csv.reader(f)
        finalVectors = np.array(list(reader)) #we convert the list of lists we read into a numpy array

    for i in range(0, len(finalVectors)):
        
        finalSetWWeights = np.array([1.52271241, -1.75498843, -0.53964048, 0.05263496, 1.44978377, -1.36963731, 3.52928002 ])
        
        testVector = np.array([int(finalVectors[i][1]), int(finalVectors[i][2]), int(finalVectors[i][3]), int(finalVectors[i][4]), int(finalVectors[i][5]), float(finalVectors[i][6]), bias])
        rawscore = np.dot(finalSetWWeights, testVector) 
        score = 1/(1+np.exp(-rawscore))
        if score <= 0.5 :
            score = 0
        else : score = 1
    #   debugging  
        print(finalVectors[i][0], score) 
        
        if score == 1 :
                final.write(finalVectors[i][0]+" POS \n")
        else :
                final.write(finalVectors[i][0]+" NEG \n") 



trainLinearRegression(weights)
testLinearRegression(weights)
extraTestLinearRegression()

