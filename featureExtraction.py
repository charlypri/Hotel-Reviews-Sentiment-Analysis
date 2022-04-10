import numpy as np
from numpy import array
import math as ma
import csv 
import fileinput
import sys
import os
import time


f1 = open("data/hotelPosT-train.txt", encoding="utf8")
f2 = open("data/hotelNegT-train.txt", encoding="utf8")
f3 = open("data/positive-words.txt", encoding="utf8")
f4 = open("data/negative-words.txt", encoding="utf8")
f5 = open("data/HW2-testset.txt", encoding="utf8")

posReviews = f1.read().split("\n")
negReviews = f2.read().split("\n")
posWords = f3.read().split()
negWords = f4.read().split()

testReviews = f5.read().split("\n")

pronounSet = {"I", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"}

x1 = 0 #count of positive words in data
x2 = 0 #count of negative words in data
x3 = 0 #1 if there is a no 0 if there isnt
x4 = 0 #count of first and second pronouns in doc
x5 = 0 #1 if ! is in the doc 0 if it is not
x6 = 0 #log of total amount of words



#feature extraction loop

################################## TRAINING DATA EXTRACTION ################################3
def generateTrainingData():
    trainFeatureList = []
    for rev in posReviews :
        x3 = 0
        x5 = 0 
        
        #First we split each review in words so we can calculate all the features
        words = rev.split(" ")
        id = words[0].split("\t")
        
        x1 = len(set(posWords).intersection(words))
        x2 = len(set(negWords).intersection(words))
        
        for word in words:
            word = word.lower()
            if len(word.split(",")) > 1 :
                sep = word.split(",")
                word = sep[0]
            if len(word.split(".")) > 1 :
                sep = word.split(".")
                word = sep[0]
            if word == "no":
                x3 = 1
            if len(word.split("!")) > 1:
                x5 = 1
    #         print(word)
        
        x4 =len(set(pronounSet).intersection(words))

        x6 = round(np.log(len(words)), 2)
        v = [str(id[0]), str(x1), str(x2), str(x3), str(x4), str(x5), str(x6), str(1)]
    #     print(v)
        if str(id[0]) != "" :
            trainFeatureList.append(v)

    for rev in negReviews :
        x3 = 0
        x5 = 0 
        
        #First we split each review in words so we can calculate all the features
        words = rev.split(" ")
        id = words[0].split("\t")
        
        for word in words:
            word = word.lower()
            if len(word.split(",")) > 1 :
                sep = word.split(",")
                word = sep[0]
            if len(word.split(".")) > 1 :
                sep = word.split(".")
                word = sep[0]
            if word == "no":
                x3 = 1
            if len(word.split("!")) > 1:
                x5 = 1
    #         print(word)
        
        x1 = len(set(posWords).intersection(words))
        
        x2 = len(set(negWords).intersection(words))
        
        x4 =len(set(pronounSet).intersection(words))

        x6 = round(np.log(len(words)), 2)
    #     print(x6)
        
        v = [str(id[0]), str(x1), str(x2), str(x3), str(x4), str(x5), str(x6), str(0)]
    #     print(v)
        if str(id[0]) != "" :
            trainFeatureList.append(v)
    
    dumpDataToCSV("trainingData.csv", trainFeatureList)

######################################### TEST SET #################################################
def generateTestData():
    testFeatureList = []
    for rev in testReviews :
        x3 = 0
        x5 = 0 
        
        #First we split each review in words so we can calculate all the features
        words = rev.split(" ")
        id = words[0].split("\t")
        
        x1 = len(set(posWords).intersection(words))
        x2 = len(set(negWords).intersection(words))
        
        for word in words:
            word = word.lower()
            if len(word.split(",")) > 1 :
                sep = word.split(",")
                word = sep[0]
            if len(word.split(".")) > 1 :
                sep = word.split(".")
                word = sep[0]
            if word == "no":
                x3 = 1
            if len(word.split("!")) > 1:
                x5 = 1
    #         print(word)
        
        x4 =len(set(pronounSet).intersection(words))

        x6 = round(np.log(len(words)), 2)
        v = [str(id[0]), str(x1), str(x2), str(x3), str(x4), str(x5), str(x6)]
    #     print(v)
        if str(id[0]) != "" :
            testFeatureList.append(v)
    
    dumpDataToCSV("testingData.csv", testFeatureList)

def dumpDataToCSV(filename, featureList):
    with open("draft.csv", 'w+') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(featureList)
        resultFile.close()    

    fd = open("draft.csv", 'r+')
    final = open(filename,'w+')

    for line in fd:
        if line != "\n":
            final.write(line)

    fd.close()
    final.close()
    os.remove("draft.csv")
    for item in featureList:
        print(item)

def main():
    generateTrainingData()
    time.sleep(1)
    generateTestData()
    
main()   
