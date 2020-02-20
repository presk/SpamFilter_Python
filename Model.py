# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:08:00 2019

@author: Keven
"""
import re
import sys
import numpy

def main():
    load = False
    print("Please select an option:")
    print("1: Generate Base Model")
    print("2: Classify Emails")
    print("3: Classify Emails using Stop-word Filtering")
    print("4: Classify Emails using Word Length Filtering")
    option = input()
    #Executes the option selected by the user
    while not load:
        if(option == '1'):
            buildModel("model.txt")
            load = True
        elif(option == '2'):
             classifyEmailList(buildModel("model.txt"), "baseline-result.txt")
             load = True
        elif(option == '3'):
            classifyEmailList(buildModel("stopword-model.txt"), "stopword-result.txt")
            load = True
        elif(option == '4'):
            classifyEmailList(buildModel("wordlength-model.txt"), "wordlength-result.txt")
            load = True
        else:
            load = False
            print("Invalid option, please try again:")
            option = input()
            
#Takes the model, calculates the result and saves it    
def classifyEmailList(model, fileName):
    print("Classifying emails:")
    
    results = readEmailSet(model)
    save(results, fileName)
       
#Compiles all the data from the test set    
def readEmailSet(model):
    cat = ["ham", "spam"]
    results = []
    j = 1
    for am in cat:
        #print("Classifying "+am+" set...")
        for i in range(1, 401):
            if(i < 10):
                num = "0000" + str(i)
            elif(i < 100):
                num = "000" + str(i)
            elif(i < 1000):
                num = "00" + str(i)
            path = "./test/test-"+am+"-"+num+".txt"
            f = open(path, "r", encoding = "latin-1")
            if f.mode == "r":
                data = f.read()
                #Calculates the score for the email currently being read
                score = classify(data, model)
                #Select the highest score and assigns email type using (argmax(sum log10(prob)))
                if(score[0] >= score[1]):
                    classif = "ham"
                else:
                    classif = "spam"
                if classif == am:
                    res = "right"
                else:
                    res = "wrong"
                results.append([j, "test-"+am+"-"+num+".txt", classif, score[0], score[1], am, res])
                j+=1
            update_progress("Classifying "+am+" emails", i/400.0)
    return results

#Calculates score from ham and spam probabilities    
def classify(email, model):
    score = [0, 0]
    #Splits the data into a list of words
    pattern = re.compile('[^a-zA-Z]')
    data = re.split(pattern, email.lower())
    datadict = {}
    modeldict = {}
    #Filters the data to remove empty words and counts the occurence of each valid word
    for word in data:
        if not word == '':
            datadict[word] = datadict.get(word, 0) + 1
    #Creates a dict where key = word and value = position in the model
    for i in range(len(model)):
        modeldict[model[i][1]] = i
    #Calculates the score for each word and multiply it by the occurence count
    for word in datadict:
        #Checks if the words is in the model (by finding its value in the dict)
        it = modeldict.get(word, -1)
        #If the word is present (return value > -1)
        if it >= 0:
            val0 = float(model[it][3])
            val1 = float(model[it][5])
            score[0] += numpy.log10(val0) * datadict[word]
            score[1] += numpy.log10(val1) * datadict[word]
    return score

#Saves the data to fileName.txt
def save(data, fileName):
    f = open(fileName, "w")
    for line in data:
        for i in line:
            f.write(str(i) + "  ")
        f.write("\n")      

#Builds the model by passing the right option (task 1-2, 3.2 and 3.3) and saves it
def buildModel(fileName):
    if fileName == "model.txt":
        model = readTrainingSet(0)
    elif fileName == "stopword-model.txt":
        model = readTrainingSet(1)
    elif fileName == "wordlength-model.txt":
        model = readTrainingSet(2)
    save(model, fileName)
    return model
    
#Reads the ham and spam training set and passes down the option for the model building   
def readTrainingSet(option):
    am = "ham"
    num = ""
    hamdata = ""
    spamdata = ""
    print("Reading training set...")
    for i in range(1, 1001):
        if(i < 10):
            num = "0000" + str(i)
        elif(i < 100):
            num = "000" + str(i)
        elif(i < 1000):
            num = "00" + str(i)
        elif(i == 1000):
            num = "01000"
        path = "./train/train-"+am+"-"+num+".txt"
        f = open(path, "r")
        if f.mode == "r":
            hamdata += f.read()
        update_progress("Reading ham", i/1000.0)
    am = "spam"
    for i in range(1, 998):
        if(i < 10):
            num = "0000" + str(i)
        elif(i < 100):
            num = "000" + str(i)
        elif(i < 1000):
            num = "00" + str(i)
        path = "./train/train-"+am+"-"+num+".txt"
        f = open(path, "r")
        if f.mode == "r":
            spamdata += f.read()  
        update_progress("Reading spam", i/997.0)   
    return createWordBank(hamdata, spamdata, option)

#Generates the list of words(wordBank) for the model
def createWordBank(hamdata, spamdata, option):
    print("")
    print("Generating word bank...")
    wordBank = []
    pattern = re.compile('[^a-zA-Z]')
    hamlist = re.split(pattern, hamdata.lower())
    spamlist = re.split(pattern, spamdata.lower())
    wordlist = hamlist + spamlist
    wordBank = dict.fromkeys(wordlist)
    return createModel(wordBank, hamlist, spamlist, option)
 
#Generates the model that fits the user's option
def createModel(wordBank, hamlist, spamlist, option):
    print("")
    model = []
    wordBank.pop('')
    tempBank = wordBank.copy()
    smoothing = 0.5
    hamdict = dict()
    spamdict = dict()
    hamlen = 0
    spamlen = 0
    stopwords = []
    #Removes stopwords from the wordBank if option 1 was selected
    if option == 1:
        path = "english-stopwords.txt"
        f = open(path, "r")
        if f.mode == "r":
            data = f.read()
            stopwords = data.split()
            for word in stopwords:
                wordBank.pop(word, 0)
    #Removes short and long words if option 2 was selected
    if option == 2:
        for word in tempBank:
            if len(word) <= 2:
                wordBank.pop(word)
            if len(word) >= 9:
                wordBank.pop(word)
    wordBank = sorted(wordBank)     

    #Counts the word occurence in the list of words from ham emails while filtering empty words      
    for word in sorted(hamlist):
        if not word == '':
            #Avoids unnecessarily counting short and long words if option 2 is selected
            if option == 2: 
                if len(word) > 2 and len(word) < 9:
                    hamdict[word] = hamdict.get(word, 0) + 1
            else:
                hamdict[word] = hamdict.get(word, 0) + 1

    #Counts the word occurence in the list of words from ham emails while filtering empty words
    for word in sorted(spamlist):
        if not word == '':
            #Avoids unnecessarily counting short and long words if option 2 is selected
            if option == 2:
                if len(word) > 2 and len(word) < 9:
                    spamdict[word] = spamdict.get(word, 0) + 1
            else:
                spamdict[word] = spamdict.get(word, 0) + 1
            
    bankLen = len(wordBank)
    i = 0
    #Adds each word to the model with their respective count and probability for ham and spam emails
    #using (count + smoothing)/(number of words * (smoothing + 1))
    for word in wordBank:
        i += 1
        hamcount = hamdict.get(word, 0)
        hamprob = (hamcount + smoothing)/(bankLen *(smoothing + 1))
        spamcount = spamdict.get(word, 0)
        spamprob = (spamcount + smoothing)/(bankLen *(smoothing + 1))
        model.append([i, word, hamcount, hamprob, spamcount, spamprob])
        update_progress("Generating base model", i/bankLen)
    return model


#https://blender.stackexchange.com/questions/3219/how-to-show-to-the-user-a-progression-in-a-script
def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()    
    

    
if __name__ == "__main__":
    main()