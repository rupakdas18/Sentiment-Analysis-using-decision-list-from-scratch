"""
Created on Sat Oct 10 20:48:30 2021
@author: Rupak Kumar Das
Course: Natural Language Processing, CS 5242
Submission Date: 21 october, 2021

Problem Description:
    This is a sentiment classifier trained for classifying movie reviews. The training dataset contains 1186 reviews
    (592 negative and 594 positives). The test dataset contains 200 reviews (100 positives and 100 negative but unmarked).
    The gold dataset contains the actual classes of those reviews (with a unique review id). The objective is to use a  decision
    list tree to classify those test reviews using unigram, bigram, and not handling features and find the accuracy, precision, and recall.
    
Example of Input and output:
    As the name of the input and output files are fixed, I didn't use any command line. But the input and output should be like below.
    program_file_name input_file  -->  output_file
    decision-list-train.py sentiment-train.txt  > sentiment-decision-list.txt
    decision-list-test.py  sentiment-decision-list.txt sentiment-test.txt  > sentiment-system-answers.txt
    decision-list-eval.py sentiment-gold.txt sentiment-system-answers.txt > sentiment-system-answers-scored.txt
    
Algorithm:
    1. Train the model and get the decision list
        1.1: For every review
            1.1.1 : lower case it
            1.1.2 : convert the short words into expanded form
            1.1.3: use Not_nadler function
            1.1.4: find out the class. If it's positive, store it in pos_list, if negative then in neg_list
        1.2: create the unigram, bigram dictionary with corresponding frequencies of both classes
        1.3: Create the probability dictionary of both classes
        1.4: find the log-likelihood of each item
            1.4.1: if log-likelihood is positive, mark the word as positive
            1.4.2: if log-likelihood is negative, mark the word as negative
            1.4.3: else ignore (log-likelihood value of zero doesn't have any impact)
        1.5: Create decision list (format: ngram (tablespace) absolute_value_of_log (tablespace) class) for each classes
        1.6: sort the decision list based on log-likelihood value and write it
        
    2. Test. Predict the class with the help of a decision list
        2.1: For every test review
            2.1.1: lowercase it
            2.1.2: convert the short words into expanded form
            2.1.3: use Not_nadler function
            2.1.4: create uni_gram and bi_gram list
            2.1.5: For every item in the decision list
                2.1.5.1: compare it with each item of the uni_gram and bi_gram list. if it finds a match, it returns the corresponding class.
                        if not then randomly select a value from 0 and 1 and return it
        2.2: Create the predicted list and write it in a file
    
    3. Evaluate the result
        3.1: open actual and predicted files and store the data in a dictionary (key = review_id and value = class)
        3.2: for each item of 2 dictionaries compare it
            3.2.1: if actual = + , predict = +, then true positive
            3.2.2: if actual = + , predict = 1, then false negative
            3.2.3: if actual = - , predict = +, then false positive
            3.2.4: if actual = - , predict = -, then true negative
        3.3: calculate, accuracy, precision and recall and write it      
"""

# Import libraries
import re
import math

# This is the not_handler feature that I found in stackoverflow. It is used to deal with "not" (example: do not like it. --> do not_like not_it)
# https://stackoverflow.com/questions/23384351/how-to-add-tags-to-negated-words-in-strings-that-follow-not-no-and-never
# first the word boundary is selected --> r"\b(?:no|don't|not)\b followed by alphanumeric and spaces (^\w and \s), up until a punctuation.
#then add not_

def not_handler(string): 
   transformed = re.sub(r"\b(?:no|don't|not)\b[\w\s]+[^\w\s]", 
       lambda match: re.sub(r"(\s+)(\w+)", r"\1not_\2", match.group(0)), 
       string, flags=re.IGNORECASE)

   return transformed

# This function converts the short words to their respective expanded version. 
def decontracted(sentence):
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    return sentence

# This Function creates a Unigram dictionary with respective frequencies
def uni_dic_create(paragraph):
    uni_dic = {}
    for word in paragraph.split():
        if word not in uni_dic.keys():
            uni_dic[word] = 1
        else:
            uni_dic[word] = uni_dic[word] + 1
    return uni_dic

# This Function creates a Bigram dictionary with respective frequencies
def bi_dic_create(paragraph):
    bi_dic = {}
    paralist = []
# create a list of all words
    for word in paragraph.split():
        paralist.append(word)
# Here 2 adjacent words are selected and a list is created. Then added those
# using a join operation
    for i in range(len(paralist)-2):
        temp_list = []
        bi_sent = ''
        temp_list.append(list(paralist[i].split(" ")))
        temp_list.append(list(paralist[i+1].split(" ")))
# covert from a list of list to a flat list
        flatList = [ item for elem in temp_list for item in elem]
        bi_sent = " ".join(flatList)
# creation of the bigram dictionary
        if bi_sent not in bi_dic:
            bi_dic[bi_sent] = 1
        else:
            bi_dic[bi_sent] = bi_dic[bi_sent] + 1
    return bi_dic

# Find the probability of n_gram tokens (frequency/total)
def probability(word_list):
    unigram_dic = uni_dic_create(word_list)
    bigram_dic = bi_dic_create(word_list)
    
    #create a full list with bigrams and unigrams
    full_dic = {**unigram_dic,**bigram_dic}
    
    prob_dic = {}    
    total = sum(full_dic.values())
    
    # Add probabilities with the token in a dictionary
    for key in full_dic.keys():
        prob_dic[key] = float(full_dic[key]/total)
        
    return prob_dic

# To create a decision list
def decision_list_creat(pos_prob, neg_prob):
    
    # if a token is found in both dictionary, find the log-likelihood value.
    # if it's positive, assign to positive class
    # if negative then to negative class
    # else (zero) then ignore
    # To make the decision list smaller, i removed the items with loglikelihood values between -0.5 to 0.5. In doesn't affect the accuracy.
    # add to the decision list
    for key in pos_prob:
        if key in neg_prob:            
            value = round(math.log2(pos_prob.get(key)/neg_prob.get(key)),4)
            abs_value = abs(value)
            if value > 0.5:
                decision_list.append([key,abs_value,1])
            elif value < -0.5 :
                decision_list.append([key,abs_value,0])
            else:
                pass
    # if the token is found in pos dictionary but not in neg dictionary, mark the token as potivite     
        else:
            abs_value = round(abs(math.log2((pos_prob.get(key)+1)/1)),4)
            if abs_value != 0.0:
                decision_list.append([key,abs_value,1])     
                
    # if the token is found in neg dictionary but not in pos dictionary, mark the token as negative             
    for key in neg_prob:
        if key not in pos_prob:
            abs_value = round(abs(math.log2(1/(neg_prob.get(key)+1))),4)
            if abs_value != 0:
                decision_list.append([key,abs_value,0]) 
               

if __name__=="__main__":
    print("Hello. This program is written by Rupak Kumar Das. This is a decision list classifier\
          that performs sentiment analysis.")
          
    pos_list = ''
    neg_list = ''
    decision_list = []
    
# Open the train data file, lowercase it, expant the short words, use not_handler function
    with open('sentiment-train.txt','r') as input_file:
        for line in input_file:
            line = line.lower()
            line = decontracted(line)
            line = not_handler(line)
            line = line.split()
            
            # if the class == 1, then postive list
            if line[1] == '1':
                pos_list = pos_list + ' '.join(line[2:])
            
            # if the class == 0, then negative list
            else:
                neg_list = neg_list + ' '.join(line[2:])
    
    # Find the probability
    pos_pro_dic = probability(pos_list)
    neg_pro_dic = probability(neg_list)
    
    # Create the decision list
    decision_list_creat(pos_pro_dic,neg_pro_dic)
    
    # Sort the decision list
    decision_list.sort(key = lambda x: x[1], reverse = True)
    print("The length of dicision list: ", len(decision_list))        
    
    #Write the decision list
    with open('sentiment-decision-list.txt', 'w') as file:
    #loop through each item of decision_list1
        for items in decision_list:
        #Write each items inside of file
            #file.write('%s\n' % items)
            file.write('{}\t{}\t{}\n'.format(items[0],items[1],items[2]))
