"""
Test program

"""

# Import Libraries
import random
import re

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

# This is the not_handler feature
def not_handler(string):
    
   transformed = re.sub(r"\b(?:not|don't|no)\b[\w\s]+[^\w\s]", 
       lambda match: re.sub(r"(\s+)(\w+)", r"\1not_\2", match.group(0)), 
       string, flags=re.IGNORECASE)

   return transformed


# Predict the class of a review
def find_class(full_list):
    
    class_result = ''
    with open('sentiment-decision-list.txt','r') as input_file:
        for decision in input_file:
            # if the decision list token matches with test_review token then assign the corresponding class of that decision list token
            # to the review class.
            match = (decision.split('\t')[0])
            for i in full_list:
                if match == i:
                    class_result = decision.split('\t')[2]
                    break
            # This else statement is to break the outer for-loop (if a immediate match found)
            else:
                continue
            break    
        
        # if no token mathces, then assign a random class 
        if len(class_result) == 0:
            class_result = random.randint(0, 1)            
 
    return class_result
                          
        
# This Function creates a Unigram dictionary with respective frequencies    
def unigram(line):
    uni_list = []
    for i in range (len(line.split())):      
        uni_list.append(line.split()[i])
          
    return uni_list

# This Function creates a Bigram dictionary with respective frequencies
def bigram(line):
    bi_list = []
    for i in range(len(line.split())-1):
        temp = line.split()[i:i+2]
        temp = str(temp[0]) + ' ' + str(temp[1])
        bi_list.append(temp)
        
    return bi_list


if __name__=="__main__":
    
# Open the test data file, lowercase it, expant the short words, use not_handler function    
    with open('sentiment-test.txt','r') as input_file:
        with open('sentiment-system-answers.txt', 'w') as file:
            for review in input_file:
                review = review.lower()
                review = decontracted(review)
                review = not_handler(review)
                #create unigram and bigram list
                uni_list = unigram(review)
                bi_list = bigram(review)
                full_list = uni_list + bi_list
                review_id = full_list[0]
                class_result = find_class(full_list)
                #write the review_id and class            
                file.write('{} {}'.format(review_id,class_result))
            
            
            
        
    
            
      
    
   
           