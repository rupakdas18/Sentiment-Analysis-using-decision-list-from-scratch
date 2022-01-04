# Sentiment-Analysis-using-decision-list-from-scratch

**Problem Description:**
This is a sentiment classifier trained for classifying movie reviews. The training dataset contains 1186 reviews
(592 negative and 594 positives). The test dataset contains 200 reviews (100 positives and 100 negative but unmarked).
The gold dataset contains the actual classes of those reviews (with a unique review id). The objective is to use a decision
list tree to classify those test reviews using unigram, bigram, and not handling features and find the accuracy, precision, and recall.

**Example of Input and output:**
As the name of the input and output files are fixed, I didn't use any command line. But the input and output should be like below.
program_file_name input_file --> output_file
decision-list-train.py sentiment-train.txt > sentiment-decision-list.txt
decision-list-test.py sentiment-decision-list.txt sentiment-test.txt > sentiment-system-answers.txt
decision-list-eval.py sentiment-gold.txt sentiment-system-answers.txt > sentiment-system-answers-scored.txt

**Algorithm:**
1. Train the model and get the decision list
  -For every review
    - lowercase it
    - convert the short words into expanded form
    - use Not_nadler function
    - find out the class. If it's positive, store it in pos_list, if negative then in neg_list
  - create the unigram, bigram dictionary with corresponding frequencies of both classes
  - Create the probability dictionary of both classes
  - find the log-likelihood of each item
    - if log-likelihood is positive, mark the word as positive
    - if log-likelihood is negative, mark the word as negative
    - else ignore (log-likelihood value of zero doesn't have any impact)
  - Create decision list (format: ngram (tablespace) absolute_value_of_log (tablespace) class) for each classes
  - sort the decision list based on log-likelihood value and write it
  
2. Test. Predict the class with the help of a decision list
  - For every test review
    - lowercase it
    - convert the short words into expanded form
    - use Not_nadler function
    - create uni_gram and bi_gram list
    - For every item in the decision list
      - compare it with each item of the uni_gram and bi_gram list. if it finds a match, it returns the corresponding class.
        if not then randomly select a value from 0 and 1 and return it
 - Create the predicted list and write it in a file
 
3. Evaluate the result
  - open actual and predicted files and store the data in a dictionary (key = review_id and value = class)
  - for each item of 2 dictionaries compare it
    - if actual = + , predict = +, then true positive
    - if actual = + , predict = 1, then false negative
    - if actual = - , predict = +, then false positive
    - if actual = - , predict = -, then true negative
  - calculate, accuracy, precision and recall and write it
