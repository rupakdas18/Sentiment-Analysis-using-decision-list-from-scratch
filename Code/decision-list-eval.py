"""
Evaluation Program
"""

actual_dic = {}
predict_dic = {}

tp = 0
fp = 0
fn = 0
tn = 0

# Make a dictionary with actual class
with open('sentiment-gold.txt','r') as actual_class:
    for value in actual_class: 
        actual_dic[value.split()[0]] = value.split()[1]
        
# Make a dictionary with predicted class       
with open('sentiment-system-answers.txt','r') as predict_class:
    for value in predict_class: 
        predict_dic[value.split()[0]] = value.split()[1]   


# calculate True positive, True Negative, False positive, False negative as described in the algorithm
for value in actual_dic:
    if actual_dic.get(value) == '1' and predict_dic.get(value) == '1' :
        tp = tp + 1                       
    elif actual_dic.get(value) == '0' and predict_dic.get(value) == '1':
        fp = fp + 1
    elif actual_dic.get(value) == '1' and predict_dic.get(value) == '0':
        fn = fn + 1
    elif actual_dic.get(value) == '0' and predict_dic.get(value) == '0':
        tn = tn + 1

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = (tp)/(tp+fp)
recall = tp/(tp+fn)

print("Accuracy: {} %".format(round(accuracy*100,4)))
print("Precision: {} %".format(round(precision*100,4)))
print("Recall: {} %".format(round(recall*100,4)))

with open('sentiment-system-answers-scored.txt', 'w') as file:
    file.write("Accuracy: {} %\n".format(round(accuracy*100,4)))
    file.write("Precision: {} %\n".format(round(precision*100,4)))
    file.write("Recall: {} %".format(round(recall*100,4)))


