import pdb
import math
import time
import sys
import re
import string

arg_length = len(sys.argv)
if(arg_length != 3):
    print("Error - wrong usage! Proper usage: python NaiveBayesClassifier.py <training_set>.txt <testing_set>.txt")
    sys.exit()

training_file = sys.argv[1]
testing_file = sys.argv[2]

# TRAIN
train_start = time.clock()
review_word_count = [[], []]
# pdb.set_trace()
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
negatives = dict(zip(["shouldn't", "not", "can't", "cannot", "won't", "isn't"], [1, 2, 3, 4, 5, 6]))
stopwords = dict(zip(["a", "about", "after", "again", "?", "." , ",", "/", "!" , "the" , "case", "ago", "he",
 "she", "all"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))
reviews = open(training_file, "r")
movie_review = reviews.readline()
movie_review = movie_review.rstrip()
while movie_review != '':
    cat = int(movie_review[-1])
    movie_review = movie_review.lower()
    tokenized_review = movie_review.split()
    review_word_count[cat].append({})
    length = len(tokenized_review)
    for i in range(0, len(tokenized_review)-1):
        word_current = tokenized_review[i]
        # pdb.set_trace()
        if word_current in stopwords:
            word_current = " "

        if (i - 1 <= 0):
            continue
        tokenized_review_word_count = tokenized_review.count(word_current)

        if (i-1 > 0 and tokenized_review[i-1] in negatives):
            word_current = tokenized_review[i-1] + " " + tokenized_review[i]
            if word_current not in review_word_count[cat][-1].keys():
                review_word_count[cat][-1][word_current] = tokenized_review_word_count
            elif word_current in review_word_count[cat][-1].keys():
                review_word_count[cat][-1][word_current] += tokenized_review_word_count
        elif (i-1 > 0 and tokenized_review[i-1] not in negatives):
            if word_current not in review_word_count[cat][-1].keys():
                review_word_count[cat][-1][word_current] = tokenized_review_word_count
            elif word_current in review_word_count[cat][-1].keys():
                review_word_count[cat][-1][word_current] += tokenized_review_word_count


    movie_review = reviews.readline()
    movie_review = movie_review.rstrip()

reviews.close()

# Create model
numPosReviews = len(review_word_count[1])
numNegReviews = len(review_word_count[0])
total_num_reviews = float(numPosReviews + numNegReviews)
probability_is_positive_review = float(numPosReviews) / total_num_reviews
probability_is_negative_review = float(numNegReviews) / total_num_reviews

neg_pos_obj = {
            "neg" : {},
            "pos" : {}
        }
sum_of_words = word_probabilites = neg_pos_obj
for pos_words_in_review, neg_words_in_review in zip(review_word_count[1], review_word_count[0]):
    for word in neg_words_in_review:
        num_neg = neg_words_in_review[word]
        if word not in sum_of_words["neg"]:
            sum_of_words["neg"][word] = num_neg
        elif word in sum_of_words["neg"]:
            sum_of_words["neg"][word] += num_neg
    for word in pos_words_in_review:
        num_pos = pos_words_in_review[word]
        if word not in sum_of_words["pos"]:
            sum_of_words["pos"][word] = num_pos
        elif word in sum_of_words["pos"]:
            sum_of_words["pos"][word] += num_pos

for word in sum_of_words["pos"]:
    count_neg = 0
    if word not in sum_of_words["neg"]:
        justtestingsometihng = 1
    else:
        count_neg = sum_of_words["neg"][word]
    total = sum_of_words["pos"][word] + count_neg
    if (sum_of_words["pos"][word] + count_neg) > 0:
        word_probabilites["pos"][word] = float(sum_of_words["pos"][word]) / total
        word_probabilites["neg"][word] = float(count_neg) / total

for word in sum_of_words["neg"]:
    pos_num = 1
    neg_num = 0
    if word in word_probabilites["pos"] == False:
        word_probabilites["pos"][word] = neg_num
        word_probabilites["neg"][word] = pos_num

train_end = time.clock()
time_to_train = int(train_end - train_start)
time_to_train = str(time_to_train) + " seconds (training)"
# END OF TRAINING
################################################################################

# CLASSIFY:
classify_start = time.clock()
reviews = open(testing_file, "r")
list_of_words = []
movie_review = reviews.readline()
movie_review = movie_review.rstrip()
while movie_review != '':
    cat = int(movie_review[-1])
    movie_review = movie_review.lower()
    review_words = movie_review.split()
    true_cat_word_obj = {
        "true_category" : cat,
        "words" : review_words
    }
    list_of_words.append(true_cat_word_obj)
    movie_review = reviews.readline()
    movie_review = movie_review.rstrip()
reviews.close()

# CLASSIFY REVIEWS
classified_reviews = []
for review_meta_data in list_of_words:
    negative_probability = math.log10(probability_is_negative_review)
    positive_probability = math.log10(probability_is_positive_review)

    for word in review_meta_data["words"][1:]:
        if word in word_probabilites["pos"]:
            if word_probabilites["pos"][word] > 0:
                positive_probability += math.log10(word_probabilites["pos"][word])
        if word in word_probabilites["neg"]:
            if word_probabilites["neg"][word] > 0:
                negative_probability += math.log10(word_probabilites["neg"][word])

    calc_cat = None
    if positive_probability > negative_probability:
        calc_cat = 1
    else:
        calc_cat = 0
    curr_true = int(review_meta_data["true_category"])
    cat_metadata = {
        "true_category" : curr_true,
        "estimated_cat" : calc_cat
    }
    classified_reviews.append(cat_metadata)

# GET CLASSIFICATION ACCURACY AND PRINT CALCULATED CATEGORY
num_correct = 0
num_incorrect = 0
count = 0
for classified_review in classified_reviews:
    estimated_cat = classified_review["estimated_cat"]
    true_cat = classified_review["true_category"]
    if true_cat != estimated_cat:
        num_incorrect += 1
    else:
        num_correct += 1
    print estimated_cat
    count += 1

classification_accuracy_str = str(float(num_correct) / float(count)) + " (testing)"

# RUN CLASSIFICATION ON TRAINING
reviews = open(training_file, "r")
list_of_words = []
movie_review = reviews.readline()
movie_review = movie_review.rstrip()
while movie_review != '':
    cat = int(movie_review[-1])
    movie_review = movie_review.lower()
    review_words = movie_review.split()
    true_cat_word_obj = {
        "true_category" : cat,
        "words" : review_words
    }
    list_of_words.append(true_cat_word_obj)
    movie_review = reviews.readline()
    movie_review = movie_review.rstrip()
reviews.close()
reviewsClassified = []
for review_metadata in list_of_words:
    negative_probability = math.log10(probability_is_negative_review)
    positive_probability = math.log10(probability_is_positive_review)
    for word in review_metadata["words"][1:]:
        if word in word_probabilites["pos"]:
            if word_probabilites["pos"][word] > 0:
                positive_probability += math.log10(word_probabilites["pos"][word])
        if word in word_probabilites["neg"]:
            if word_probabilites["neg"][word] > 0:
                negative_probability += math.log10(word_probabilites["neg"][word])
    est_cat = 0
    if(negative_probability > positive_probability):
        est_cat = 0
    else:
        est_cat = 1
    curr_true = int(review_metadata["true_category"])
    true_est_cat_obj = {
        "true_category" : curr_true,
        "estimated_cat" : est_cat
    }
    reviewsClassified.append(true_est_cat_obj)

# Get training accuracy
correctNum = 0
incorrNum = 0
count = 0
for classified_review in reviewsClassified:
    true_cat = classified_review["true_category"]
    est_cat = classified_review["estimated_cat"]
    if true_cat != est_cat:
        incorrNum += 1
    else:
        correctNum += 1
    count += 1

training_accuracy_str = str(float(correctNum) / float(count)) + " (training)"
classify_end = time.clock()
time_to_label = int(classify_end - classify_start)
time_to_label = str(time_to_label) + " seconds (labeling)"
# Print final results`
print(time_to_train)
print(time_to_label)
print(training_accuracy_str)
print(classification_accuracy_str)
