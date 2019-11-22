import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords    
import string
from collections import Counter

with open('C:/Users/Kazi Zainab Khanam/Documents/Tweets_Cleaned_text.txt', 'r',encoding="utf16") as shakes:
    text = shakes.read()
    tweets = sent_tokenize(text) 
    total_documents = len(tweets)
   # print(tweets)
def _create_frequency_matrix(tweets): # the frequency of words in each tweet
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in tweets:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix
def _create_tf_matrix(freq_matrix):# the tf(t) is calculated by the number of times term t appears in a document / total number of terms in a document
    tf_matrix = {}                # so document is the tweet and and the term is the word in the tweet

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_tweet = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_tweet

        tf_matrix[sent] = tf_table

    return tf_matrix
def _create_documents_per_words(freq_matrix): # how many tweets contain the same word as tweet is a document so this returns a table of tweets per words
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):# calculates IDF(t) = log_e(Total number of documents / Number of documents with term t in it) so document is a tweet and term is the word in the tweet
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):# creating the tf-idf matrix by multiplying the tf_matrix and idf_matrix
    tf_idf_matrix = {}                           # produces the tf-idf score of each of the words in a sentence

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _score_tweets(tf_idf_matrix) -> dict: # score a tweet by its word's TF Basic algorithm: adding the TF frequency of every non-stop word in a tweet divided by total no of words in a tweet

    tweetValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_tweet = 0

        count_words_in_tweet = len(f_table)
        for word, score in f_table.items():
            total_score_per_tweet += score

        tweetValue[sent] = total_score_per_tweet / count_words_in_tweet

    return tweetValue
def _find_average_score(tweetValue) -> int:#Finding the average score from the tweet value dictionary by calculating the average tweet score
    tweetValues = 0
    for entry in tweetValue:
        tweetValues += tweetValue[entry]

    # Average value of a tweet from original tweets_basedonScore_text
    average = (tweetValues / len(tweetValue))

    return average
def _generate_tweets_basedonScore(tweets, tweetValue, threshold):
    tweet_count = 0
    tweets_basedonScore = ''

    for tweet in tweets:
        if tweet[:15] in tweetValue and tweetValue[tweet[:15]] >= (threshold):
            tweets_basedonScore += " " + tweet
            tweet_count += 1

    return tweets_basedonScore
# 2 Create the Frequency matrix of the words in each tweet.
freq_matrix = _create_frequency_matrix(tweets)
print(freq_matrix)

# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
#print(tf_matrix)

# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
#print(count_doc_per_words)

'''
Inverse document frequency (IDF) is how unique or rare a word is.
'''
# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
#print(idf_matrix)

# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
print(tf_idf_matrix)
# scoring the tweets by using the tf-idf matrix
tweet_scores = _score_tweets(tf_idf_matrix)
#print(tweet_scores)

# 8 Find the threshold
threshold = _find_average_score(tweet_scores)
#print(threshold)

# 9 Important Algorithm: Generate the tweets_basedonScore
tweets_basedonScore = _generate_tweets_basedonScore(tweets, tweet_scores, 1.3 * threshold)
print(tweets_basedonScore)