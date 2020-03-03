import numpy as np
from Review import *
from queue import *

class Collection:

    def __init__(self):
        self.listOfReviews = []
        self.nbOfReviews = 0
        self.dft = {}
        self.tfidf = {}
        self.relevantWordsAndScores =[]
        self.relWords = []
        self.totalNbOfWord = 0
        self.sortedTFIDF = []

    def AddReview(self, review):
        self.listOfReviews.append(review)
        self.nbOfReviews += 1

    def GetListOfReviews(self):
        return self.listOfReviews

    def SetDFT(self):
        for review in self.listOfReviews:
            for word in review._setOfWords:
                if word in self.dft:
                    self.dft[word] = self.dft[word] + 1
                else:
                    self.dft[word] = 1
        return self.dft

    def SetTFIDF(self):

        if len(self.dft) == 0:
            print("dft is empty")
            return self.dft

        self.tfidf = self.tfidf.fromkeys(self.dft)

        # initialise tfidf
        for word in self.tfidf:
            self.tfidf[word] = 0

        # For each review, we take each of its words and compute its tfidf
        # and add it to the current tfidf dictionary value associated with.
        # At the end, we divide the total by their dft
        # to have an average of their tfidf score when they appear in a document

        for review in self.listOfReviews:
            words = review.GetSetOfWords()
            for word in words:
                self.tfidf[word] = self.tfidf[word] \
                                   + (ComputeTfIdf( review,word,self.dft,
                                                  self.nbOfReviews)\
                                   * review.GetScore())
            # NOT SURE !! Weight the score through the utility score from review
        for word in self.tfidf:
            self.tfidf[word] = self.tfidf[word] / self.dft[word]
        return self.tfidf

    def SetRelevantWords(self, proportion):

        if proportion > 1 and proportion < 0:
            print("proportion should be a percentage")
            return 0

        priorityTFIDF = PriorityQueue()

        for word in self.tfidf:

            score = self.tfidf[word]
            priorityTFIDF.put((score, word))
            self.totalNbOfWord = self.totalNbOfWord + 1

        # we define the relevant terms as the upper proportion of the words
        nbOfRelWords = int(self.totalNbOfWord * proportion)

        # sorting is done, so we trade the queue for a list
        while not priorityTFIDF.empty():
            self.sortedTFIDF.append(priorityTFIDF.get())

        # we save the value of the treshold
        treshold = (self.sortedTFIDF[self.totalNbOfWord - nbOfRelWords])[0]
        print("treshold value is : {}".format(treshold))

        for word in self.sortedTFIDF:
            score = word[0]
            if score >= treshold:
                # we create a list of the words and their scores
                self.relevantWordsAndScores.append(word)
                # we create a list of only the words
                self.relWords.append(word[1])

        return self.relevantWordsAndScores




def ComputeTfIdf(review, word, df, N):
    """

    :param review: a review object
    :param word: a word of which we want to compute the tfidf score in this review
    :param df: a dictionary containing all the words and their corresponding
                number of document
    :param N: total number of documents
    :return: the tfidf score of "word" in "review"
    """
    tf = review.GetTf(word)
    dft = df[word]
    tfidf = tf * np.log(N / dft)

    return tfidf





