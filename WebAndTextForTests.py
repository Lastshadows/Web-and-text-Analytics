import numpy
from queue import *
from Review import *
import matplotlib.pyplot as plt
from Collection import *
from sklearn import datasets, linear_model
import array
from sklearn.metrics import mean_squared_error, r2_score



def ExtractHelpful(line):
    """

    :param line:  a string containing a complete review line
    :return: the 2 'helpful' scores as a pair of strings
    """
    # looking for the brackets around the scores
    yIndex = line.find("]") - 1
    xIndex = line.find("[") + 1

    # we isolate in a pair of strings the both part of the review score [x,y]
    scorePart = line[xIndex:yIndex + 1]
    xy = scorePart.split(", ")
    return xy

def ExtractReview(line):
    """
    :param line: the whole line of the amazon review
    :return: a list of all the words of the actual review
    """

    s = "reviewtext"
    s2 = "overall"
    start = line.find(s) + 11
    end = line.find(s2) - 3
    pureReview = line[start:end]
    review = pureReview.split(" ")
    return review


def CleanStopWord(listOfWords, stopWords):
    """
    takes a list of words and removes all stop words specified in a list
    defined in this function.
    :param listOfWords: the list to clean
    :stopWords : list of stop words that are not interesting
    :return: the cleaned list
    """

    cleanedListOfWords =[]

    for words in listOfWords:
        if words not in stopWords:
            cleanedListOfWords.append(words)

    return cleanedListOfWords


def RemovePunct(listOfWord, stopWords):
    """

    :param listOfWord: List of word to be cleaned from punctuation
    :param stopWords: list of stop words that are not interesting
    :return: a list of words, lower case, without any punctuation
    """

    cleanedListOfWords = []
    punctuationSymbols =['!', ',', '.', '?', "'", '(', ')']


    for word in listOfWord:
        #if a word is not entirely alphabetical, we have a look
        if (word.isalpha()) != True:
            for i, charact in enumerate (word) :
                if charact in punctuationSymbols:
                    if i==0 :
                        cleanedListOfWords.append(word[1:])
                    if i==len(word) -1:
                        cleanedListOfWords.append(word[0:len(word)-1])
                    else:
                        if word[0:i] not in stopWords:
                            cleanedListOfWords.append(word[0:i])
                        word = word[i+1:]
                        if word not in stopWords:
                            cleanedListOfWords.append(word)

        else:
            cleanedListOfWords.append(word)

    return cleanedListOfWords


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
    tfidf = tf * np.log(N/dft)

    return tfidf


def main():

    # ---------------------------------------------------------------
    #
    #                          Cleaning
    #
    # ---------------------------------------------------------------

    # opening the file and reading it
    f = open("data.txt", "r")
    contents = f.readlines()
    nbOfDoc = 0
    stopWords = ('i', 'the', 'a', 'an', 'to', 'it', 'as', 'and', 'is', 'does',
                 'not', 'was', 'so', 'than', 'of', 'for', 'my', 'you', 'we',
                 'they', 'this', 'that', 'with', 'are', 'were', 'your', 'their',
                 'no', 'yes', 'or', 'them', 'did', 'had', 'will', 'may', 'mine',
                 '', 's', 've', 'd', 'can', 'on', 'up', 'down', 'but', 'or',
                 'me','out', ',', 'if', 'by', "don't", "i've",'re', 'be', 'in',
                 'd', 'have', 'all','got', 'go', 'much', '.', 'on', 'one',
                 'should', 'have','these' )

    collection = Collection()
    # reading all lines one by one
    for line in contents:

        # xy contains the 2 helpfulness scores
        xy = ExtractHelpful(line)

        # if not useless, we look at the line
        if xy[1] != "0":
            score = int(xy[0]) / int(xy[1])

            # review is a list of the strings contained in the original review
            # but the strings are only separated according to white spaces
            # in the original review (see split function)
            # therefore we need to clean the strings of the punctuations symbols
            # and of the stop words
            rawReview = ExtractReview(line.lower())
            cleanedReview = CleanStopWord(rawReview, stopWords)
            cleanedReview = RemovePunct(cleanedReview, stopWords)
            cleanedReview = RemovePunct(cleanedReview, stopWords)

            # collection is a list of perfectly cleaned
            # reviews and their scores.
            review = Review(cleanedReview, score)
            if review.nbOfWords is not 0 :
                collection.AddReview(review)

            nbOfDoc = nbOfDoc + 1

    # ---------------------------------------------------------------
    #
    #                   Feature Selection
    #
    # ---------------------------------------------------------------

    # we create a dictionary dft that will contain all the words encountered in
    # any document. The words will be the key and be paired with the number of
    # documents that contain them

    dft = collection.SetDFT()

    print("\n dft : ")
    print(dft)

    tfidf = collection.SetTFIDF()

    print("\n tfidf : ")
    print(tfidf)

    relWordsScores = collection.SetRelevantWords(0.5)

    print("\n relevant words and scores : " )
    print(relWordsScores)

    relWords = collection.relWords

    print("\n relevant words only : ")
    print(relWords)

    # we now have all our sorted relevant words stocked in relWords

    # ---------------------------------------------------------------
    #
    #                   Training & Predictions
    #
    # ---------------------------------------------------------------

    allReviews = collection.GetListOfReviews()
    trainingColl = Collection()
    testColl = Collection()
    i = 1

    # we create two  collections, one of training, one of test
    for review in allReviews:
        if i%10 == 0:
            testColl.AddReview(review)
        else:
            trainingColl.AddReview(review)
        i += 1

    print("\n test and train set are done")

    trainSorter = PriorityQueue()
    testSorter  = PriorityQueue()

    # xTrain will be used for the training of the regressions
    xTrain = np.ndarray((1, trainingColl.nbOfReviews))
    xTest = np.ndarray((1, testColl.nbOfReviews))

    # xTrainList and yTrain will be used for plots. yTrain will also be used for
    # training purposes
    xTrainList = []
    yTrain = []
    xTestList = []
    yTest = []

    # for every review we compute a score based on the sum of the tfidf scores
    # of the relevant words divided by the number of relevant words, and we
    # associate it with the relevance score of the review. We put this tuple in
    # the priority queue to sort them by review score

    for review in trainingColl.listOfReviews:
        nbOfRelWords = 0
        reviewClass = review.GetScore()
        reviewScore = 0
        for word in review.GetSetOfWords():
            if word in collection.relWords:
                nbOfRelWords += 1
                reviewScore +=  tfidf[word]
        # reviewScore /= review.nbOfWords
        if reviewScore  != 0:
            reviewScore /= nbOfRelWords
        trainSorter.put((reviewScore, reviewClass))

    print("all training reviews are treated")

    for review in testColl.listOfReviews:
        nbOfRelWords = 0
        reviewClass = review.GetScore()
        reviewScore = 0
        for word in review.GetSetOfWords():
            if word in collection.relWords:
                nbOfRelWords += 1
                reviewScore +=  tfidf[word]
        # reviewScore /= review.nbOfWords
        if reviewScore  != 0:
            reviewScore /= nbOfRelWords
        trainSorter.put((reviewScore, reviewClass))

    i = 0
    while not trainSorter.empty():

        info = trainSorter.get()
        xTrain[0][i] = info[0]
        xTrainList.append(info[0])
        yTrain.append(info[1])
        i+=1

    i = 0
    while not testSorter.empty():
        info = testSorter.get()
        xTest[0][i] = info[0]
        xTestList.append(info[0])
        yTest.append(info[1])
        i += 1

    print("x and y built")

    xTrain2 = np.reshape(xTrain, (-1, 1))
    xTest2  = np.reshape(xTest, (-1, 1))

    print("converted to arrays ! Starting linear model ")

    linModel = linear_model.LinearRegression()
    linModel.fit(xTrain2,yTrain)
    trainTest = linModel.predict(xTrain2)
    poly = np.polyfit(xTrainList, yTrain, 15)
    model = np.poly1d(poly)

    print("prediction made for training set")

    plt.plot(xTrainList, yTrain, '.', xTrainList, trainTest, xTrainList, model(xTrainList))
    plt.show()

    test = linModel.predict(xTest2)
    plt.plot(xTestList, yTest, '.', xTestList, test, xTestList,
             model(xTestList))
    plt.show()




if __name__ == "__main__":
    main()
