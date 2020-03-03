import numpy as np

class Review:

    def __init__(self, bagOfWords, score):
        self._setOfWords = set(bagOfWords)
        self._bagOfWords = bagOfWords
        self._score = score
        self._tf = {}
        self._class = 0

        for word in self._bagOfWords:
            if word in self._tf:
                self._tf[word] = self._tf[word] + 1
            else :
                self._tf[word] = 1

        tmp = score * 10
        self._class = int(tmp)

    def GetScore(self):
        return self._score

    def GetClass(self):
        return self._class

    def HasWord(self, word):
        return word in self._bagOfWords

    def GetBagOfWords(self):
        return self._bagOfWords

    def GetSetOfWords(self):
        return self._setOfWords

    def GetTf(self, word):
        return self._tf[word]

    def PrintTF(self):
        print(self._tf)