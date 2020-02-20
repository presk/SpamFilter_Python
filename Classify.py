import re
# import numpy
import math
# import sys
# import mathplotlib
import os


class Classifier():

    def __init__(self):
        self.word_count = {}
        self.spam_word_count = {}
        self.ham_word_count = {}
        self.spam_words = 0
        self.ham_words = 0
        self.stop_words = []
        self.ham_prob = {}
        self.spam_prob = {}
        self.spam_file = 0
        self.ham_file = 0

        with open("english-stopwords.txt", "r", encoding="latin-1") as f:
            for line in f:
                for word in re.split("[^a-zA-Z]", line.lower()):
                    if not word:
                        continue
                    self.stop_words.append(word)

    def train(self, dir_path, mode=None):
        for filename in os.listdir(dir_path):
            if "spam" in filename:
                self.spam_file += 1
            else:
                self.ham_file += 1

            with open(dir_path + filename, "r", encoding="latin-1") as f:
                for line in f:
                    for word in re.split("[^a-zA-Z]", line.lower()):
                        if not word:
                            continue

                        if mode == "stopword":
                            if word in self.stop_words:
                                continue
                        elif mode == "length":
                            if len(word) > 9 or len(word) < 2:
                                continue
                        self.word_count[word] = self.word_count.get(word, 0) + 1
                        if "spam" in filename:
                            self.spam_word_count[word] = self.spam_word_count.get(word, 0) + 1
                            self.spam_words += 1
                        else:
                            self.ham_word_count[word] = self.ham_word_count.get(word, 0) + 1
                            self.ham_words += 1

        for k in self.word_count:
            self.spam_prob[k] = (self.spam_word_count.get(k, 0) + 0.5) / (self.spam_words + 0.5 * len(self.word_count))
            self.ham_prob[k] = (self.ham_word_count.get(k, 0) + 0.5) / (self.ham_words + 0.5 * len(self.word_count))

    def print_model(self, filename):
        with open(filename, "w", encoding="latin-1") as f:
            for i, k in enumerate(sorted(self.word_count)):
                f.write("{}  {}  {}  {}  {}  {}\r\n".format(i, k, self.ham_word_count.get(k, 0)
                                                            , self.ham_prob[k], self.spam_word_count.get(k, 0),
                                                            self.spam_prob[k]))

    def reset(self):
        self.word_count = {}
        self.spam_word_count = {}
        self.ham_word_count = {}
        self.spam_words = 0
        self.ham_words = 0
        self.ham_prob = {}
        self.spam_prob = {}
        self.spam_file = 0
        self.ham_file = 0

    def test(self, file):
        total_files = self.spam_file + self.ham_file

        spam = math.log10(self.spam_file / total_files)
        ham = math.log10(self.ham_file / total_files)

        with open(file, "r", encoding="latin-1") as f:
            for line in f:
                for word in re.split("[^a-zA-Z]", line.lower()):
                    if not word:
                        continue
                    # math.log10(1) = 0 <- ignore words not in dictionary
                    spam += math.log10(self.spam_prob.get(word, 1))
                    ham += math.log10(self.ham_prob.get(word, 1))

        if ham > spam:
            return ("ham", ham, spam)
        else:
            return ("spam", ham, spam)


def test_batch(classifer, dir_path, output_file):
    matrix = [0, 0, 0, 0]  # [Correct Ham, Incorrect Ham, Correct Spam, Incorrect Ham]
    with open(output_file, "w", encoding="latin-1") as f:
        for index, filename in enumerate(sorted(os.listdir(dir_path))):
            res = classifer.test(dir_path + filename)
            if "ham" in filename:
                correct = "ham"
                if res[0] == correct:
                    matrix[0] += 1
                else:
                    matrix[1] += 1
            else:
                correct = "spam"
                if res[0] == correct:
                    matrix[3] += 1
                else:
                    matrix[2] += 1
            verdict = "right" if correct == res[0] else "wrong"
            f.write("{}  {}  {}  {}  {}  {}  {}\r\n".format(index, filename, res[0], res[1], res[2], correct, verdict))

    beta = math.pow(1, 2)
    total = sum(matrix)
    acc = sum([matrix[0], matrix[3]]) / sum(matrix)
    p_ham = matrix[0] / sum([matrix[0], matrix[1]])
    p_spam = matrix[3] / sum([matrix[3], matrix[2]])
    r_ham = matrix[0] / sum([matrix[0], matrix[2]])
    r_spam = matrix[3] / sum([matrix[3], matrix[1]])
    f_ham = ((beta + 1) * p_ham * r_ham) / (beta * p_ham + r_ham)
    f_spam = ((beta + 1) * p_spam * r_spam) / (beta * p_spam + r_spam)
    print(matrix)
    print("Accuracy - {}".format(acc))
    print("Precision of ham - {}".format(p_ham))
    print("Precision of spam - {}".format(p_spam))
    print("Recall of ham - {}".format(r_ham))
    print("Recall of spam -  {}".format(r_spam))
    print("F1-Measure of ham - {}".format(f_ham))
    print("F1-Measure of spam - {}".format(f_spam))


c = Classifier()

# ========
# Baseline
# ========
print("===== Baseline ====")
c.train("./train/")
c.print_model("model.txt")
test_batch(c, "./test/", "baseline-result.txt")
c.reset()

# ===============
# Stopword Filter
# ===============
print("===== Stopword ====")
c.train("./train/", "stopword")
c.print_model("stopword-model.txt")
test_batch(c, "./test/", "stopword-result.txt")
c.reset()

# =============
# Length Filter
# =============
print("===== Word Length ====")
c.train("./train/", "length")
c.print_model("wordlength-model.txt")
test_batch(c, "./test/", "wordlength-result.txt")
c.reset()
