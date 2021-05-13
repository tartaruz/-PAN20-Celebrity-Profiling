import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import f1_score, precision_score, recall_score
import sklearn
import pickle
import math

# God BLESS
# https://scikit-learn.org/0.15/modules/scaling_strategies.html


class Classifiers:
    def __init__(self, retrival, vectorizer, config):
        self.retrival = retrival
        self.retrival.get_training_labels()
        self.vectorizer = vectorizer
        self.GENDER_classifiers = [
            Perceptron(alpha=0.01, verbose=0),
            SGDClassifier(),
            PassiveAggressiveClassifier(),
            MultinomialNB(),
            BernoulliNB()
        ]
        self.OCCUPATION_classifiers = [
            Perceptron(alpha=0.01, verbose=0),
            SGDClassifier(),
            PassiveAggressiveClassifier(),
            MultinomialNB(),
            BernoulliNB()
        ]
        self.YEAR_classifiers = [
            Perceptron(alpha=0.01, verbose=0),
            SGDClassifier(),
            PassiveAggressiveClassifier(),
            MultinomialNB(),
            BernoulliNB()
        ]
        self.percent = 10 / 100
        self.steps = 0

    def train(self):
        self.terminal_info("Training All Classifiers")

        for index_row, row in self.retrival.current_celebrity.iterrows():
            if (index_row) % int(int(self.retrival.size)*self.percent*1.5) == 0:
                self.terminal_info("Status", percent=str(
                    int(index_row*100/int(self.retrival.size))))
            feed = [element for element in row[1] if len(element) > 0]
            flat_feed = [item for sublist in feed for item in sublist]

            x = self.vectorizer.transform([" ".join(flat_feed)]).toarray()
            y = [
                self.retrival.current_labels.loc[self.retrival.current_labels['id'] == row.id]]

            for classifier in self.OCCUPATION_classifiers:
                y_occupation = [y[0]["occupation"].to_string(index=False)]
                classifier.partial_fit(x, y_occupation, classes=[
                                       "sports", "performer", "creator", "politics"])

            for classifier in self.GENDER_classifiers:
                y_gender = [y[0]["gender"].to_string(index=False)]
                classifier.partial_fit(x, y_gender, classes=["male", "female"])

            for classifier in self.YEAR_classifiers:
                y_year = [y[0]["birthyear"].to_string(index=False)]
                y_year = self.dicrete_year(y_year)
                classifier.partial_fit(x, y_year, classes=[
                                       "4", "5", "6", "7", "8", "9"])

    def dicrete_year(self, year):
        return [year[0][2]]

    def predict(self, test_df):
        self.terminal_info("Predict With all classifiers")

        results_gender = [[] for _ in range(len(self.GENDER_classifiers))]
        results_occupation = [[] for _ in range(len(self.OCCUPATION_classifiers))]
        results_year = [[] for _ in range(len(self.YEAR_classifiers))]
        
        
        for index_row, row in test_df.current_celebrity.iterrows():
            feed = [element for element in row[1] if len(element) > 0]
            flat_feed = [item for sublist in feed for item in sublist]

            x = self.vectorizer.transform([" ".join(flat_feed)]).toarray()
            y = [test_df.current_labels.loc[test_df.current_labels['id'] == row.id]]

            for index_occ in range(len(self.OCCUPATION_classifiers)):
                y_occupation = [y[0]["occupation"].to_string(index=False)]
                prediction = self.OCCUPATION_classifiers[index_occ].predict(x)
                results_occupation[index_occ].append(prediction[0])

            for index_gender in range(len(self.GENDER_classifiers)):
                y_gender = [y[0]["gender"].to_string(index=False)]
                prediction = self.GENDER_classifiers[index_gender].predict(x)
                results_gender[index_gender].append(prediction[0])


            for index_year in range(len(self.YEAR_classifiers)):
                y_year = [y[0]["birthyear"].to_string(index=False)]
                y_year = self.dicrete_year(y_year)
                prediction = self.YEAR_classifiers[index_year].predict(x)
                results_year[index_year].append(prediction[0])

        return results_occupation, results_gender, results_year

    def get_labels(self,test_df):
        self.terminal_info("Retriving ")
        labels_gender = [[] for _ in range(len(self.GENDER_classifiers))]
        labels_occupation = [[] for _ in range(len(self.OCCUPATION_classifiers))]
        labels_year = [[] for _ in range(len(self.YEAR_classifiers))]
        
        
        for index_row, row in test_df.current_celebrity.iterrows():
            y = [test_df.current_labels.loc[test_df.current_labels['id'] == row.id]]

            for index_occ in range(len(self.OCCUPATION_classifiers)):
                y_occupation = [y[0]["occupation"].to_string(index=False)]
                labels_occupation[index_occ].append(y_occupation[0])

            for index_gender in range(len(self.GENDER_classifiers)):
                y_gender = [y[0]["gender"].to_string(index=False)]
                labels_gender[index_gender].append(y_gender[0])


            for index_year in range(len(self.YEAR_classifiers)):
                y_year = [y[0]["birthyear"].to_string(index=False)]
                y_year = self.dicrete_year(y_year)
                labels_year[index_year].append(y_year[0])

        return labels_occupation, labels_gender, labels_year

    def evaluate_all(self, test_df):
        y_occupation, y_gender, y_year = self.get_labels(test_df)
        results_occupation, results_gender, results_year = self.predict(test_df)
        
        names = ["Perceptron", "SGDClassifier", "PassiveAggressive", "MultinomialNB","BernoulliNB"]
        avr = "macro"
        for index, predicitons in enumerate(results_occupation):
            print(f"[Occupation]\t[{names[index]}]")
            res = f1_score(predicitons, y_occupation[index], average=avr)
            res_precition = precision_score(predicitons, y_occupation[index], average=avr)
            res_recall = recall_score(predicitons, y_occupation[index], average=avr)
            print(f"F1:\t\t{res}\nPrecision:\t{res_precition}\nRecall:\t\t{res_recall}\n")

        for index, predicitons in enumerate(results_gender):
            print(f"[Gender]    \t[{names[index]}]")
            res = f1_score(predicitons, y_gender[index], average=avr)
            res_precition = precision_score(predicitons, y_gender[index], average=avr)
            res_recall = recall_score(predicitons, y_gender[index], average=avr)
            print(f"F1:\t\t{res}\nPrecision:\t{res_precition}\nRecall:\t\t{res_recall}\n")

        for index, predicitons in enumerate(results_year):
            print(f"[Year]      \t[{names[index]}]")
            # print(predicitons)
            # print( y_year[index])
            res = f1_score(predicitons, y_year[index], average=avr)
            res_precition = precision_score(predicitons, y_year[index], average=avr)
            res_recall = recall_score(predicitons, y_year[index], average=avr)
            print(f"F1:\t\t{res}\nPrecision:\t{res_precition}\nRecall:\t\t{res_recall}\n")


    def terminal_info(self, info, percent=None):
        if not percent:
            print(f"\n[ Step: {self.steps} ]\tInitialization [{str(info)}]")
            self.steps += 1
        else:
            print(f"     |--------> ¦{str(info)}¦\t{percent}%")





