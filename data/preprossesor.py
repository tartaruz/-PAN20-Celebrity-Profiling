import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from pandas.core.common import flatten
import pickle
from nltk.stem import PorterStemmer
import os

# Handels the prporssesing of the Data. Need a Retrival object to handel to prevent copying the large amount of data
class Preprossesor:
    def __init__(self, retrival, config):
        self.retrival = retrival
        self.stopwords = {element: True for element in list(
            set(stopwords.words('english')))}
        self.steps = 0
        self.percent = 10 / 100

        save_file_name = self.filename_from_config(config)

        if config["save_mode"]:
            if self.check_for_load(config, save_file_name):
                print(
                    f"[\t\tFound 🥒 File\t\t]\t - Drop Pipeline for preprossesing'{save_file_name}'")
                self.load_retrival(config, save_file_name)

            else:
                print(f"[\t\tNo 🥒 File\t\t]\t - Starts pipeline for preprossesing")
                self.pipeline(config, save_file_name)

    #The prosseses the tweets go though
    def pipeline(self, config, save_file_name):

        if config["replacement_of_emoji"]:
            self.terminal_info("Replacement Of Emoji")
            self.replace_emoji()
        if config["replacement_of_link"]:
            self.terminal_info("Replacement Of URLs")
            self.replace_link()
        if config["removal_of_puntation"]:
            self.terminal_info("Removal of Puntation")
            self.remove_puntation()
        if config["removal_of_stopwords"]:
            self.terminal_info("Removal of Stopwords")
            self.remove_stopwords()
        if config["normalization_with_stemmer"]:
            self.terminal_info("Normalization(Stemming)")
            self.normalization()
        if config["save_mode"]:
            self.terminal_info(" SAVING ")
            self.save_retrival(config, save_file_name)

    #The filename of the  pickle file is created. It is a binary system that gives a 1 if the prosses was used or not.
    def filename_from_config(self, config):
        save_file_name = [0, 0, 0, 0, 0]
        if config["replacement_of_emoji"]:
            save_file_name[0] = 1
        if config["replacement_of_link"]:
            save_file_name[1] = 1
        if config["removal_of_puntation"]:
            save_file_name[2] = 1
        if config["removal_of_stopwords"]:
            save_file_name[3] = 1
        if config["normalization_with_stemmer"]:
            save_file_name[4] = 1
        return ''.join(str(nr) for nr in save_file_name)

    # THe prosses of removing emojies. It uses regex 
    def replace_emoji(self):
        # Based on the wikipedia page  and the GITHUB POST https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
        # https://en.wikipedia.org/wiki/Unicode_block
        for index_row, row in self.retrival.current_celebrity.iterrows():
            feed = []
            # print(index_row,(self.retrival.size*self.percent), ((index_row) / (self.retrival.size*self.percent)))
            if (index_row) % int(int(self.retrival.size)*self.percent) == 0:
                self.terminal_info("Status", percent=str(int(index_row*100/int(self.retrival.size))))

            for index_tweet, tweet in enumerate(row[1]):
                text = tweet
                # flags (iOS)
                text = re.sub("[\U0001F1E0-\U0001F1FF]+", " EMOJIFLAG ", text)
                # symbols & pictographs
                text = re.sub("[\U0001F300-\U0001F5FF]+",
                              " EMOJISYMBOL ", text)
                # emoticons
                text = re.sub("[\U0001F600-\U0001F64F]+",
                              " EMOJIEMOTICON ", text)
                # transport & map symbols
                text = re.sub("[\U0001F680-\U0001F6FF]+",
                              " EMOJITRANSPORTMAP ", text)
                # alchemical symbols
                text = re.sub("[\U0001F700-\U0001F77F]+",
                              " EMOJIALCHEMICAL ", text)
                # Geometric Shapes Extended
                text = re.sub("[\U0001F780-\U0001F7FF]+",
                              " EMOJIGEOMETRY ", text)
                # Supplemental Arrows-C
                text = re.sub("[\U0001F800-\U0001F8FF]+",
                              " EMOJISUPPLEMENTARROWS ", text)
                # Supplemental Symbols and Pictographs
                text = re.sub("[\U0001F900-\U0001F9FF]+",
                              " EMOJISUPPLEMENTSYMBOLS ", text)
                # Chess Symbols
                text = re.sub("[\U0001FA00-\U0001FA6F]+", " EMOJICHESS ", text)
                # Symbols and Pictographs Extended-A
                text = re.sub("[\U0001FA70-\U0001FAFF]+",
                              " EMOJISYMBOLPICTOGRAPHEXTRA ", text)
                text = re.sub("[\U00002702-\U000027B0]+",
                              " EMOJIDINGBATS ", text)                
                # Dingbats
                text = re.sub("[\U000024C2-\U0001F251]+",
                              " EMOJIDINGBATS ", text)                
                # Dingbats
                feed.append(text)
            self.retrival.current_celebrity.at[index_row, "text"] = feed
            del feed

    # Replacement of URL link with a ned token
    def replace_link(self):
        # https://www.tutorialexample.com/best-practice-to-extract-and-remove-urls-from-python-string-python-tutorial/
        pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
        for index_row, row in self.retrival.current_celebrity.iterrows():
            feed = []
            if (index_row) % int(int(self.retrival.size)*self.percent) == 0:
                self.terminal_info("Status", percent=str(int(index_row*100/int(self.retrival.size))))

            for index_tweet, tweet in enumerate(row[1]):
                text = re.sub(pattern, "LINKURL", tweet)
                feed.append(text)
                del text
            self.retrival.current_celebrity.at[index_row, "text"] = feed
            del feed

    #Removial of puctation
    def remove_puntation(self):
        punct = string.punctuation+"’" + '“' + '”' + "…"
        for index_row, row in self.retrival.current_celebrity.iterrows():
            feed = []
            if (index_row) % int(int(self.retrival.size)*self.percent) == 0:
                self.terminal_info("Status", percent=str(int(index_row*100/int(self.retrival.size))))

            for index_tweet, tweet in enumerate(row[1]):
                text = tweet.translate(str.maketrans('', '', punct)).lower()
                feed.append(text)
                del text
            self.retrival.current_celebrity.at[index_row, "text"] = feed
            del feed

    # Stopword Remvoal
    def remove_stopwords(self):
        for index_row, row in self.retrival.current_celebrity.iterrows():
            feed = []
            if (index_row) % int(int(self.retrival.size)*self.percent) == 0:
                self.terminal_info("Status", percent=str(int(index_row*100/int(self.retrival.size))))

            for index_tweet, tweet in enumerate(row[1]):
                text = tweet.split(" ")
                text = [word.replace('\n', ' ').replace('\r', '') for word in text if (
                    not word.lower() in self.stopwords) and (word.lower())]
                text = [word for word in text if not word == "rt"]
                feed.append(text)
                del text
            self.retrival.current_celebrity.at[index_row, "text"] = feed
            del feed

    #Stemming
    def normalization(self):
        stemmer = PorterStemmer()

        for index_row, row in self.retrival.current_celebrity.iterrows():
            feed = []
            if (index_row) % int(int(self.retrival.size)*self.percent) == 0:
                self.terminal_info("Status", percent=str(int(index_row*100/int(self.retrival.size))))

            for index_tweet, tweet in enumerate(row[1]):
                text = [stemmer.stem(word) for word in tweet]
                feed.append(text)
                del text
            self.retrival.current_celebrity.at[index_row, "text"] = feed
            del feed

    #Stores the instance of the retrival object after prerprossesing
    def save_retrival(self, config, save_file_name):
        fileName = config["save_path"]
        if self.retrival.is_training_set:
            fileName += "training/"
        else:
            fileName += "test/"

        fileName += save_file_name
        fileName += config["pickle_suffix"]
        pickle.dump(self.retrival, open((fileName), "wb"))

    # Get the pickled data to skip preprssesing
    def load_retrival(self, config, save_file_name):
        fileName = config["save_path"]
        if self.retrival.is_training_set:
            fileName += "training/"
        else:
            fileName += "test/"

        fileName += save_file_name
        fileName += config["pickle_suffix"]
        loaded_retrival = pickle.load(open((fileName), "rb"))
        self.retrival = loaded_retrival

    def check_for_load(self, config, save_file_name):
        save_file_name += config["pickle_suffix"]
        path = config["save_path"]
        if self.retrival.is_training_set:
            path += "training/"
        else:
            path += "test/"
        entries = os.listdir(path)
        if save_file_name in entries:
            return True
        else:
            return False

    # Used to print nicly on the terminal to look like a hacker
    def terminal_info(self, info, percent=None):
        if not percent:
            print(f"\n[ Step: {self.steps} ]\tInitialization [{str(info)}]")
            self.steps += 1
        else:
            print(f"     |--------> ¦{str(info)}¦\t{percent}%")

        # print(self.retrival.current_celebrity)

    # Created the vecotizer for later usage
    def create_TFIDF_vectorizer(self, config):
        self.terminal_info("Create TF_IDF vector")
        vectorizer = TfidfVectorizer(max_features=config["vocabulary_size"], sublinear_tf = True)
        
        # vectorizer = TfidfVectorizer()
        all_tweet = []
        
        for index_row, row in self.retrival.current_celebrity.iterrows():
            feed = []
            if (index_row) % int(int(self.retrival.size)*self.percent) == 0:
                self.terminal_info("Status", percent=str(int(index_row*100/int(self.retrival.size))))
            for index_tweet, tweet in enumerate(row[1]):
                if len(tweet)>0:
                    tweet = " ".join(tweet)
                    feed.append(tweet)
                # del text
            # self.retrival.current_celebrity.at[index_row, "text"] = feed
            all_tweet.extend(feed)
            del feed
            # print(len(all_tweet))
        
        self.terminal_info("Fitting TF_IDF vector")
        vectorizer.fit(all_tweet)
        print(f"Total size:\t\t{len(all_tweet)}\nTfidfVectorizer size:\t{len(vectorizer.get_feature_names())}\nPercent:\t\t{len(all_tweet)*100/len(vectorizer.get_feature_names())}")
        self.terminal_info("Status", 100)
        del all_tweet
        return vectorizer
