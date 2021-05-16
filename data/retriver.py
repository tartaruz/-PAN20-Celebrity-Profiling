import ujson as json
import pandas as pd
import pickle
import time
import psutil
import sys


class Retriver():
    def __init__(self, config, training=False):
        self.config = config
        self.current_labels = None
        self.current_celebrity = None
        self.is_training_set = training
        if training:
            print(">"+" Initialization Training Modus..")
            self.size = 1920*(config["train_test_split"])
            self.training_mode()
        else:
            print(">"+" Initialization Testing Modus..")
            self.size = 1920*(1 - config["train_test_split"])
            self.testing_mode()

    # Retrives from the training data set
    def training_mode(self):
        self.get_training_celebrity()
        self.get_training_labels()

    # Retrives from the test data set
    def testing_mode(self):
        self.get_test_celebrity()
        self.get_test_labels()

    # Gets the labesl data for training
    def get_training_labels(self):
        path = self.config["training_path"]+"labels_train.ndjson"
        time = self.timer()
        # print(f"Retriving[  ]:\t{end_path}")
        df = self.ndjson_to_dataframe(path)
        self.terminal_info(path, time)
        self.current_labels = df


    # Gets the celebrity data for training
    def get_training_celebrity(self):
        path = self.config["training_path"]+"celebrity_train.ndjson"
        time = self.timer()
        df = self.ndjson_to_dataframe(path)
        self.fileTitle = path
        self.terminal_info(path, time)
        self.current_celebrity = df

    def terminal_info(self, path, time):
        print(f"|\tTime:{round(self.timer(start_time=time),2)}\tRAM:{round(self.get_ram(),2)}%\t|\t - Retriving[✅]: {path}")
       
    # Gets the labesl data for test
    def get_test_labels(self):
        path = self.config["test_path"] + "labels_test.ndjson"
        time  = self.timer()
        df = self.ndjson_to_dataframe(path)
        self.terminal_info(path, time )
        self.current_labels = df

    # Gets the celebrity data for test
    def get_test_celebrity(self):
        path = self.config["test_path"]+"celebrity_test.ndjson"
        time  = self.timer()
        df = self.ndjson_to_dataframe(path)
        self.fileTitle = path
        self.terminal_info(path, time)
        self.currentFile = df.copy()
        self.current_celebrity = df

    # Converts the ndjson file to a datafram from pandas
    def ndjson_to_dataframe(self, path):
        records = map(json.loads, open(path))
        df = pd.DataFrame.from_records(records)
        del records
        return df

    # Returns the usage of ram
    def get_ram(self):
        ram = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        limit = 10
        if ram < limit:
            sys.exit(f"[Forced Quit ❌]\tLess than {limit}% RAM left")
        return ram

    # To find how much time a prosses takes
    def timer(self, start_time=None):
        if start_time is None:
            start_time = time.time()
            return start_time
        else:
            return (time.time() - start_time)


