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
            self.training_mode()
            self.size = len(self.current_labels.index)
        else:
            print(">"+" Initialization Testing Modus..")
            self.testing_mode()
            self.size = len(self.current_labels.index)

    def training_mode(self):
        self.get_training_celebrity()
        self.get_training_labels()

    def testing_mode(self):
        self.get_test_celebrity()
        self.get_test_labels()

    def get_training_labels(self):
        path = self.config["training_path"]+"labels_train.ndjson"
        time = self.timer()
        # print(f"Retriving[  ]:\t{end_path}")
        df = self.ndjson_to_dataframe(path)
        self.terminal_info(path, time)
        self.current_labels = df

    def get_training_celebrity(self):
        path = self.config["training_path"]+"celebrity_train.ndjson"
        time = self.timer()
        df = self.ndjson_to_dataframe(path)
        self.fileTitle = path
        self.terminal_info(path, time)
        self.current_celebrity = df

    def terminal_info(self, path, time):
        print(f"|\tTime:{round(self.timer(start_time=time),2)}\tRAM:{round(self.get_ram(),2)}%\t|\t - Retriving[✅]: {path}")
       
    def get_test_labels(self):
        path = self.config["test_path"] + "labels_test.ndjson"
        time  = self.timer()
        df = self.ndjson_to_dataframe(path)
        self.terminal_info(path, time )
        self.current_labels = df

    def get_test_celebrity(self):
        path = self.config["test_path"]+"celebrity_test.ndjson"
        time  = self.timer()
        df = self.ndjson_to_dataframe(path)
        self.fileTitle = path
        self.terminal_info(path, time)
        self.currentFile = df.copy()
        self.current_celebrity = df

    def ndjson_to_dataframe(self, path):
        records = map(json.loads, open(path))
        df = pd.DataFrame.from_records(records)
        del records
        return df

    def get_ram(self):
        ram = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        limit = 10
        if ram < limit:
            sys.exit(f"[Forced Quit ❌]\tLess than {limit}% RAM left")
        return ram


    def timer(self, start_time=None):
        if start_time is None:
            start_time = time.time()
            return start_time
        else:
            return (time.time() - start_time)


