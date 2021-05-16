from data.retriver import Retriver
from data.preprossesor import Preprossesor
from data.init_files import Divider

from collections import Counter
from config import config
from classifier import Classifiers


#Divide data "Divider"
divider = Divider(config["train_test_split"])
divider.create_split()
del divider

# Object containg control over data - The Retrival
training = Retriver(config, training=True)

# Preprossesing - The Preprossesor
p_train = Preprossesor(training, config)
vectorizer = p_train.create_TFIDF_vectorizer(config)

#Update Retival
training = p_train.retrival

# Train classifiers with training data - The Classifier
classifier = Classifiers(training,vectorizer, config)

# Test Data
# classifier.predict_gender(test, "occupation") - Evaluation

test = Retriver(config, training=False)
p_test = Preprossesor(test, config)
test = p_test.retrival
classifier.train()
classifier.evaluate_all(test)

