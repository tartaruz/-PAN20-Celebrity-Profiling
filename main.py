from data.retriver import Retriver
from data.preprossesor import Preprossesor
from collections import Counter
from config import config
from classifier import Classifiers

# Object containg control over data
training = Retriver(config, training=True)
test = Retriver(config, training=False)


# Preprossesing 
p_train = Preprossesor(training, config)
p_test = Preprossesor(test, config)
vectorizer = p_train.create_TFIDF_vectorizer(config)

#Update Retival
training = p_train.retrival
test = p_test.retrival

# Train classifiers with training data
classifier = Classifiers(training,vectorizer, config)

# Test Data
# classifier.predict_gender(test, "occupation")

classifier.train()
classifier.predict(test)
