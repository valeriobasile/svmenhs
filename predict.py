#!/usr/bin/env python

from joblib import load
import sys
import re

def dummy(doc):
    return doc

def preprocess(text):
    text = text.lower()
    text = re.sub(r"([^\w]?)@[\w]*([^\w]?)", r"\1MENTION\2", text)
    text = re.sub(r"([^\w]?)#[\w]*([^\w]?)", r"\1HASHTAG\2", text)
    text = re.sub(r"([^\w]?)https?://[\w./]*([^\w]?)", r"\1URL\2", text)
    tokens = re.split(r"[\W]+", text)
    return tokens


# loading pre-trained models
cv = load("cv.joblib")
model = load("model.joblib")

#read input
preprocessed = [preprocess(sys.argv[1])]

# process input
X_data = cv.transform(preprocessed)

# predict label
pred = model.predict(X_data)
print (pred[0])
