#!/usr/bin/env python

import logging as log
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

log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=log.INFO)

preprocessed = [preprocess(sys.argv[1])]

log.info("vectorization")
cv = load("cv.joblib")
X_data = cv.transform(preprocessed)
model = load("model.joblib")
pred = model.predict(X_data)
print (pred[0])
