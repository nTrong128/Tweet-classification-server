from contextlib import asynccontextmanager
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec


import joblib
import os


class FeatureType(str, Enum):
    tf_idf = "tf_idf"
    word2vec = "word2vec"
    bag_of_words = "bag_of_words"


class ModelType(str, Enum):
    knn = "KNN"
    bayes = "Bayes"
    decision_tree = "Decision Tree"
    random_forest = "Random Forest"
    logistic_regression = "Logistic Regression"
    svm_linear = "SVM Linear"
    svm_non_linear = "SVM Non-linear"


models = {}
preprocess_model = {}


def vectorize_text(text, model):
    words = word_tokenize(text.lower())
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return (
        np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    )


def tf_idf(text: str):
    return preprocess_model[FeatureType.tf_idf].transform([text])


def word2vec(text: str):
    return [vectorize_text(text, preprocess_model[FeatureType.word2vec])]


def bag_of_words(text: str):
    return preprocess_model[FeatureType.bag_of_words].transform([text])


@asynccontextmanager
async def lifespan(app: FastAPI):
    preprocess_model[FeatureType.word2vec] = Word2Vec.load(
        "saved_models/word2vec/word2vec_model"
    )
    preprocess_model[FeatureType.tf_idf] = joblib.load(
        "saved_models/tf_idf/tfidf_vectorizer.joblib"
    )
    preprocess_model[FeatureType.bag_of_words] = joblib.load(
        "saved_models/bag_of_words/count_vectorizer.joblib"
    )

    models[FeatureType.tf_idf] = {
        ModelType.knn: joblib.load("saved_models/tf_idf/KNN.joblib"),
        ModelType.bayes: joblib.load("saved_models/tf_idf/Bayes.joblib"),
        ModelType.decision_tree: joblib.load(
            "saved_models/tf_idf/Decision Tree.joblib"
        ),
        ModelType.random_forest: joblib.load(
            "saved_models/tf_idf/Random Forest.joblib"
        ),
        ModelType.logistic_regression: joblib.load(
            "saved_models/tf_idf/Logistic Regression.joblib"
        ),
        ModelType.svm_linear: joblib.load("saved_models/tf_idf/SVM Linear.joblib"),
        ModelType.svm_non_linear: joblib.load(
            "saved_models/tf_idf/SVM Non-linear.joblib"
        ),
    }
    models[FeatureType.word2vec] = {
        ModelType.knn: joblib.load("saved_models/word2vec/KNN.joblib"),
        ModelType.bayes: joblib.load("saved_models/word2vec/Bayes.joblib"),
        ModelType.decision_tree: joblib.load(
            "saved_models/word2vec/Decision Tree.joblib"
        ),
        ModelType.random_forest: joblib.load(
            "saved_models/word2vec/Random Forest.joblib"
        ),
        ModelType.logistic_regression: joblib.load(
            "saved_models/word2vec/Logistic Regression.joblib"
        ),
        ModelType.svm_linear: joblib.load("saved_models/word2vec/SVM Linear.joblib"),
        ModelType.svm_non_linear: joblib.load(
            "saved_models/word2vec/SVM Non-linear.joblib"
        ),
    }
    models[FeatureType.bag_of_words] = {
        ModelType.knn: joblib.load("saved_models/bag_of_words/KNN.joblib"),
        ModelType.bayes: joblib.load("saved_models/bag_of_words/Bayes.joblib"),
        ModelType.decision_tree: joblib.load(
            "saved_models/bag_of_words/Decision Tree.joblib"
        ),
        ModelType.random_forest: joblib.load(
            "saved_models/bag_of_words/Random Forest.joblib"
        ),
        ModelType.logistic_regression: joblib.load(
            "saved_models/bag_of_words/Logistic Regression.joblib"
        ),
        ModelType.svm_linear: joblib.load(
            "saved_models/bag_of_words/SVM Linear.joblib"
        ),
        ModelType.svm_non_linear: joblib.load(
            "saved_models/bag_of_words/SVM Non-linear.joblib"
        ),
    }
    print("All models loaded successfully")
    yield  # Allow the app to run

    # Clear models dictionary on shutdown
    models.clear()
    print("Models cleared")


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str
    model: ModelType
    feature_type: FeatureType


@app.post("/predict")
async def predict(request: PredictionRequest):
    feature_type = request.feature_type
    model_name = request.model
    text = request.text

    try:
        model = models[feature_type][model_name]
    except KeyError:
        raise HTTPException(status_code=400, detail="Model or feature type not found")

    preprocessing_text = None

    if feature_type == "tf_idf":
        preprocessing_text = tf_idf(text).reshape(1, -1)
    elif feature_type == "word2vec":
        preprocessing_text = word2vec(text)
    elif feature_type == "bag_of_words":
        preprocessing_text = bag_of_words(text).toarray()

    prediction = model.predict(preprocessing_text)[0]
    if prediction == 0:
        prediction = "Negative"
    else:
        prediction = "Positive"

    json_compatible_item_data = jsonable_encoder({"prediction": prediction})
    return JSONResponse(content=json_compatible_item_data)
