import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def Preprocessing(dataset):
    dataset['text'] = dataset['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    dataset['title'] = dataset['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    s_word = stopwords.words('english')
    dataset['text'] = dataset['text'].apply(lambda x: " ".join(x.lower() for x in x.split() if x not in s_word))
    dataset['title'] = dataset['title'].apply(lambda x: " ".join(x.lower() for x in x.split() if x not in s_word))
    stemm = PorterStemmer()
    dataset['text'] = dataset['text'].apply(lambda x: " ".join([stemm.stem(word) for word in x.split()]))
    dataset['title'] = dataset['title'].apply(lambda x: " ".join([stemm.stem(word) for word in x.split()]))
    # Feature Extraction
    col_1 = dataset["title"]
    col_2 = dataset["text"]
    X = TfIdf(col_1, col_2)
    Y = dataset['label']
    return X, Y


def TfIdf(col_1, col_2):
    cv = TfidfVectorizer()
    x = cv.fit_transform(col_1, col_2)
    df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names_out())
    return df


def Random_Forest_model(X_train, Y_train, X_test, Y_test):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, Y_train)
    prediction = rf_classifier.predict(X_test)
    print("Random Forest ////////////////////////////////")
    print(f"Test Set Accuracy : {accuracy_score(Y_test, prediction) * 100} %\n\n")


def Passive_Classification_model(X_train, Y_train, X_test, Y_test):
    model = PassiveAggressiveClassifier()
    model.fit(X_train, Y_train)
    print("Passive Classification ////////////////////")
    prediction = model.predict(X_test)
    print(f"Test Set Accuracy : {accuracy_score(Y_test, prediction) * 100} %\n\n")


def train_test(dataset):
    X, y = Preprocessing(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=10)
    Random_Forest_model(X_train, y_train, X_test, y_test)
    Passive_Classification_model(X_train, y_train, X_test, y_test)


def visualization_result():
    return None


dataset = pd.read_csv("news.csv")
train_test(dataset)
