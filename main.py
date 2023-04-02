import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay, recall_score, precision_score, f1_score, classification_report
import pandas as pd
from seaborn import heatmap
from sklearn import tree


def format_data(data):
    data_list = data.values.tolist()
    symptoms = {}
    for disease in data_list:
        for element in disease:
            if (disease[0] == element) or (type(element) == float):
                continue
            symptoms[element] = 1
    symptoms = list(symptoms.keys())
    new_symptoms = []
    for sym in symptoms:
        new_symptoms.append(str(sym).replace(" ", ""))
    data = create_new_dataframe(data_list, symptoms)
    symptoms.remove('Disease')
    return data, symptoms


def create_new_dataframe(data_list, symptoms):
    symptoms.sort()
    new_data_list = []
    for disease in data_list:
        new_row = format_row(disease, symptoms)
        new_data_list.append(new_row)
        idx = 0
    symptoms.insert(0, 'Disease')
    new_data = pd.DataFrame(new_data_list, columns=symptoms)
    return new_data


def format_row(disease, symptoms):
    name = disease[0]
    disease.remove(name)
    new_disease = []
    for element in disease:
        if type(element) != float:
            new_disease.append(element)
    new_disease.sort()
    row = [name]
    idx = 0
    for symptom in symptoms:
        if len(new_disease) == idx:
            idx -= 1
        if (new_disease[idx] == symptom):
            row.append(1)
            idx += 1
        else:
            row.append(0)
    return row


def build_model_multinomial_LogReg(data, train_test_ratio, chosen_features, target, random_state=45, model=LogisticRegression(multi_class="multinomial", random_state=42)):

    data_train, data_test = train_test_split(data, test_size=train_test_ratio, random_state=random_state)
    X_train, X_test = data_train[chosen_features], data_test[chosen_features]
    y_train, y_test = data_train[target], data_test[target]

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)

    prec = precision(cm)
    rec = recall(cm)
    f1 = (2 * prec * rec)/(prec + rec)
    # ROC_curve(y_test, y_pred_test)
    
    return model


def recall(cm):
    if not (cm[1,1]+cm[1,0]):
        return 0
    return cm[1,1] / (cm[1,1]+cm[1,0])


def precision(cm):
    if not (cm[1,1]+cm[0,1]):
        return 0
    return cm[1,1] / (cm[1,1]+cm[0,1])


def ROC_curve(y_test, preds):
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.show()

def query(model, symptoms):
    print(symptoms)
    given_symptoms = input("Podaj objawy oddzielając spacją:")
    given_symptoms = given_symptoms.split()
    given_symptoms.sort()
    idx = 0
    row = []
    for symptom in symptoms:
        if len(given_symptoms) == idx:
            idx -= 1
        if (given_symptoms[idx] == symptom):
            row.append(1)
            idx += 1
        else:
            row.append(0)
    row = [row]
    predicted_illness = model.predict(row)
    print(f"Pacjent choruje na {predicted_illness[0]}")
    

dataset = pd.read_csv('dataset.csv')
dataset, features = format_data(dataset)
target = 'Disease'
model = build_model_multinomial_LogReg(dataset, 0.2, features, target)
query(model, features)
