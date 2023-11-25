from flask import Flask, render_template, request, send_file
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
import csv
import os

app = Flask(__name__)

class ModelTrainer:
    # Iniciando os parametros
    def __init__(self, nome_clissificador, parametro1, parametro2, parametro3):
        self.nome_clissificador = nome_clissificador
        self.parametro1 = parametro1
        self.parametro2 = parametro2
        self.parametro3 = parametro3

    #Função que retorna o classificador escolhido
    def _escolhaClassificador(self):
        if self.nome_clissificador == 'KNN':
            return KNeighborsClassifier(n_neighbors=self.parametro2)
        elif self.nome_clissificador == 'SVM':
            return SVC(C=self.parametro1)
        elif self.nome_clissificador == 'MLP':
            return MLPClassifier(hidden_layer_sizes=(self.parametro2,), max_iter=1000)
        elif self.nome_clissificador == 'DT':
            return DecisionTreeClassifier(max_depth=self.parametro2)
        elif self.nome_clissificador == 'RF':
            return RandomForestClassifier(n_estimators=self.parametro2)
        else:
            return None
    #Treinar modelo escolhido
    def treinar_modelo(self, X_train, y_train, X_test):
        clf = self._escolhaClassificador()
        if clf is not None:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return y_pred
        else:
            return None
    #avalia o desempenho do modelo
    def desempenho_modelo(self, y_test, y_pred):
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred, average='macro')
        return accuracy, f1_score
    #matriz de confusão e plota grafico
    def matriz(self, y_test, y_pred):
        conf = metrics.confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 5))
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.getvalue()).decode()

        return img_data

    def salvar_resultado(self, y_test, y_pred):
        result_csv = StringIO()
        csv_writer = csv.writer(result_csv)
        csv_writer.writerow(['Actual', 'Predicted'])
        for actual, predicted in zip(y_test, y_pred):
            csv_writer.writerow([actual, predicted])

        result_csv.seek(0)
        temp_csv_file = f'results_{self.nome_clissificador}_{self.parametro1}_{self.parametro2}_{self.parametro3}.csv'
        temp_csv_path = os.path.join('tmp', temp_csv_file)
        with open(temp_csv_path, 'w') as csv_file:
            csv_file.write(result_csv.getvalue())

        return temp_csv_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    nome_clissificador = request.form['classifier']
    parametro1 = float(request.form.get('parametro1', 5.0))
    parametro2 = int(request.form.get('parametro2', 10))
    parametro3 = request.form.get('parametro3')

    X, y = np.random.rand(100, 2), np.random.choice([0, 1], size=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ModelTrainer(nome_clissificador, parametro1, parametro2, parametro3)
    y_pred = trainer.treinar_modelo(X_train, y_train, X_test)

    if y_pred is not None:
        accuracy, f1_score = trainer.desempenho_modelo(y_test, y_pred)
        img_data = trainer.matriz(y_test, y_pred)
        csv_path = trainer.salvar_resultado(y_test, y_pred)

        return render_template('result.html', classifier=nome_clissificador, accuracy=accuracy, f1_score=f1_score, img_data=img_data, csv_path=csv_path)
    else:
        return "Invalid classifier selected"

if __name__ == '__main__':
    
    app.run(debug=True)
