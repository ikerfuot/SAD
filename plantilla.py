# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# ------------------------------------
# FUNCIONES AUXILIARES Y ARGS
# ------------------------------------

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('configuration.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# ------------------------------------
# FUNCIONES PARA CALCULAR METRICAS
# ------------------------------------

def calculate_fscore(y_true, y_pred):
    fscore_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    fscore_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return fscore_micro, fscore_macro

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def calculate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0)

# ------------------------------------
# FUNCIONES DE PREPROCESADO
# ------------------------------------

def select_features(data_subset):
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.
    """
    try:
        # 1. Numéricas (Solo miramos en data_subset, NADA de 'data' global)
        numerical_feature = data_subset.select_dtypes(include=['int64', 'float64']) 
            
        # 2. Categóricas
        categorical_feature = data_subset.select_dtypes(include='object')
        umbral = args.preprocessing.get("unique_category_threshold", 10)
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= umbral]
        
        # 3. Texto
        text_feature = data_subset.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, list(numerical_feature.columns))
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, list(text_feature.columns))
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, list(categorical_feature.columns))
            
        return numerical_feature, text_feature, categorical_feature
        
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_missing_values(X_train, X_dev):
    """
    Procesa valores faltantes. Fit en Train, Transform en Train y Dev.
    """
    num_train, cat_train = select_features(X_train)[0], select_features(X_train)[2]
    
    for col in num_train.columns:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_dev[col] = X_dev[col].fillna(mean_val)

    for col in cat_train.columns:
        if not X_train[col].mode().empty:
            mode_val = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_val)
            X_dev[col] = X_dev[col].fillna(mode_val)
            
    return X_train, X_dev

def reescaler(X_train, X_dev):
    """
    Rescala características. Fit en Train, Transform en Train y Dev.
    """
    num_cols = select_features(X_train)[0].columns
    if len(num_cols) == 0: return X_train, X_dev
    
    scaling = args.preprocessing.get("scaling", "standard")
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        return X_train, X_dev
        
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_dev.loc[:, num_cols] = scaler.transform(X_dev[num_cols])
    
    with open('output/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return X_train, X_dev

def cat2num(X_train, X_dev):
    """
    Convierte categóricas a numéricas. Fit en Train, Transform en Train y Dev.
    """
    cat_cols = select_features(X_train)[2].columns
    le = LabelEncoder()
    for col in cat_cols:
        X_train.loc[:, col] = le.fit_transform(X_train[col].astype(str))
        X_dev.loc[:, col] = X_dev[col].map(lambda s: s if s in le.classes_ else '<unknown>')
        le.classes_ = np.append(le.classes_, '<unknown>')
        X_dev.loc[:, col] = le.transform(X_dev[col].astype(str))
    return X_train, X_dev

def simplify_text(data_subset):
    """
    Simplifica texto.
    """
    text_cols = select_features(data_subset)[1].columns
    if len(text_cols) == 0: return data_subset
    
    idioma = args.preprocessing.get("language", "spanish")
    stop_words = set(stopwords.words(idioma)) 
    stemmer = PorterStemmer()
    
    for col in text_cols:
        data_subset.loc[:, col] = data_subset[col].fillna("").astype(str).apply(
            lambda text: ' '.join([stemmer.stem(w) for w in word_tokenize(text.lower()) 
                                   if w.isalnum() and w not in stop_words])
        )
    return data_subset

def process_text(X_train, X_dev):
    """
    Vectoriza texto. Fit en Train, Transform en Train y Dev.
    """
    text_cols = select_features(X_train)[1].columns
    if len(text_cols) == 0: return X_train, X_dev
    
    X_train = simplify_text(X_train)
    X_dev = simplify_text(X_dev)
    
    tipo_proceso = args.preprocessing.get("text_process", "none")
    if tipo_proceso == "tf-idf":
        vectorizer = TfidfVectorizer()
    elif tipo_proceso == "bow":
        vectorizer = CountVectorizer()
    else:
        return X_train, X_dev
        
    train_text = X_train[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    dev_text = X_dev[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    matrix_train = vectorizer.fit_transform(train_text)
    matrix_dev = vectorizer.transform(dev_text)
    
    with open('output/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    cols_vec = vectorizer.get_feature_names_out()
    
    X_train = pd.concat([X_train.drop(columns=text_cols).reset_index(drop=True), 
                         pd.DataFrame(matrix_train.toarray(), columns=cols_vec)], axis=1)
    X_dev = pd.concat([X_dev.drop(columns=text_cols).reset_index(drop=True), 
                       pd.DataFrame(matrix_dev.toarray(), columns=cols_vec)], axis=1)
                       
    return X_train, X_dev

def over_under_sampling(X_train, y_train):
    """
    Realiza over o under sampling SOLO en los datos de entrenamiento.
    """
    estrategia = args.preprocessing.get("sampling", "none").lower()
    if estrategia == "none": return X_train, y_train
    
    try:
        if estrategia == "smote":
            sampler = SMOTE(random_state=42)
        elif estrategia == "oversampling":
            sampler = RandomOverSampler(random_state=42)
        elif estrategia == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X_train, y_train
            
        X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
        return X_train_bal, y_train_bal
    except Exception as e:
        return X_train, y_train

def drop_features(data_subset):
    """
    Elimina las columnas especificadas.
    """
    to_drop = [col for col in args.preprocessing.get("drop_features", []) if col in data_subset.columns]
    if to_drop:
        data_subset = data_subset.drop(columns=to_drop)
    return data_subset

def preprocesar_datos():
    global data
    data = drop_features(data)
    
    # Borrar filas con valores nulos en la columna a predecir
    if data[args.prediction].isnull().any():
        data = data.dropna(subset=[args.prediction]).reset_index(drop=True)

    # Eliminar filas con clases que tengan solo 1 elemento (rompen el stratify)
    conteo_clases = data[args.prediction].value_counts()
    clases_validas = conteo_clases[conteo_clases > 1].index # Nos quedamos con las que aparecen 2 o más veces
    if len(clases_validas) < len(conteo_clases):
        data = data[data[args.prediction].isin(clases_validas)].reset_index(drop=True)

    # Dividir datos
    X = data.drop(columns=[args.prediction])
    y = data[args.prediction]
    
    test_size = args.preprocessing.get("test_size", 0.25)
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    X_train, X_dev = X_train.copy(), X_dev.copy()
    
    # Comprobar balanceo y si es necesario balancear
    distribucion = y_train.value_counts(normalize=True) * 100
    if distribucion.min() < args.preprocessing.get("imbalance_threshold", 30.0):
        X_train, y_train = over_under_sampling(X_train, y_train)
        
    # Preprocesar (FIT TRAIN, TRANSFORM TRAIN+DEV)
    X_train, X_dev = process_missing_values(X_train, X_dev)
    X_train, X_dev = cat2num(X_train, X_dev)
    X_train, X_dev = process_text(X_train, X_dev)
    X_train, X_dev = reescaler(X_train, X_dev)
    
    return X_train.values, X_dev.values, y_train.values, y_dev.values


# ------------------------------------
# FUNCIONES DE ENTRENAMIENTO
# ------------------------------------
 
def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        y_pred = gs.predict(x_dev)
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred)[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred)[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, y_pred))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, y_pred))

def kNN(x_train, x_dev, y_train, y_dev):
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos.
    """
    print(Fore.GREEN+"Haciendo barrido de hiperparametros para kNN"+Fore.RESET)
    param_grid = getattr(args, 'kNN', {})
    knn_clf = KNeighborsClassifier()
    gs = GridSearchCV(estimator=knn_clf, param_grid=param_grid, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    
    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    
    print("Tiempo de ejecución:"+Fore.MAGENTA, f"{end_time - start_time:.2f}",Fore.RESET+ "segundos")
    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs)

def decision_tree(x_train, x_dev, y_train, y_dev):
    """
    Función para implementar el algoritmo de árbol de decisión.
    """
    param_grid = {
        'max_depth': args.decision_tree.get('max_depth'),
        'min_samples_split': args.decision_tree.get('min_samples_split'),
        'criterion': args.decision_tree.get('criterion'),
        'min_impurity_decrease': args.decision_tree.get('threshold_IG', [0.0]), 
        'min_samples_leaf': args.decision_tree.get('min_samples_leaf', [1]),
        'ccp_alpha': args.decision_tree.get('ccp_alpha', [0.0])
    }

    dt_clf = DecisionTreeClassifier(random_state=42)
    gs = GridSearchCV(
            estimator=dt_clf,
            param_grid=param_grid,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator,
            verbose=0
    )

    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    
    print("Tiempo de ejecución:"+Fore.MAGENTA, f"{end_time - start_time:.2f}",Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
def random_forest(x_train, x_dev, y_train, y_dev):
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    """
    print(Fore.YELLOW + "\nProcesando Random Forest (GridSearchCV)..." + Fore.RESET)
    
    # Intentamos leer args.random_forest. Si no existe, devuelve {}
    param_grid = getattr(args, 'random_forest', {})
    
    rf_clf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    
    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)
    
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)
    
# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)

    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    if args.mode == "train":
        # Preprocesamos los datos
        print("\n- Preprocesando datos...")
        x_train, x_dev, y_train, y_dev = preprocesar_datos() # esto devuelve ya los splits

        if args.mode == "train":
            # Ejecutamos el algoritmo seleccionado
            print("\n- Ejecutando algoritmo...")
            if args.algorithm == "kNN":
                try:
                    kNN(x_train, x_dev, y_train, y_dev)
                    print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                    sys.exit(0)
                except Exception as e:
                    print(e)
            elif args.algorithm == "decision_tree":
                try:
                    decision_tree(x_train, x_dev, y_train, y_dev)
                    print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                    sys.exit(0)
                except Exception as e:
                    print(e)
            elif args.algorithm == "random_forest":
                try:
                    random_forest(x_train, x_dev, y_train, y_dev)
                    print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                    sys.exit(0)
                except Exception as e:
                    print(e)
            else:
                print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
                sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv('output/data-prediction.csv', index=False)
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)