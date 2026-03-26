# -*- coding: utf-8 -*-
"""
Script para la evaluación de modelos de clasificación.

"""
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import argparse
from colorama import Fore

# Sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# Nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# ------------------------------------
# CARGA DE CONFIGURACIÓN Y ARGUMENTOS
# ------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Script de Inferencia (Test)")
    parser.add_argument("-f", "--file", help="Fichero CSV con datos nuevos", required=True)
    parser.add_argument("-c", "--config", help="Fichero JSON de configuración", default="configuration.json")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(args, key, value)
    return args

# ------------------------------------
# FUNCIONES DE APOYO (ESPEJO DE TRAIN)
# ------------------------------------

def select_features(df):
    numerical = df.select_dtypes(include=['int64', 'float64'])
    # Simulamos el umbral del JSON para categóricas
    umbral = 10 
    categorical = df.select_dtypes(include='object')
    categorical = categorical.loc[:, categorical.nunique() <= umbral]
    text = df.select_dtypes(include='object').drop(columns=categorical.columns)
    return numerical, text, categorical

def simplify_text(df, args):
    text_cols = select_features(df)[1].columns
    if len(text_cols) == 0: return df
    
    stop_words = set(stopwords.words(args.preprocessing.get("language", "spanish"))) 
    stemmer = PorterStemmer()
    
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).apply(
            lambda text: ' '.join([stemmer.stem(w) for w in word_tokenize(text.lower()) 
                                   if w.isalnum() and w not in stop_words])
        )
    return df

# ------------------------------------
# PREPROCESADO PARA TEST (SOLO TRANSFORM)
# ------------------------------------

def preprocesar_test(df, args):
    # 1. Eliminar columnas innecesarias
    to_drop = [col for col in args.preprocessing.get("drop_features", []) if col in df.columns]
    df = df.drop(columns=to_drop)
    
    # 2. Valores Faltantes (Cargando medias/modas guardadas)
    # Nota: El train debe guardar un diccionario con estos valores
    if os.path.exists('output/missing_values.pkl'):
        fill_values = pickle.load(open('output/missing_values.pkl', 'rb'))
        df = df.fillna(value=fill_values)

    # 3. Categorías a Números (Cargando LabelEncoders guardados)
    cat_cols = select_features(df)[2].columns
    if os.path.exists('output/label_encoders.pkl'):
        encoders = pickle.load(open('output/label_encoders.pkl', 'rb'))
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                # Manejar etiquetas nuevas que no estaban en train
                df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

    # 4. Texto (Solo Transform)
    if os.path.exists('output/vectorizer.pkl'):
        vectorizer = pickle.load(open('output/vectorizer.pkl', 'rb'))
        df = simplify_text(df, args)
        text_cols = select_features(df)[1].columns
        if len(text_cols) > 0:
            text_combined = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            matrix = vectorizer.transform(text_combined)
            df_text = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
            df = pd.concat([df.drop(columns=text_cols).reset_index(drop=True), df_text], axis=1)

    # 5. Escalado (Solo Transform)
    if os.path.exists('output/scaler.pkl'):
        scaler = pickle.load(open('output/scaler.pkl', 'rb'))
        num_cols = df.select_dtypes(include=[np.number]).columns
        # IMPORTANTE: Asegurarse de pasar solo las columnas que el scaler conoce
        df.loc[:, num_cols] = scaler.transform(df[num_cols])
        
    return df

# ------------------------------------
# MAIN EJECUCIÓN
# ------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Descargas necesarias
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    print(f"{Fore.CYAN}- Cargando datos de test: {args.file}{Fore.RESET}")
    if not os.path.exists(args.file):
        print(Fore.RED + "Error: Fichero no encontrado." + Fore.RESET)
        sys.exit(1)
        
    data = pd.read_csv(args.file)
    data_original = data.copy()

    # Preprocesar
    print(Fore.CYAN + "- Aplicando preprocesamiento de Test..." + Fore.RESET)
    X_test = preprocesar_test(data, args)

    # Cargar Modelo
    print(Fore.CYAN + "- Cargando mejor modelo entrenado..." + Fore.RESET)
    if os.path.exists('output/modelo.pkl'):
        model_gs = pickle.load(open('output/modelo.pkl', 'rb'))
    else:
        print(Fore.RED + "Error: No existe modelo.pkl en /output. Ejecuta el train primero." + Fore.RESET)
        sys.exit(1)

    # Predicción
    print(Fore.YELLOW + "- Realizando predicciones..." + Fore.RESET)
    # El objeto model_gs (GridSearchCV) usa automáticamente el best_estimator_
    predicciones = model_gs.predict(X_test.values)

    # Guardar resultados
    data_original['PREDICCION'] = predicciones
    data_original.to_csv('output/data-prediction.csv', index=False)
    
    print(f"{Fore.GREEN}¡Éxito! Resultados guardados en 'output/data-prediction.csv'{Fore.RESET}")