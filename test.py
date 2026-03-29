# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import argparse
from colorama import Fore
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ------------------------------------
# CARGA DE CONFIGURACIÓN Y ARGUMENTOS
# ------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Script de Inferencia (Test)")
    parser.add_argument("-f", "--file", help="Fichero CSV con datos nuevos", required=True)
    parser.add_argument("-p", "--prediction", help="Columna objetivo (Target)", required=True)
    
    args = parser.parse_args()
    
    # Carga automática del JSON (sin bandera -c, igual que en plantilla.py)
    config_path = 'configuration.json'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    else:
        print(Fore.YELLOW + "Aviso: No se encontró configuration.json. Se usarán valores por defecto." + Fore.RESET)
        
    return args

# ------------------
# FUNCIONES DE APOYO
# ------------------

def select_features(df, args):
    numerical = df.select_dtypes(include=['int64', 'float64'])
    umbral = args.preprocessing.get("unique_category_threshold", 10)
    categorical = df.select_dtypes(include='object')
    categorical = categorical.loc[:, categorical.nunique() <= umbral]
    text = df.select_dtypes(include='object').drop(columns=categorical.columns)
    return numerical, text, categorical

def simplify_text(df, args):
    _, text_cols_df, _ = select_features(df, args)
    text_cols = text_cols_df.columns
    if len(text_cols) == 0: return df
    
    idioma = args.preprocessing.get("language", "spanish")
    stop_words = set(stopwords.words(idioma)) 
    stemmer = PorterStemmer()
    
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).apply(
            lambda text: ' '.join([stemmer.stem(w) for w in word_tokenize(text.lower()) 
                                   if w.isalnum() and w not in stop_words])
        )
    return df

# ----------------------
# PREPROCESADO PARA TEST
# ----------------------

def preprocesar_test(df, args):
    # 1. ELIMINAR COLUMNA OBJETIVO (Evita el ValueError de strings)
    target = args.prediction
    if target in df.columns:
        df = df.drop(columns=[target])

    # 2. Eliminar columnas indicadas en drop_features
    to_drop = [col for col in args.preprocessing.get("drop_features", []) if col in df.columns]
    df = df.drop(columns=to_drop)
    
    # 3. Valores Faltantes (Cargando medias/modas de Train)
    if os.path.exists('output/missing_values.pkl'):
        fill_values = pickle.load(open('output/missing_values.pkl', 'rb'))
        # Filtramos para rellenar solo las columnas que existen en este CSV
        valid_fills = {k: v for k, v in fill_values.items() if k in df.columns}
        df = df.fillna(value=valid_fills)

    # 4. Categorías a Números (LabelEncoder)
    if os.path.exists('output/label_encoders.pkl'):
        encoders = pickle.load(open('output/label_encoders.pkl', 'rb'))
        for col, le in encoders.items():
            if col in df.columns:
                # Manejo de etiquetas desconocidas
                df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

    # 5. Texto (Solo Transform)
    if os.path.exists('output/vectorizer.pkl'):
        vectorizer = pickle.load(open('output/vectorizer.pkl', 'rb'))
        df = simplify_text(df, args)
        _, text_cols_df, _ = select_features(df, args)
        text_cols = text_cols_df.columns
        if len(text_cols) > 0:
            text_combined = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            matrix = vectorizer.transform(text_combined)
            df_text = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
            df = pd.concat([df.drop(columns=text_cols).reset_index(drop=True), df_text], axis=1)

    # 6. Escalado (Solo Transform)
    if os.path.exists('output/scaler.pkl'):
        scaler = pickle.load(open('output/scaler.pkl', 'rb'))
        # Detectamos las columnas numéricas
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) > 0:
            # Transformamos los datos
            scaled_data = scaler.transform(df[num_cols])
            
            # Asignamos columna por columna para permitir que Pandas
            # cambie el tipo de dato de int64 a float64 automáticamente
            for i, col in enumerate(num_cols):
                df[col] = scaled_data[:, i]
            
    return df

# ------------------------------------
# MAIN
# ------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Descargas NLTK
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    if not os.path.exists(args.file):
        print(Fore.RED + f"Error: No se encuentra el archivo {args.file}" + Fore.RESET)
        sys.exit(1)
        
    data = pd.read_csv(args.file)
    data_original = data.copy()

    # Preprocesar
    X_test_clean = preprocesar_test(data, args)

    # Cargar el mejor modelo (GridSearch)
    if os.path.exists('output/modelo.pkl'):
        model_gs = pickle.load(open('output/modelo.pkl', 'rb'))
    else:
        print(Fore.RED + "Error: No existe modelo.pkl. Entrena el modelo primero." + Fore.RESET)
        sys.exit(1)

    # Predicción (usa el array de numpy limpio de strings)
    print(Fore.CYAN + "Realizando predicciones..." + Fore.RESET)
    predicciones = model_gs.predict(X_test_clean.values)

    # IMPRIMIR RESULTADOS
    target = args.prediction
    if target in data_original.columns:
        print(Fore.MAGENTA + f"\n=== RESULTADOS EN TEST ===" + Fore.RESET)
        
        y_true = data_original[target]
        
        # filtrar filas con valores no nulos en la columna objetivo
        mascara_validos = y_true.notnull()
        y_true_eval = y_true[mascara_validos].astype(str)
        predicciones_eval = predicciones[mascara_validos]

        # para fallo de ValueError: could not convert string to float
        if hasattr(predicciones_eval, "astype"):
            predicciones_eval = predicciones_eval.astype(str)
        else:
            predicciones_eval = np.array([str(p) for p in predicciones_eval])

        print(Fore.YELLOW + "> F1-score micro:" + Fore.RESET, f1_score(y_true_eval, predicciones_eval, average='micro', zero_division=0))
        print(Fore.YELLOW + "> F1-score macro:" + Fore.RESET, f1_score(y_true_eval, predicciones_eval, average='macro', zero_division=0))
        print(Fore.YELLOW + "> Informe de clasificación:\n" + Fore.RESET, classification_report(y_true_eval, predicciones_eval, zero_division=0))
        
        print(Fore.YELLOW + "> Matriz de confusión:\n" + Fore.RESET)
        cm = confusion_matrix(y_true_eval, predicciones_eval)
        # Obtenemos etiquetas únicas uniendo las reales y las predichas para que cuadren las columnas
        etiquetas = sorted(list(set(y_true_eval) | set(predicciones_eval)))
        df_cm = pd.DataFrame(cm, index=[f"Real: {e}" for e in etiquetas], columns=[f"Pred: {e}" for e in etiquetas])
        print(df_cm)
    else:
        print(Fore.YELLOW + "\n[!] No se encontró la columna objetivo. Solo se generarán predicciones." + Fore.RESET)

    # Guardar resultados
    data_original['PREDICCION'] = predicciones
    data_original.to_csv('output/data-prediction.csv', index=False)
    
    print(Fore.GREEN + "Predicciones guardadas en 'output/data-prediction.csv'" + Fore.RESET)