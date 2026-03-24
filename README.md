# Valverde OUT - SAD

Plantilla para entrenamiento y evaluación de modelos de clasificación para la asignatura Sistemas de Apoyo a la Decisión.

El código está estructurado para particionar los datos, preprocesarlos, realizar una búsqueda de hiperparámetros y guardar el modelo para su posterior evaluación.

## Instalación y Preparación del Entorno

Es más fácil de lo que parece.

1. Crear el entorno virtual:
   python -m venv venv

2. Activar el entorno:
   source venv/bin/activate

3. Instalar dependencias:
   pip install -r requirements.txt

## Configuración (configuration.json)

Todos los hiperparámetros de los modelos y las reglas de preprocesado se controlan desde el archivo configuration.json. El script lee este archivo automáticamente. 

## Uso y Ejecución

El script funciona mediante argumentos por línea de comandos. La sintaxis básica es:

python plantilla_decisionTrees.py -m <MODO> -f <ARCHIVO_CSV> -p <COLUMNA_OBJETIVO> -a <ALGORITMO>

### Argumentos disponibles:
- -m, --mode: Modo de ejecución. Puede ser train (entrenamiento y validación) o test (evaluación ciega final). (Obligatorio)
- -f, --file: Ruta al fichero .csv con los datos. (Obligatorio)
- -p, --prediction: Nombre de la columna objetivo (Target) que se quiere predecir. (Obligatorio)
- -a, --algorithm: Algoritmo a ejecutar. Opciones válidas: kNN, decision_tree, random_forest. (Obligatorio)
- -e, --estimator: Métrica para elegir el mejor modelo (por defecto accuracy).
- -c, --cpu: Número de CPUs a utilizar para el GridSearch (por defecto -1, usa todos los núcleos).
- -v, --verbose: Muestra por terminal las métricas detalladas (Mejores parámetros, F1-score, Matriz de confusión).
- --debug: Muestra información adicional sobre cómo se han separado las columnas (numéricas, categóricas, texto) durante el preprocesado.

---

### Ejemplo 1: Fase de Entrenamiento (train)

En este modo, el sistema lee el CSV, lo divide en Train y Dev (Validación), balancea solo el Train si es necesario, busca los mejores hiperparámetros y guarda el modelo ganador.

python plantilla.py -m train -f penguins.csv -p sex -a kNN -v

### Ejemplo 2: Fase de Evaluación Definitiva (test)

El script no divide ni entrena nada; simplemente carga el modelo previamente guardado, preprocesa el nuevo archivo e imprime las predicciones.

python plantilla.py -m test -f penguins_test_secreto.csv -p sex -a kNN

## Archivos de Salida

Tras la ejecución, el script generará automáticamente una carpeta llamada output/ donde se guardarán:
- modelo.pkl: El modelo entrenado ganador (incluye el pipeline del GridSearch).
- scaler.pkl: El objeto instanciado para el escalado de datos numéricos.
- vectorizer.pkl: El objeto instanciado (TF-IDF/BOW) para los datos de texto.
- modelo.csv: Un registro de todos los parámetros probados por el GridSearch y su puntuación.
- data-prediction.csv: Generado solo en modo test, contiene el dataset original con una columna añadida al final con las predicciones del modelo.


![alt text](https://pbs.twimg.com/media/HDYSNYTXIAAbeFc.jpg)
