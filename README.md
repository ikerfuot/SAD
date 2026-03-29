**Python 3.12 | Train: `python train.py -f datos.csv -p target -a algoritmo -v` | Test: `python test.py -f datos_test.csv -p target`**

# **SAD**

Sistema modular para el entrenamiento y evaluación de modelos de clasificación. Esta herramienta permite automatizar el preprocesamiento, la búsqueda de hiperparámetros (GridSearch) y la inferencia sobre nuevos datos.

---

### 🚀 Configuración del Entorno y Librerías
Para que este proyecto funcione correctamente, es fundamental contar con la versión adecuada de Python.

**Versión de Python: 3.12**
#### 1. Crear el entorno virtual
Abre tu terminal en la carpeta del proyecto y ejecuta:
#### Crear el entorno (llamado 'venv')
En terminal: python -m venv venv

En conda: conda create --name venv python=3.12
#### Activar el entorno:
En terminal:source venv/bin/activate

En conda: conda activate venv
#### 2. Descargar las librerías
Una vez activado el entorno, instalamos todo lo necesario con estos comandos:

En terminal: pip install -r requirements.txt
En conda:

conda install --yes --file requirements.txt

o si no probar con:

conda install pip

pip install -r requirements.txt

---

### 📂 Funciones de los Archivos del Proyecto

Cada archivo tiene una responsabilidad única dentro del flujo de Machine Learning.

* **train.py**: Es el motor de entrenamiento que limpia los datos, busca los mejores hiperparámetros y guarda el modelo optimizado en un archivo.
* **test.py**: Es el script de predicción que utiliza los objetos guardados para transformar datos nuevos y generar resultados sin re-entrenar el modelo.
* **configuration.json**: Es el centro de control que define las reglas de preprocesamiento y los rangos de búsqueda para los algoritmos sin tocar el código.
* **requirements.txt**: Es la lista de dependencias necesarias para asegurar que todos los usuarios tengan instaladas las mismas versiones de las librerías.

---

### Argumentos Obligatorios:
* -f, --file: Ruta al archivo .csv.
* -p, --prediction: Nombre de la columna objetivo (target).
* -a, --algorithm: Algoritmo a utilizar:
    * kNN: Vecinos más cercanos.
    * decision_tree: Árbol de decisión.
    * random_forest: Bosque aleatorio.
    * naive_bayes: Prueba automáticamente Gaussian, Multinomial y Bernoulli.

### Opciones Adicionales:
* -v, --verbose: Muestra mejores parámetros, F1-Score (micro/macro), Informe de clasificación y Matriz de confusión.
* -e, --estimator: Métrica para el GridSearch (f1_micro, f1_macro, accuracy, precision, recall).
* -c, --cpu: Número de núcleos a usar en el entrenamiento (default: -1, todos).

## ⚙️ Configuración (configuration.json)

El archivo JSON controla el comportamiento del preprocesamiento y los rangos de búsqueda de los modelos.

### 1. Bloque de Preprocesamiento (preprocessing)
Controla cómo se limpian y transforman los datos antes de entrar al modelo.

* **scaling**: Método de escalado numérico. Opciones: "standard", "minmax" o "none".
* **sampling**: Estrategia para manejar datasets desbalanceados. Opciones: "smote", "oversampling", "undersampling" o "none".
* **imbalance_threshold**: Porcentaje mínimo de la clase minoritaria (ej: 20.0). Si una clase tiene menos presencia que este valor, se activa el sampling.
* **dev_size**: Proporción del dataset original reservada para la evaluación (ej: 0.15).
* **drop_features**: Lista de nombres de columnas que quieres ignorar por completo (ej: ["id", "name"]).
* **unique_category_threshold**: Número máximo de valores únicos para que una columna de texto se considere "categórica" (y se use LabelEncoder) en lugar de "texto libre".
* **language**: Idioma para el procesamiento de texto y eliminación de stopwords (ej: "spanish", "english").
* **text_process**: Técnica de vectorización para columnas de texto. Opciones: "tf-idf", "bow" (Bag of Words) o "none".

---

### 2. Configuración de Algoritmos (Hiperparámetros)
Define las listas de valores que el `GridSearchCV` probará para encontrar la mejor combinación.

#### **kNN (K-Nearest Neighbors)**
* **n_neighbors**: Lista de números de vecinos a probar (ej: [1, 3, 5, 11]).
* **weights**: Función de peso. Opciones: ["uniform", "distance"].
* **p**: Métrica de distancia. 1 para Manhattan, 2 para Euclídea.

#### **Decision Tree**
* **criterion**: Función para medir la calidad de la división. Opciones: ["gini", "entropy"].
* **max_depth**: Profundidad máxima del árbol. Usar null para profundidad ilimitada.
* **min_samples_split**: Número mínimo de muestras necesarias para dividir un nodo.
* **min_samples_leaf**: Número mínimo de muestras que debe tener un nodo hoja.
* **ccp_alpha**: Parámetro de complejidad para la poda (pruning) del árbol.

#### **Random Forest**
* **n_estimators**: Número de árboles en el bosque (ej: [50, 100, 200]).
* **criterion**: Calidad de la división (igual que en Decision Tree).
* **max_features**: Número de características a considerar en cada división. Opciones: ["sqrt", "log2"].

#### **Naive Bayes**
* **type**: Escoge el modo en el que se va a ejecutar Naive Bayes. Opciones (solo se puede usar una de ellas): "gaussian", "multinomial", "bernoulli".
* **var_smoothing**: (Para Gaussian) Porción de la varianza máxima añadida para estabilidad (ej: [1e-9, 1e-8]).
* **alpha**: (Para Multinomial/Bernoulli) Parámetro de suavizado Laplace/Lidstone.
* **binarize**: (Solo para Bernoulli) Umbral para binarizar características.

---

## 📂 Salidas (Carpeta output/)

* modelo.pkl: El modelo ganador.
* scaler.pkl / vectorizer.pkl / label_encoders.pkl: Objetos para transformar datos en el test.
* modelo.csv: Historial de todas las combinaciones probadas y sus resultados.
* data-prediction.csv: (generado por test.py) El CSV original con la columna PREDICCION añadida.

## Ejemplo
`python train.py -f penguins.csv -p sex -a random_forest -e f1_macro -v`
`python test.py -f penguins_test.csv -p sex -v`