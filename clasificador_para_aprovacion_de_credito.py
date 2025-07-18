"""
clasificador_para_aprovacion_de_credito.py

Este script implementa un modelo de Machine Learning para predecir el incumplimiento
de pago de clientes de tarjetas de crédito. Realiza un flujo completo, incluyendo:

- Carga y exploración inicial de datos.
- Preprocesamiento de datos: limpieza, manejo de valores inválidos, balanceo de clases.
- Ingeniería de características: codificación One-Hot para variables categóricas.
- División de datos en conjuntos de entrenamiento y prueba.
- Entrenamiento y evaluación de un clasificador RandomForestClassifier.
- Optimización de hiperparámetros del modelo usando RandomizedSearchCV.
- Visualización de la matriz de confusión para evaluar el rendimiento del modelo.

El objetivo es demostrar la aplicación de técnicas de ML para la evaluación de riesgo crediticio.

Librerías principales utilizadas: pandas, numpy, scikit-learn, matplotlib, seaborn, xlrd.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import warnings

# Suppress warnings:
# Se deshabilitan las advertencias para una salida de consola más limpia.
# Esto es útil para presentaciones, pero en desarrollo, las advertencias son importantes.
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# Configuración estética de Seaborn para gráficos
sns.set_context('notebook')
sns.set_style('white')

# --- Funciones para el flujo de análisis ---

def load_data(url: str) -> pd.DataFrame:
    """
    Carga el conjunto de datos de clientes de tarjetas de crédito desde una URL.

    Args:
        url (str): La URL del archivo Excel (.xls) a cargar.

    Returns:
        pd.DataFrame: El DataFrame de pandas con los datos cargados.
    """
    print(f"Cargando datos desde: {url}")
    # 'header=1' porque la primera fila es el título general, los nombres de columnas están en la segunda.
    df = pd.read_excel(url, header=1)
    print("Primeras 5 filas del dataset:")
    print(df.head())
    print(f"Forma inicial del dataset: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza el preprocesamiento inicial del DataFrame:
    - Renombra la columna objetivo.
    - Elimina columnas no informativas (ej. 'ID').
    - Maneja valores atípicos/inválidos en columnas categóricas ('EDUCATION', 'MARRIAGE').

    Args:
        df (pd.DataFrame): El DataFrame original.

    Returns:
        pd.DataFrame: El DataFrame preprocesado sin valores atípicos/faltantes.
    """
    print("\n--- Preprocesamiento de Datos ---")
    # Renombrar la columna objetivo para mayor claridad
    df.rename({'default payment next month': 'DEFAULT'}, axis='columns', inplace=True)

    # Eliminar la columna 'ID' ya que no aporta información predictiva
    df.drop('ID', axis=1, inplace=True)
    print("Columnas después de renombrar y eliminar 'ID':")
    print(df.head())

    # Revisar valores únicos e inválidos en columnas clave
    print(f"\nValores únicos en 'SEX': {df['SEX'].unique()}")
    print(f"Valores únicos en 'MARRIAGE': {df['MARRIAGE'].unique()}")
    print(f"Valores únicos en 'EDUCATION': {df['EDUCATION'].unique()}")

    # Contar valores faltantes (NaN) - Aunque para este dataset, los "inválidos" son 0s.
    print(f"Número de valores faltantes en 'SEX': {len(df[pd.isnull(df.SEX)])}")
    print(f"Número de valores faltantes en 'MARRIAGE': {len(df[pd.isnull(df.MARRIAGE)])}")
    print(f"Número de valores faltantes en 'EDUCATION': {len(df[pd.isnull(df.EDUCATION)])}")
    print(f"Número de valores faltantes en 'AGE': {len(df[pd.isnull(df.AGE)])}")

    # Contar y filtrar filas con valores '0' en EDUCATION o MARRIAGE
    # Se considera '0' como un valor inválido o "desconocido" en estas categorías.
    invalid_education_marriage_count = len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)])
    print(f"Número de puntos de datos con '0' en 'EDUCATION' o 'MARRIAGE': {invalid_education_marriage_count}")

    # Filtrar el DataFrame para remover estas filas
    df_cleaned = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)].copy()
    print(f"Forma del dataset después de filtrar valores inválidos: {df_cleaned.shape}")

    return df_cleaned

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza el balanceo del dataset utilizando downsampling para la clase mayoritaria.
    Esto es crucial para datasets desbalanceados en problemas de clasificación.

    Args:
        df (pd.DataFrame): El DataFrame de entrada con la columna 'DEFAULT'.

    Returns:
        pd.DataFrame: El DataFrame balanceado.
    """
    print("\n--- Balanceo del Dataset (Downsampling) ---")
    # Separar las clases
    df_no_default = df.loc[df['DEFAULT'] == 0]
    df_default = df.loc[df['DEFAULT'] == 1]

    print(f"Longitud de la clase 'No Incumplió' (DEFAULT=0): {len(df_no_default)}")
    print(f"Longitud de la clase 'Incumplió' (DEFAULT=1): {len(df_default)}")

    # Downsample la clase mayoritaria ('No Incumplió') para igualar la minoritaria
    # Se elige n_samples=1000 como en el código de la primera prueba.
    # Esto asume que la clase minoritaria tiene al menos 1000 muestras.
    sample_size = min(len(df_no_default), len(df_default)) # Mejor tomar el tamaño de la clase menor
    if sample_size > 0:
        # Si la clase 'default' es menor a 1000, ajustamos el downsample_size
        downsample_size = min(1000, sample_size)
        df_no_default_downsampled = resample(df_no_default,
                                             replace=False, # Muestreo sin reemplazo
                                             n_samples=downsample_size, # Tamaño de la muestra
                                             random_state=0) # Para reproducibilidad
        df_default_downsampled = resample(df_default,
                                          replace=False, # Muestreo sin reemplazo
                                          n_samples=downsample_size, # Aseguramos el mismo tamaño
                                          random_state=0) # Para reproducibilidad

        # Fusionar los datasets downsampleados
        df_balanced = pd.concat([df_no_default_downsampled, df_default_downsampled])
        print(f"Forma del dataset balanceado (downsampled): {df_balanced.shape}")
        # Muestra la distribución de la clase objetivo en el dataset balanceado
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x='DEFAULT', data=df_balanced, palette='rocket')
        for container in ax.containers:
            ax.bar_label(container)
        plt.title("Distribución de la Clase 'DEFAULT' en el Dataset Balanceado")
        plt.xlabel("Incumplimiento (0=No, 1=Sí)")
        plt.ylabel("Número de Observaciones")
        plt.show()
    else:
        print("Advertencia: Una de las clases está vacía, no se pudo realizar el balanceo.")
        df_balanced = df # Retorna el DataFrame original si no se puede balancear

    return df_balanced

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la ingeniería de características, específicamente la codificación One-Hot
    para las columnas de historial de pago.

    Args:
        df (pd.DataFrame): El DataFrame balanceado con las características.

    Returns:
        pd.DataFrame: El DataFrame con las características codificadas.
    """
    print("\n--- Ingeniería de Características (Codificación One-Hot) ---")
    # Se excluyen 'DEFAULT' (objetivo) y otras categóricas si no se van a codificar
    # 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' no fueron codificadas en el original,
    # se asume que se usan directamente o se transformarán de otra forma.
    # Se seleccionan las columnas de historial de pago para codificación.
    columns_to_encode = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Asegurarse de que estas columnas existen antes de intentar codificar
    existing_cols_to_encode = [col for col in columns_to_encode if col in df.columns]
    
    # Crear un DataFrame con las variables independientes antes de codificar
    # Se excluyen las columnas que no se usarán como características para el modelo,
    # incluyendo 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' si no se van a codificar.
    # En el código original se eliminan, por lo que se mantienen así.
    X_features_base = df.drop(['DEFAULT', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'], axis=1).copy()
    print(f"Forma de las características antes de la codificación: {X_features_base.shape}")

    # Aplicar One-Hot Encoding a las columnas de historial de pago
    X_encoded = pd.get_dummies(data=X_features_base, columns=existing_cols_to_encode, dtype=int) # dtype=int para 0/1
    print(f"Forma de las características después de la codificación One-Hot: {X_encoded.shape}")
    print("Primeras 5 filas de las características codificadas:")
    print(X_encoded.head())
    return X_encoded

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 0) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        X (pd.DataFrame): DataFrame con las características (variables independientes).
        y (pd.Series): Serie con la variable objetivo (variable dependiente).
        test_size (float): Proporción del dataset a incluir en la división de prueba.
        random_state (int): Semilla para la reproducibilidad de la división.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) como DataFrames/Series.
    """
    print("\n--- División de Datos en Entrenamiento y Prueba ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f'Forma de X_train: {X_train.shape}')
    print(f'Forma de X_test: {X_test.shape}')
    print(f'Forma de y_train: {y_train.shape}')
    print(f'Forma de y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = "RandomForestClassifier",
                             max_depth: int = 2):
    """
    Entrena un clasificador RandomForest, evalúa su rendimiento y muestra la matriz de confusión.

    Args:
        X_train (pd.DataFrame): Características del conjunto de entrenamiento.
        y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
        X_test (pd.DataFrame): Características del conjunto de prueba.
        y_test (pd.Series): Etiquetas del conjunto de prueba.
        model_name (str): Nombre del modelo para fines de impresión.
        max_depth (int): Profundidad máxima de los árboles en el RandomForest.

    Returns:
        sklearn.ensemble.RandomForestClassifier: El modelo entrenado.
    """
    print(f"\n--- Entrenamiento y Evaluación del Modelo: {model_name} ---")
    clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    print(f"Entrenando {model_name}...")
    clf.fit(X_train, y_train)

    # Calcular la precisión general
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión (Accuracy) del modelo {model_name}: {accuracy:.2%}')

    class_names = ['No Incumplió', 'Incumplió']

    # Generar y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatriz de Confusión para {model_name}:\n{cm}")

    # Calcular el porcentaje de predicciones correctas por clase
    for i, class_name in enumerate(class_names):
        correct_predictions = cm[i, i]
        total_actual_instances = cm[i, :].sum() # Suma de la fila i para el total de instancias reales de esa clase
        if total_actual_instances > 0:
            class_accuracy = correct_predictions / total_actual_instances * 100
            print(f'Porcentaje de predicciones correctas para "{class_name}": {class_accuracy:.2f}%')
        else:
            print(f'No hay instancias reales de "{class_name}" en el conjunto de prueba para calcular precisión.')


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión para {model_name}')
    plt.show()

    return clf

def optimize_model_hyperparameters(estimator, X_train: pd.DataFrame, y_train: pd.Series,
                                   param_grid: dict, n_iter: int = 27, cv: int = 3) -> RandomForestClassifier:
    """
    Optimiza los hiperparámetros de un estimador utilizando RandomizedSearchCV.

    Args:
        estimator: El estimador (modelo) a optimizar.
        X_train (pd.DataFrame): Características del conjunto de entrenamiento.
        y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
        param_grid (dict): Diccionario de distribuciones de parámetros a muestrear.
        n_iter (int): Número de iteraciones (muestras de parámetros) a probar.
        cv (int): Número de pliegues (folds) para la validación cruzada.

    Returns:
        sklearn.ensemble.RandomForestClassifier: El mejor estimador encontrado.
    """
    print("\n--- Optimización de Hiperparámetros con RandomizedSearchCV ---")
    rf_random = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        random_state=0,
        verbose=1, # Muestra el progreso de la búsqueda
        n_jobs=-1, # Usa todos los núcleos de CPU disponibles
    )

    print("Iniciando la búsqueda aleatoria de hiperparámetros...")
    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_params_
    best_estimator = rf_random.best_estimator_

    print(f'\nMejores parámetros encontrados: {best_params}')
    print(f'Mejor estimador (modelo) es: {best_estimator}')

    return best_estimator

# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    # --- Configuración de URL del Dataset ---
    DATA_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UEym8G6lwphKjuhkLgkXAg/default%20of%20credit%20card%20clients.xls'

    # --- Parámetros de Clasificador y Búsqueda ---
    INITIAL_RF_MAX_DEPTH = 2
    
    RANDOM_SEARCH_PARAM_GRID = {
        'max_depth': [3, 4, 5, 6, 7], # Ampliar el rango para una mejor búsqueda
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4, 5],
    }
    RANDOM_SEARCH_N_ITER = 50 # Aumentar las iteraciones para una búsqueda más exhaustiva
    RANDOM_SEARCH_CV_FOLDS = 5 # Aumentar folds para una validación cruzada más robusta

    # --- Flujo del Análisis ---

    # 1. Cargar datos
    df_raw = load_data(DATA_URL)

    # 2. Preprocesar datos
    df_cleaned = preprocess_data(df_raw)

    # 3. Balancear el dataset
    df_balanced = balance_dataset(df_cleaned)

    # 4. Ingeniería de características (codificación One-Hot)
    # y = df_downsample['DEFAULT'].copy() -> se define y aquí
    y_target = df_balanced['DEFAULT'].copy()
    X_features_encoded = feature_engineering(df_balanced) # X es el resultado de esta función

    # 5. Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(X_features_encoded, y_target, test_size=0.3, random_state=0)

    # 6. Entrenar y evaluar el modelo base (RandomForestClassifier)
    # Se inicializa un clasificador base para RandomizedSearchCV
    base_rf_classifier = RandomForestClassifier(random_state=0)
    
    # Evaluar un modelo inicial con max_depth para tener una referencia
    initial_clf_rf = train_and_evaluate_model(
        X_train, y_train, X_test, y_test,
        model_name=f"RandomForest (Depth={INITIAL_RF_MAX_DEPTH})",
        max_depth=INITIAL_RF_MAX_DEPTH
    )

    # 7. Optimizar hiperparámetros
    best_rf_classifier = optimize_model_hyperparameters(
        base_rf_classifier, # Usamos el estimador base aquí, RandomizedSearchCV ajustará los hiperparámetros
        X_train, y_train,
        RANDOM_SEARCH_PARAM_GRID,
        n_iter=RANDOM_SEARCH_N_ITER,
        cv=RANDOM_SEARCH_CV_FOLDS
    )

    # 8. Evaluar el modelo con los mejores hiperparámetros
    print("\n--- Evaluación del Modelo Optimizado ---")
    final_y_pred = best_rf_classifier.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_y_pred)
    print(f'Precisión (Accuracy) del Modelo Optimizado: {final_accuracy:.2%}')

    class_names_final = ['No Incumplió', 'Incumplió']
    final_cm = confusion_matrix(y_test, final_y_pred)
    print(f"\nMatriz de Confusión para el Modelo Optimizado:\n{final_cm}")

    for i, class_name in enumerate(class_names_final):
        correct_predictions = final_cm[i, i]
        total_actual_instances = final_cm[i, :].sum()
        if total_actual_instances > 0:
            class_accuracy = correct_predictions / total_actual_instances * 100
            print(f'Porcentaje de predicciones correctas para "{class_name}" (Optimizado): {class_accuracy:.2f}%')
        else:
            print(f'No hay instancias reales de "{class_name}" en el conjunto de prueba para calcular precisión (Optimizado).')

    ConfusionMatrixDisplay.from_estimator(
        best_rf_classifier,
        X_test,
        y_test,
        display_labels=class_names_final,
        cmap=plt.cm.Blues,
    )
    plt.title('Matriz de Confusión del Modelo Optimizado')
    plt.show()

    print("\n--- Análisis de Aprobación de Créditos Completado ---")