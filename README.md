# Prediccion de credito bancario mediante tecnicas de MLL 🗳
Este proyecto implementa un modelo de Machine Learning para predecir la probabilidad de incumplimiento de pago (default) de clientes de tarjetas de crédito. El problema clave que se maneja para resolver es ayudar a las instituciones financieras a tomar decisiones más informadas sobre la aprobación de créditos, reduciendo el riesgo de pérdidas por incumplimiento y optimizando la gestión de cartera. Se entrena y optimiza un clasificador Random Forest para este propósito, utilizando datos históricos de clientes.

## Tecnologias usadas 🐍
- pandas: manipulación, limpieza y análisis de datos tabulares (estructuración del dataset de clientes).
- numpy: operaciones numéricas eficientes, especialmente con arreglos de datos.
- matplotlib: creación de visualizaciones estáticas (gráficos de barras, matrices de confusión).
- seaborn: creación de visualizaciones estadísticas más atractivas y complejas (gráficos de conteo).
- scikit-learn (sklearn): preprocesamiento de datos (manejo de desbalance con resample, codificación One-Hot con get_dummies) y ETL.
- xlrd: dependencia necesaria para pandas para leer archivos Excel antiguos (.xls).
- warnings: controlar la visualización de advertencias durante la ejecución del script.

## Consideraciones en Instalación ⚙️
Si usamos pip:

pip install -q 

  numpy==1.26.4 \
    
  pandas==2.2.2 \
    
  matplotlib==3.9.0 \
    
  seaborn==0.13.2 \
    
  scikit-learn==1.4.2 \
    
  xlrd==2.0.1


En esta ocasion el codigo se escribio en Jupyter Notebook para Python .

## Ejemplo de uso 📎
El script realiza un flujo completo de Machine Learning, desde la carga y limpieza de datos hasta el entrenamiento y la optimización de un modelo de clasificación.
 1. Carga de Datos: El dataset de clientes de tarjetas de crédito se carga directamente desde una URL.
 2. Preprocesamiento de Datos: Renombramiento de columnas para mayor claridad; balanceo de clases utilizando un downsampling para equilibrar el número de instancias de clientes.
 3. División de Datos: El dataset se divide en conjuntos de entrenamiento y prueba (70% para entrenamiento, 30% para prueba).
 4. Entrenamiento y Evaluación del Modelo Base: un clasificador RandomForestClassifier con hiperparámetros iniciales y se visualiza la matriz de confusión para entender el rendimiento del modelo en cada clase.
 5. Optimización de Hiperparámetros (Randomized Search): utiliza RandomizedSearchCV para buscar los mejores hiperparámetros (max_depth, min_samples_split, min_samples_leaf) para el RandomForestClassifier.

## Contribuciones 🖨️
Si te interesa contribuir a este proyecto o usarlo independiente, considera:
- Hacer un "fork" del repositorio.
- Crear una nueva rama (git checkout -b feature/nueva-caracteristica).
- Realizar tus cambios y "commitearlos" (git commit -am 'Agregar nueva característica').
- Subir tus cambios a la rama (git push origin feature/nueva-caracteristica).
- Abrir un "Pull Request".

## Licencia 📜
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE (si aplica) para más detalles.
