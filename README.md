# 🚀Proyecto Machine Learning

## 📌 Descripción
El objetivo de este proyecto es realizar un preprocesamiento de los datos proporcionados, seguido de la implementación de un modelo de regresión lineal y un modelo de regresión logística.

## 🗂️ Estructura del Proyecto
├── data

	├── raw
	
│   │   └── dataset_estudiantes.csv # Se encuentra el fichero de análisis proporcionado por la escuela

	├── processed
	
│   │   └── dataset_estudiantes_clean.csv # Es el resultado de aplicar 01_LimpiezaDatos_EDA.ipynb

│   │   └── df_regresion.csv # Es el resultado de aplicar 02_Preproceso.ipynb

│   │   └── df_clasificacion.csv # Es el resultado de aplicar 02_Preproceso.ipynb

│
├── models

│   │   └── modelo_regresion.pkl # Modelo de regresión lineal entrenado para predicción de nota_final

│   │   └── modelo_clasificacion.pkl # Modelo de Regresión Logística entrenado para predicción de aprobado/suspenso

│
├── notebooks

│   │   └── 01_LimpiezaDatos_EDA.ipynb   # Limpieza de datos, análisis exploratorio y detección de inconsistencias (EDA)

│   │   └── 02_Preproceso.ipynb    # Preprocesamiento de variables, codificación, escalado y gestión de outliers 

│   │   └── 03_Regresion.ipynb    # Construcción, entrenamiento y evaluación del modelo de regresión para predecir nota_final 

│   │   └── 04_Clasificacion.ipynb    # Construcción, entrenamiento y evaluación del modelo de clasificación para predecir aprobado/suspenso

├── README.md # Descripción del proyecto

## 🛠️ Instalación y Requisitos
En este proyecto es necesario utilizar:

+ Python 3.13.2 (es la versión que he utilizado para hacer los ejercicios)
+ Librerias: Pandas, Numpy, Matplotlib, Seaborn, KNNImputer, Scikit-learn, Joblib
+ Visual Studio Code

## 🎯 Criterios
- Análisis exploratorio (EDA).
- Preprocesamiento.
- Entrenamiento y validación del modelo de regresión.
- Entrenamiento y validación del modelo de clasificación.
- Informe en un archivo README
- Correcta entrega en GitHub

## 📝 Dataset
El conjunto de datos 'dataset_estudiantes.csv' trata sobre el rendimiento académico de estudiantes y tiene las siguientes características:

### Columnas del dataset
+ horas_estudio_semanal: Número de horas de estudio a la semana.
+ nota_anterior: Nota que obtuvo el alumno en la convocatoria anterior.
+ tasa_asistencia: Tasa de asistencia a clase en porcentaje.
+ horas_sueno: Promedio de horas que duerme el alumno al día.
+ edad: Edad del alumno.
+ nivel_dificultad: Dificultad del alumno para el estudio.
+ tiene_tutor: Indica si el alumno tiene tutor o no.
+ horario_estudio_preferido: Horario de estudio preferido por el alumno.
+ estilo_aprendizaje: Forma de estudio que emplea el alumno.

### Variables objetivo:
+ Regresión: nota_final (variable continua entre 0 y 100)
+ Clasificación: aprobado (variable binaria: 1 si la nota es ≥ 60, 0 en caso contrario)

## 📊 Análisis de proyecto
### 1. Introducción y Preparación de los Datos
Se ha llevado a cabo un Análisis Exploratorio de Datos (EDA) sobre el rendimiento académico de 1000 estudiantes, para aplicar un modelo de regresión lineal y un modelo de regresión logística, con el objetivo de predecir, en base a las variables analizadas, su nota final o si aprobará o no.

Se ha decidido dividir los distintos pasos en diferentes ficheros para una mejor legibilidad. Partimos del fichero 'dataset_estudiantes.csv', al que se le aplicó una limpieza de datos y se realizó el análisis exploratorio en '01_LimpiezaDatos_EDA.ipynb', dando como resultado el fichero 'dataset_estudiantes_clean.csv'.
Este último se tomará como origen en el fichero '02_Preproceso.ipynb' para realizar el preprocesamiento de los datos y su escalado. Al finalizar se guardarán dos DataFrames para su análisis 'df_regresion.csv' y 'df_clasificacion.csv'. En cada uno de ellos implementará el modelo de regresión correspondiente, los ficheros utilizados son: '03_Regresion.ipynb' y '04_Clasificacion.ipynb'

### 2. 01_LimpiezaDatos_EDA

**EDA**  

**Importar librerías**

Importamos las librerías necesarias para hacer nuestro análisis:

##### Manipulación de datos
- pandas
- numpy

##### Visualización
- matplotlib.pyplot
- seaborn

##### Normalización de texto
- unicodedata 

##### Imputación de datos faltantes
- sklearn.impute(KNNImputer)

Se utiliza la opcion 'display.max_columns' para ver todas columnas del fichero. Se carga el fichero de origen en 'df_raw' y se realiza un 'head' para ver que contiene los datos.

**Análisis Exploratorio**  

Se utiliza el atributo 'shape' para saber que el fichero contiene 1000 filas y 11 columnas. También se usa 'info' para tener una informacion general del fichero. Se ve que hay 7 columnas númericas (5 de ellas tipo float y 2 como int) y 4 categóricas. Además que las columnas 'horas_sueno', 'horario_estudio_preferido' y 'estilo_aprendizaje' poseen nulos. 

Se realiza una copia del dataFrame en uno nuevo ('df') para dejar sin modificar el fichero original. 

Se hace un 'round(2)' para las columnas numéricas ya que en la vista previa del 'head' se observaban 6 decimales, reduciéndolos a 2 para mejorar la legibilidad.

En el siguiente paso, se describen las columnas.


**Verificación de duplicados**  

Se usa el atributo 'duplicated' y 'sum' para verificar duplicados; no se detectaron registros duplicados.


**Verificación y manejo de nulos**

Se hace un 'isna().sum' para identificar los valores nulos. Como se mencionó antes, existen nulos en tres columnas: 
- 'horas_sueno' (150 nulos)
- 'horario_estudio_preferido' (100 nulos)
- 'estilo_aprendizaje' (50 nulos).


**Identificar variables categóricas y numéricas**

Se indentifican las columnas categóricas y las numéricas para aplicar funciones específicas más adelante.

- Columnas numéricas: ['horas_estudio_semanal', 'nota_anterior', 'tasa_asistencia', 'horas_sueno', 'edad', 'nota_final', 'aprobado']
- Columnas categóricas: ['nivel_dificultad', 'tiene_tutor', 'horario_estudio_preferido', 'estilo_aprendizaje']
	   
Para el control de nulos de las columnas 'horario_estudio_preferido' y 'estilo_aprendizaje'(categóricas), se rellenarán con el valor 'Desconocido'
Para 'horas_sueno' (numérica), se utiliza KNNImputer, preservando patrones entre variables en lugar de asignar un valor fijo.

Se verifica nuevamente que los nulos han desaparecido y se crea una función para eliminar tildes en 'nivel_dificultad' y 'tiene_tutor', evitando problemas futuros.


**Variables numéricas**

**Estadísticas descriptivas** 
 
Se realiza un 'describe().T' para mostrar las estadísticas descriptivas de forma legible. No se detecta nada fuera de lo común.


**Visualizaciones**

Se realiza un 'histplot' para cada variable numérica (25 grupos):

*1. Distribución de horas_estudio_semanal:*
Distribución normal, la mayor parte de estudiantes dedica entre 7 y 14 horas semanales. Destaca un grupo de estudiantes que estudia solo 1 hora.

*2. Distribución de nota_anterior:*
Distribución normal, concentrada entre 60 y 80 puntos. Destaca un grupo con la nota máxima.

*3. Distribución de tasa_asistencia:*
La mayoría tiene entre 65% y 85%, destacando un grupo con 100% de asistencia.

*4. Distribución de horas_sueno:*
La mayoría duerme entre 6,5 y 8,5 horas. Sorprende que un cuarto de los estudiantes duerma menos de 6,5 horas. Dos grupos extremos duermen 4 y 10 horas.

*5. Distribución de edad:*
Distribución relativamente uniforme entre 18 y 29 años, con ligera mayoría de 18 años.

*6. Distribución de nota_final:*
Distribución centrada entre 62% y 80%. Más aprobados que suspensos.

*7. Distribución de aprobado:*
Mayor número de aprobados que suspensos.


**Variables categóricas**

**Estadísticas descriptivas** 

Se aplica un 'describe().T'. Todas las columnas tienen 1000 registros:

- 'nivel_dificultad', 3 categorías, predominando "Medio" (504 estudiantes)
- 'tiene_tutor', 2 categorías, predominando "No" (597 estudiantes).
- 'horario_estudio_preferido', 4 categorías, predominando "Noche" (344 estudiantes).
- 'estilo_aprendizaje', 5 categorías, predominando "Visual" (363 estudiantes).


**Valores únicos y frecuencias**

Se itera sobre 'cat_cols' para mostrar valores únicos y frecuencia de cada uno.


**Visualizaciones**

Se genera un gráfico de barras para cada variable categórica, ajustando dinámicamente la anchura (mín. 7, máx. 25):

*1. Distribución de nivel_dificultad:* Predomina "Medio", seguido de "Fácil" y "Difícil".

*2. Distribución de tiene_tutor:* Más estudiantes sin tutor.

*3. Distribución de horario_estudio_perfecto:* Predomina "Noche"; hay un grupo de 100 estudiantes con valor "Desconocido".

*4. Distribución de estilo_aprendizaje:* Predomina "Visual", seguido de "Auditivo", "Kinestésico", "Lectura/Escritura" y "Desconocido" (50 estudiantes).


**Matriz de Correlación**

Se aplica la matriz de correlación para las variables numéricas utilizando el atributo 'corr', mostrando únicamente la parte triangular para evitar duplicados. No hay ninguna correlacion alta o media alta.Hay algunas correlaciones moderadas:
- Entre 'aprobado' y 'nota_final', lo cual tiene sentido: a mayor 'nota_final', más probable es que el estudiante esté aprobado.
- Entre 'nota_final' y 'horas_estudio_semanal', indicando que a mayor número de horas de estudio semanal, mayor suele ser la nota.
- Entre 'nota_final' y 'nota_anterior', lo que sugiere que los alumnos que aprobaron el examen anterior tienen más probabilidades de aprobar el examen final.
- Entre 'horas_estudio_semanal' y 'nota_anterior', siguiendo la misma lógica de la 'nota_final'.

Existen algunas correlaciones bajas entre otras variables y algunas correlaciones negativas, pero son prácticamente nulas.


**Relaciones Cruzadas**


**1. Variable objetivo nota_final**


***Variables numéricas***

Se genera un gráfico de dispersión por cada variable numérica. En el eje x se encuentra la variable de estudio, en el eje y la variable objetivo 'nota_final'. Además, se ha incluido la variable 'aprobado' con otro color para visualizar mejor la relación entre tres variables.

*1. Horas_estudio_semanal:*
Se observa una gran concentración de estudiantes con entre 7 y 14 horas de estudio semanal, como se mencionó anteriormente. Todos los estudiantes que han estudiado más de 16 horas semanales aprobaron. Llama la atención que varios estudiantes que estudiaron solo 1 hora semanal también aprobaron.

*2. Nota_anterior:*
La mayoría de los estudiantes que aprobaron el examen anterior también aprobaron el examen final. Destaca que algunos alumnos que suspendieron (nota inferior a 60) en el examen anterior lograron aprobar el examen final. También hay casos de alumnos que aprobaron el primer examen pero suspendieron el final.
Podemos concluir que si un estudiante no supera el 60% en ambos exámenes, no aprueba. En general, más alumnos aprobaron el examen final que el anterior.

*3. Tasa_asistencia:*
La mayoría de los aprobados tienen una asistencia entre 65% y 85%. Todos los alumnos con asistencia del 100% aprobaron. Entre los que menos asistieron, hubo más aprobados que suspensos.

*4. Horas_sueno:*
Los aprobados suelen dormir entre 6,5 y 8,5 horas. Sorprende que algunos estudiantes que duermen solo 4 horas aprobaron, al igual que los que duermen 10 horas.

*5. Edad:*
No se observan patrones destacables; los valores están bastante equilibrados.

*6. Aprobado:*
Para aprobar se requiere superar el 60% de la nota final. Se confirma que hay más aprobados que suspensos.


***Variables categóricas***

Se crean gráficos de bigotes por cada columna. En el eje x está la variable de estudio y en el eje y la variable objetivo. Se agrupan por la variable de estudio y se ordenan según la media de la variable objetivo.

*1. Nivel_dificultad:*
Todos los estudiantes tienen 'nota_final' superior al 60%, lo que indica que la mayoría aprueba según su categoría. La media más alta se encuentra en los alumnos con nivel de dificultad 'Fácil'.

*2. Tiene_tutor:*
Los alumnos que tienen tutor presentan una mejor 'nota_final' que los que no lo tienen. Sin embargo, algunos alumnos sin tutor lograron notas más altas que la media de los que sí lo tenían, apareciendo como outliers.

*3. Horario_estudio_preferido:*
Todas las categorías muestran una media de notas similar. Destaca que los alumnos que estudian por la tarde obtuvieron la mejor nota media final. Algunos registros se clasifican como 'desconocido', y sería recomendable reasignarlos para confirmar si los resultados se mantienen.

*4. Estilo_aprendizaje:*
Resultados similares a los anteriores. El mejor promedio se observa en la categoría 'Lectura/Escritura'.


**2. Variable objetivo aprobado**

Se genera un histograma si las variables son numéricas o un countplot si son categóricas. En el eje x está la variable de estudio y en el eje y la variable objetivo.

*1. Horas_estudio_semanal:*
Los estudiantes que dedican menos horas al estudio son los que más suspenden. Algunos casos excepcionales, incluso estudiando 15 horas, suspendieron, pero no es lo habitual. Todos los estudiantes que estudian más de 15 horas aprueban.

*2. Nota_anterior:*
Muchos alumnos que suspendieron el examen anterior lograron aprobar el final. La mayoría de los que suspendieron el examen anterior también suspendieron el examen final. Algunos casos excepcionales, como un estudiante con nota máxima en el examen anterior, suspendieron el examen final. Los grupos donde la mayoría suspendió el primer examen tienden a no aprobar el segundo examen.

*3. Tasa_asistencia:*
A mayor asistencia, menor proporción de suspensos. Se observa un pequeño repunte de suspensos entre los alumnos con 100% de asistencia. La mayor concentración de suspensos se encuentra en tasas de asistencia entre 60-65%.

*4. Horas_sueno:*
Los estudiantes que duermen entre 6,5 y 8 horas son los que más aprueban. Sorprenden algunos casos de estudiantes que duermen 4 horas y aprueban, similar a los que duermen 10 horas. Curiosamente, los estudiantes que duermen alrededor de 7 horas son los que más suspenden.

*5. Edad:*
Los estudiantes de 29 años concentran la mayor cantidad de aprobados y suspensos. El resto de edades está bastante equilibrado.

*6. Nivel_dificultad:*
Los estudiantes con nivel de dificultad 'Medio' concentran la mayoría de aprobados y suspensos. Esto refleja que la categoría 'Medio' tiene más alumnos.

*7. Tiene_tutor:*
Más aprobados se encuentran entre los alumnos que no tienen tutor. Sin embargo, entre los que sí tienen tutor, la proporción de suspensos es menor.

*8. Horario_estudio_preferido:*
Los alumnos que estudian por la 'Tarde' y la 'Noche' concentran la mayoría de aprobados y suspensos, siendo ligeramente superior el grupo nocturno. Los de la mañana también tienen una buena tasa de aprobados.

*9. Estilo_aprendizaje:*
Destaca que los estudiantes con estilo de aprendizaje 'Visual' tienen un mejor desempeño relativo, tanto en aprobados como en suspensos. Es interesante que esta categoría no aparecía en el primer lugar cuando la variable objetivo era 'nota_final'.

*10. Nota_final:*
Los estudiantes que superan el 60% aprueban; los que no, suspenden. Se observa que los estudiantes que obtienen exactamente 60% presentan más suspensos que aprobados, lo que indica que se evalúa alguna otra variable para decidir la aprobación final.


**Análisis de Inconsistencias**

Se añadieron reglas para identificar posibles inconsistencias y así eliminarlas para no afectar el análisis. Se creó una lista vacía para ir almacenando las inconsistencias detectadas. Las reglas fueron:

1. 'Horas_estudio_semanal' no podía superar las 60 horas semanales.
2. 'Nota_final' y las 'Nota_anterior' debían estar entre 0 y 100.
3. 'Hora_sueno' debía estar entre 0 y 24 horas.
4. 'Edad' debía estar entre 18 y 100 años.

Al finalizar, la lista de inconsistencias resultó vacía, confirmando que no había datos fuera de rango.


**Export fichero limpio**

Se exporta el fichero limpio para los siguientes pasos del análisis. Se guarda con el nombre 'dataset_estudiantes_clean.csv' en la ruta ../data/processed/.

### 3. 02_Preproceso

**Importar librerías**  
Importamos las librerías necesarias para hacer nuestro análisis:

###### Manipulación de datos
- pandas
- numpy
- math

###### Estadística y normalización
- scipy.stats (zscore)
- sklearn.preprocessing (MinMaxScaler, OneHotEncoder, OrdinalEncoder)

###### Visualización
- matplotlib.pyplot
- seaborn

Se utiliza la opcion 'display.max_columns' para ver todas columnas del fichero. Se Carga el fichero de origen en 'df_raw' y se realiza un 'head' para ver que contiene los datos.


**Carga de datos**  

Se cargan los datos de fichero que se exportó anteriormente con los datos limpios. El fichero se llama 'dataset_estudiantes_clean.csv'


**Gestión de Outliers**

Se realizado un Boxplot por cada variable númerica del fichero manteniendo la columna 'Aprobado'. Al ser una variable binaria se decidió no tenerla en cuenta en el estudio de los outliers, porque no tiene sentido visualmente; sin embargo, en el siguiente paso es útil para ver si los outliers aprueban. Se calcula el número de gráficos y el número de columnas, ajustando el tamaño de manera dinámica.

Al analizar los outliers no vemos que nos esten sesgando los datos de estudio, por lo que entendemos que no va a afectar a la hora de aplicar algún método de marchine learning.


**Detección de outliers mediante el método IQR**

Se crea un diccionario para almacenar toda la información de forma organizada. Se calcula el índice intercuartílico para visualizar los outliers. Se definen dos límites (1.5 por encima del cuartil 3 y 1.5 por debajo del cuartil 1), y los datos que quedan fuera de estos límites se consideran outliers. Sólo tenemos 3 variables con outliers:

*1. horas_estudio_semanal:*  
Aparecen 4 estudiantes que dedicaron más horas de estudio. Al analizar las otras variables, no parece que esta información afecte a los datos. Hay personas que necesitan más tiempo para estudiar o retener la información. Aunque dediques más horas, no garantiza la máxima nota, pero sí que puedes aprobar. 

*2. tasa_asistencia:*  
Se encuentran otros 4 estudiantes con la tasa de asistencia más baja. No parece que sesguen los datos: 3 de ellos aprobaron y solo 1 suspendió. También se observa que este último dedicó solo 1 hora semanal de estudio.

*3. nota_final:*  
Aparecen outliers en ambos extremos: 2 con la nota más alta y 3 con las notas más bajas. Estos últimos dedicaron pocas horas de estudio comparados con los compañeros que aprobaron. Como mencionamos, estos outliers no afectan significativamente al análisis.


**Detección de outliers mediante el método Z-score**

Otra forma de analizar los outliers es utilizando el 'Z-score'. Se buscan los que están a más de 3 desviaciones estándar de la media y aparecen 4 outliers: 
- 2 estudiantes dedicando 25 'horas_estudio_semanal'.
- 2 alumnos con las 'nota_final' más bajas con un 30% y un 40%.

Como se dijo en caso anterior no sse van a eliminar esos registros porque no están afectando a los datos de estudio.


**Regresión**

*La variable objetivo es nota_final*
Se ha creado una copia del dataFrame 'df_reg' para trabajar con ese y no modificar el 'df_clean'. Además se define la variable objetivo 'nota_final'.


**Codificación**

El objetivo es convertir las variables categóricas en valores numéricos.

*1. OrdinalEncoder*
Se utiliza 'OrdinalEncoder' para las variables con orden: 'nivel_dificultad' y 'tiene_tutor'. Se convierten los valores de cada columna a números, asegurando que 'nivel_dificultad' empiece desde 1.

*2. OneHotEncoder*
Se utiliza 'OneHotEncoder' para variables nominales sin orden: 'horario_estudio_preferido' y 'estilo_aprendizaje'. Con fit, se aprenden todas las categorías y se les asigna un valor 0/1. Luego se recortan los nombres de los encabezados para que el dataframe sea legible. Se crea un dataframe 'encoded_df' con el mismo índice que 'df_reg' para combinarlo fácilmente. Finalmente, se eliminan las columnas originales y se añaden las nuevas codificadas a 'df_reg'.

Para mantener el preprocesamiento del estudio de los dos modelos (regresión y clasificación) se realiza una copia del dataframe antes de hacer el siguiente paso.


**Escalado**


*MinMaxScaler*
Se realiza un escalado de los datos para que todas la variables se encuentren en la misma escala entre 0 y 1 menos la variable objetivo 'nota_final'


**Clasificación**

Se realiza lo mismo que en el paso anterior pero ahora excluyendo la variable 'aprobado'


**Guardar Dfs preprocesados**

Se guarda 'df_regresion.csv' en la carpeta '../data/processed' para su posterior uso. Se hace lo mismo con 'df_clasificacion.csv'

### 4. 03_Regresión

**Importar librerías**  
Importamos las librerías necesarias para hacer nuestro análisis:

##### Manipulación de datos
- pandas
- numpy

##### Opcional: ver todas las columnas del DataFrame
pd.set_option('display.max_columns', None)

##### Visualización
- matplotlib.pyplot
- seaborn 

##### Guardado y carga de modelos
- joblib

##### División de datos y validación
- sklearn.model_selection (train_test_split)

##### Modelos de regresión:
##### Regresión lineal básica
- sklearn.linear_model (LinearRegression)

##### Regresiones con regularización
- sklearn.linear_model (Ridge, Lasso, ElasticNet)

##### Modelo no lineal y flexible
- sklearn.ensemble (RandomForestRegressor)

##### Transformaciones y generación de características
- sklearn.preprocessing (PolynomialFeatures)

##### Métricas de evaluación
- sklearn.metrics (r2_score, mean_squared_error, mean_absolute_error)


**Carga de datos**  

Se cargan los datos preprocesados del fichero 'df_regresion.csv'

*Separación del conjunto de datos*
Se separa la variable 'nota_final' de las variables predictoras. Luego, se divide el dataset en entrenamiento (80%) y prueba (20%) con 'train_test_split':
- Tamaño del conjunto de entranmiento: (800, 17)
- Tamaño del conjunto de prueba: (200, 17)


**Entrenamiento del modelo**  

Se entrena el modelo con 'LinearRegression', pasando las variables predictoras y la respuesta. Luego se predicen nuevos valores con 'predict'.


**Validación del modelo**  

*Comparación con Scatterplot*
Se grafica la predicción frente a los valores reales. Se añade una línea diagonal de referencia; idealmente todos los puntos azules deberían estar sobre ella. Se observa una ligera desigualdad.

*Comparación de distribuciones*
Se comparan las distribuciones reales y predichas mediante histogramas. Los resultados son similares, pero se observan algunas diferencias.


**Residuos**  

Los residuos son la diferencia entre valores reales y predichos. Se visualizan en scatterplots e histogramas. Idealmente deberían centrarse en 0. Se observa un pico en torno a 0 y errores menos frecuentes a mayor magnitud, lo cual es normal.


**Importancia de las caracterísitcas** 

Se calcula la importancia mediante los coeficientes lineales. Destacan:
- 'aprobado': mayor coeficiente, coherente con los estudiantes aprobados que tienen 'nota_final' > 60%.
- 'horas_estudio_semanal': más horas → mayor nota.
- 'nota_anterior': estudiantes que aprobaron antes tienden a aprobar ahora.
- 'tasa_asistencia': influye en la nota final.

Se añade un gráfico de barras para visualizar la importancia de manera clara.


**Métricas**

Se evalúa desempeño en entrenamiento y prueba:
- *R²*: no es muy alto por lo que hay bastante variabilidad que el modelo no captura. La diferencia train y test no es muy grande, indica que no hay un sobreajuste fuerte, aunque el modelo tampoco tiene mucha capacidad predictiva.
- *MAE y RMSE*: Los errores son homogéneos y no hay valores extremos desproporcionados.

Dado este comportamiento, se decidió entrenar un modelo más potente para intentar mejorar las métricas: un 'RandomForestRegressor'. Para aumentar su capacidad predictiva, se generaron interacciones lineales entre las variables originales utilizando 'PolynomialFeatures'. Esto significa que se crearon nuevas variables que representan el producto de pares de variables originales, sin elevar al cuadrado ninguna de ellas, permitiendo al modelo capturar relaciones conjuntas que no se reflejan usando solo las variables individuales.

Posteriormente, se entrenó el 'RandomForest' con parámetros optimizados para reducir el sobreajuste. Esta combinación de interacciones lineales y RandomForest optimizado permitió capturar relaciones más complejas entre variables, mejorando el desempeño predictivo mientras se mantenía un equilibrio entre ajuste y generalización. Los resultados obtenidos fueron:

- *Train R²*: Aumenta de 0.54 a 0.66, mostrando que el modelo explica más varianza en el entrenamiento gracias a las interacciones y a la mayor capacidad del RandomForest.
- *Test R²*: Se mantiene aproximadamente igual (0.48 → 0.47), lo que indica que la capacidad de generalización no mejoró significativamente; el límite parece estar dado por la información disponible en las features.
- *Train MAE/RMSE*: Disminuyen, lo que refleja que el modelo se ajusta mejor a los datos de entrenamiento.
- *Test MAE/RMSE*: Ligeramente mayores que el modelo lineal, reflejando un pequeño sobreajuste.

En conclusión, el modelo mejorado es más potente y flexible, pero la información contenida en las features actuales sigue limitando el R² en el conjunto de prueba.


**Entrenamiento final**

Se entrena el modelo final utilizando todo el conjunto de datos, de manera que aproveche toda la información disponible para ajustar los coeficientes. Una vez entrenado, se guarda el modelo en un archivo llamado 'modelo_regresion.pkl' usando joblib.dump. Esto permite preservar el modelo entrenado y reutilizarlo en producción sin necesidad de volver a entrenarlo, asegurando que las predicciones futuras se realicen con el mismo modelo.

Además, Se probaron varios modelos de regresión lineal con técnicas de regularización (Ridge, Lasso y ElasticNet) para mejorar la capacidad predictiva del modelo inicial.

- Ridge: mantuvo resultados similares al modelo lineal simple, con R² de test 0.49 y errores MAE/RMSE casi iguales.
- Lasso: mostró resultados comparables, ligeramente inferiores en train, pero con R² de test 0.49, sin mejora significativa.
- ElasticNet: tuvo un desempeño menor tanto en entrenamiento como en test (R² de test 0.45), indicando que no capturaba mejor las relaciones de las variables.

En general, ninguna de estas técnicas de regularización superó el rendimiento del modelo lineal simple. Por ello, se decidió finalmente utilizar el RandomForestRegressor optimizado con interacciones, que mostró un mejor desempeño en entrenamiento (R² train 0.66) y mantuvo una generalización estable en test (R² test 0.47), ofreciendo un equilibrio entre ajuste y capacidad predictiva superior a los modelos lineales.


**Resultados**

El modelo lineal simple explicó parte de la varianza (R² test ≈ 0.49).
El RandomForest con interacciones mejoró el ajuste en entrenamiento (R² train ≈ 0.66), pero el R² en test se mantuvo limitado (≈ 0.47).
Las variables más influyentes fueron: 'aprobado', 'horas_estudio_semanal', 'nota_anterior' y 'tasa_asistencia'.

### 5. 04_Clasificacion

**Importar librerías**  
Importamos las librerías necesarias para hacer nuestro análisis:

##### Manipulación de datos
- pandas

##### Opcional: ver todas las columnas del DataFrame
pd.set_option('display.max_columns', None)

##### Visualización
- matplotlib.pyplot

##### Guardado y carga de modelos
- joblib

##### División de datos y validación
- sklearn.model_selection (train_test_split)

##### Modelos de regresión
- sklearn.linear_model (LogisticRegression)

##### Métricas de evaluación
- sklearn.metrics (confusion_matrix, ConfusionMatrixDisplay)
- sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score)


**Carga de datos**  

Se cargan los datos de fichero que se exportó anteriormente con el preprocesamiento hecho, el fichero se llama 'df_clasificacion.csv'

*Separación del conjunto de datos*
Se separa la variable objetivo 'aprobado' del resto de variables predictoras.
A continuación, se divide el dataset en entrenamiento (80%) y prueba (20%) utilizando la función 'train_test_split'. Los tamaños resultantes son:
- Tamaño del conjunto de entranmiento: (800, 17)
- Tamaño del conjunto de prueba: (200, 17)


**Entrenamiento del modelo**  

Se entrena el modelo de regresión logística usando el conjunto de entrenamiento, pasando como argumentos las variables predictoras y la variable objetivo 'aprobado'. Después, se obtienen las predicciones con 'predict. Asimismo, se utiliza 'predict_proba' para obtener las probabilidades de que cada estudiante apruebe o suspenda, lo que permite analizar la certeza de las predicciones del modelo.


**Validación del modelo**  

Se utiliza una matriz de confusión para evaluar el rendimiento. En el conjunto de prueba de 200 estudiantes:
- El modelo acertó con 192 estudiantes: 185 que aprueban y 7 que suspenden correctamente.
- Hubo 8 falsos positivos, es decir, estudiantes que suspendieron pero que el modelo predijo que aprobarían.

Esta matriz permite identificar errores específicos del modelo y analizar si existe un sesgo hacia la clase mayoritaria (aprobados).


**Métricas**

Se calculan las métricas de desempeño tanto para el conjunto de entrenamiento como para el de prueba:
- *Train Accuracy*: El modelo acierta en un 92% de los casos de entrenamiento
- *Train Precission*: Cuando predice positivo, acierta en un 92%
- *Train Recall*: Detecta todos los positivos reales
- *Train F1-score*: Muy buen balance entre precisión y recall.
- *Test Accuracy*: En los datos nuevos, el modelo funciona incluso mejor con un 96%
- *Test Precission*: Altísima precisión en los positivos predichos.
- *Test Recall*: Mantiene recall, no se le escapa ningún positivo real
- *Test F1-score*: ha sido de un 98%, lo que confirma el excelente equilibrio.

En conclusión, no se observa sobreajuste: el modelo generaliza bien. El hecho de que en test el rendimiento sea incluso un poco mejor puede deberse a que la muestra de prueba es más “fácil” de clasificar o simplemente a la variabilidad estadística (suerte en el split).

Se comprueba que la variable está desbalanceada, existe un 89.8% de estudiantes que aprueban frente al 10.2% que no, por lo que suelen afectar a las métricas y se va a hacer que se ponderen en función de como de representativas sea cada una de esas categorías. 
- *Accuracy*: Sigue siendo alto (92% en train, 96% en test)
- *Precision*: En train es 0.93 y en test 0.96. Esto significa que, en promedio, cuando el modelo predice una clase (incluyendo las minoritarias), acierta muy bien. La ponderación hace que las clases con más ejemplos pesen más en el cálculo.
- *Recall*: También muy alto (0.92 en train, 0.96 en test). Esto indica que el modelo detecta correctamente la mayoría de instancias de todas las clases, incluso de la minoritaria, ya que el promedio ponderado refleja el recall global ajustado por frecuencia.
- *F1-score*: El balance entre precisión y recall es sólido (0.90 en train y 0.95 en test). Al ser ponderado, este valor muestra que el modelo logra un equilibrio general en todas las clases, sin dejar que la clase mayoritaria 'tape' los errores en la minoritaria.

El modelo generaliza muy bien: las métricas en test son incluso ligeramente mejores que en train porque no hay sobreajuste.
Gracias a la ponderación, se confirma que el modelo mantiene buen rendimiento incluso con clases desbalanceadas, no solo optimizando para la clase mayoritaria.
El hecho de que train tenga un F1 un poco menor (0.90 vs 0.95 en test) probablemente se deba a variabilidad en el split (quizás en train había más ejemplos complicados de la clase minoritaria).


**Importancia de las caracterísitcas** 

Se mide la importancia de las características utilizando los coeficientes lineales del modelo de regresión logística.

- La variable más influyente es 'nota_final', lo que tiene sentido, ya que la mayoría de los estudiantes que superan el 60% de la nota final aprueban.
- Otras variables también aportan información, aunque con menor peso.

Además se añade también un gráfico de barras para ver el resultado de manera más visual.


**Entrenamiento final**

Tas entrenar el modelo de clasificación con todo el conjunto de datos, se guardó en 'modelo_clasificacion.pkl' mediante 'joblib.dump', lo que permite reutilizarlo en producción sin necesidad de volver a entrenarlo.


**Comparativa de variantes de regresión logística con regularización**

Se evaluaron tres variantes para mejorar la capacidad predictiva y la estabilidad:

*1. Logistic (L2 – Ridge)*
Presenta el mejor desempeño global. En entrenamiento alcanza un accuracy de 0.92 y un F1 de 0.90, mientras que en prueba mejora hasta 0.96 en accuracy y 0.95 en F1. Estos valores indican que el modelo generaliza muy bien y ofrece un equilibrio adecuado entre precisión y recall.

*2. Logistic (L1 – Lasso)*
Obtiene un rendimiento algo inferior, con accuracy de 0.89 en entrenamiento y 0.92 en prueba. El F1 en prueba se sitúa en 0.89, lo que refleja un comportamiento correcto pero menos robusto que Ridge. Su principal ventaja es la capacidad de selección automática de variables, lo que puede ser útil en escenarios con muchas características irrelevantes.

*3. Logistic (ElasticNet)*
Muestra resultados similares a Lasso: accuracy de 0.89 en entrenamiento y 0.92 en prueba, con un F1 de 0.89. Aunque combina las propiedades de L1 y L2, en este caso no supera el rendimiento de Ridge y presenta menor precisión en entrenamiento (0.79), lo que sugiere un ajuste menos equilibrado.

En conclusión, entre las tres variantes evaluadas, la regresión logística con regularización L2 (Ridge) se posiciona como la mejor opción, al ofrecer el mayor rendimiento en el conjunto de prueba (Accuracy 0.96, F1 0.95) y una excelente capacidad de generalización.


**Resultados**

El modelo de regresión logística alcanzó un rendimiento excelente.
Accuracy: 92% en train y 96% en test.
Precision / Recall / F1-score: Todos muy altos, incluso con la clase desbalanceada (89.8% aprobados vs 10.2% suspensos).
La variable más influyente fue la nota_final, lo cual es coherente.

## 🧠 Conclusión

1. El preprocesamiento de los datos (limpieza, imputación de nulos, escalado y codificación) permitió obtener un dataset consistente y listo para aplicar modelos de machine learning.

2. El modelo de regresión lineal ofrece un rendimiento aceptable, pero limitado: predice con cierta precisión la nota final, aunque queda bastante varianza sin explicar. Modelos más complejos como RandomForest logran mejorar el ajuste en train, pero no aportan una mejora significativa en test.

3. El modelo de regresión logística funcionó de manera sobresaliente para clasificar estudiantes entre aprobados y suspensos. A pesar del desbalance de clases, el modelo generalizó muy bien, con métricas muy altas en train y test.

4. Entre las variables más influyentes destacan la nota anterior, las horas de estudio semanal y la asistencia, lo que resulta coherente con la realidad académica.

5. En conclusión, la clasificación es mucho más fiable que la regresión en este caso, dado que predecir si un estudiante aprueba es más estable que estimar su nota exacta.
   
## 🤝 Contribuciones
Las contribuciones son bienvenidas. Si deseas mejorar el proyecto, por favor abre un pull request o una issue.

## ✒️ Autores

Alejandro Pedraza

@alexPedrazaG