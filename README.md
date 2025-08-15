## 1. Instalación requerimientos

Se recomienda usar ambiente conda o venv previamente:

conda create tarea2_venv
conda activate tarea2_venv
Luego instalar dependencias con:

pip install -r requirements.txt

## 2. Servicio de predicción

El modelo disponibilizado mediante un servicio de API corresponde a un clasificador Naive Bayes de supervivencia de pasajeros del Titanic. El modelo fue entrenado en tareas del curso "Machine Learning" sobre el dataset público de Titanic disponible en: https://www.kaggle.com/datasets/yasserh/titanic-dataset

Las características del modelo son las siguientes:

- "pclass": clase de embarque. Son 3 clases (1 = 1st, 2 = 2nd o 3 = 3rd).
- "sex": género del pasajero. Acepta solamente "male" o "female".
- "age": edad del pasajero. Es considerada un continuo, por lo cual se puede ingresar 29.5 por ejemplo.
- "sibsp": hermanos o cónyuges a bordo. Se esperan enteros positivos o cero.
- "parch": padres o hijos del pasajero a bordo. Se esperan enteros positivos o cero.
- "fare": tarifa pagada por el pasajero. Es una variable continua positiva.
- "embarked": puerto donde se embarcó el pasajero. Son 3 categorías "C", "S" o "Q".


## 3. API desplegada en Render 

API desplegada en:

https://tarea-2-desarrollo-de-proyectos-y.onrender.com

donde se debe aplicar el confidence deseado para la predicción (por defecto 0.5) y el endpoint para predicción es predict, quedando el enlace de la siguiente forma:

https://tarea-2-desarrollo-de-proyectos-y.onrender.com/predict?confidence=0.5

## 4. Inputs

El body de la petición tiene la siguiente forma para el endpoint GET "predict":

{
  "pclass": 1,
  "sex": "male",
  "age": 30,
  "sibsp": 1,
  "parch": 1,
  "fare": 500,
  "embarked": "C"
}

donde este ejemplo es un hombre en primera clase, 30 años de edad, 1 cónyuge a bordo, 1 hijo a bordo, pagó 500 dólares y se embarcó en el puerto C

## 5. Modelo cargado
Modelo entrenado disponible en model/titanic_classifier.joblib




