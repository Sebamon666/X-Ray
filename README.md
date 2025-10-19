Este proyecto implementa una API web de diagnóstico automático a partir de radiografías, utilizando redes neuronales convolucionales (CNN) entrenadas con PyTorch y desplegadas en Render.

Flujo técnico

1. Entrenamiento del modelo
- Se utilizó la arquitectura ResNet-18 preentrenada en ImageNet.
- El modelo fue ajustado y evaluado con métricas como AUC, accuracy y F1-score.
- La versión final alcanzó un desempeño superior al 98 % de AUC.

2. Despliegue en Render
- El código fuente se aloja en GitHub, desde donde Render realiza los builds automáticos.
- El modelo model_resnet18.pth se carga directamente al iniciar el servidor Flask.
- La API se ejecuta con Gunicorn y expone una interfaz web simple para subir imágenes y obtener predicciones.

3. Funcionamiento
- El usuario arrastra o selecciona una radiografía (JPG/PNG).
- La imagen se procesa y se clasifica en dos categorías: NORMAL o PNEUMONIA
- Se devuelve la predicción y el nivel de confianza del modelo.

4. Tecnologías utilizadas
- PyTorch / Torchvision — entrenamiento y arquitectura ResNet-18
- Flask — API REST y servidor web
- Gunicorn — despliegue en Render
- GitHub — control de versiones
- Render — hosting de la aplicación

5.  Métricas destacadas
- AUC	      0.999
- Accuracy	0.982
- Recall	  0.996
- F1 score	0.982
