Flujo de trabajo del proyecto

1\.	Desarrollo local (Python + Flask)

o	La aplicación se programa en Python usando Flask, que es el marco para crear la API.

o	En el computador se prueban los cambios antes de publicarlos.

2\.	Control de versiones (GitHub)

o	El código se guarda en un repositorio de GitHub.

o	Cada vez que se hace un cambio, se sube (push) al repositorio para mantener el historial y compartirlo fácilmente.

3\.	Servidor de producción (Render)

o	Render es la plataforma que recibe el código desde GitHub y lo convierte en un servicio accesible desde internet.

o	Cada vez que subimos cambios a GitHub, Render los detecta y actualiza la aplicación automáticamente.

4\.	Ejecución en producción (Gunicorn)

o	Render usa Gunicorn, un servidor que permite que varias personas usen la API al mismo tiempo de manera estable y rápida.

o	Gunicorn se encarga de “servir” la aplicación Flask en internet.

5\.	Resultado final

o	La API queda disponible en una dirección web (ejemplo: https://mi-api.onrender.com).

o	Cualquier usuario o sistema puede conectarse a esa URL para consultar o enviar datos.



