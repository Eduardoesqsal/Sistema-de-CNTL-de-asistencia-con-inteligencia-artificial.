Sistema de Asistencia Automatizada con Reconocimiento Facial

Este proyecto permite registrar asistencia automáticamente utilizando DeepFace y la webcam de tu computadora.

Requisitos

Windows 10 o 11

Python 3.11

Conda
 instalado

Webcam disponible

Estructura de carpetas
face_attendance_system/
│
├─ faces/               # Imágenes de referencia de cada persona
│   ├─ juan.jpg
│   ├─ maria.png
│   └─ ...
├─ main.py              # Script principal
├─ attendance.csv       # Archivo generado automáticamente
├─ requirements.txt     # Dependencias
└─ README.md


Nota: Cada archivo en faces/ debe llevar como nombre el nombre de la persona (sin espacios, extensión .jpg/.png/.jpeg).

Instalación con Conda

Abrir Anaconda Prompt y crear un entorno nuevo:

conda create -n face_env python=3.11
conda activate face_env


Instalar dependencias usando pip desde requirements.txt:

pip install -r requirements.txt

requirements.txt recomendado
numpy==1.26.1
opencv-python==4.9.0.80
matplotlib==3.7.2
tensorflow-intel==2.15
keras==2.15
deepface==0.0.93
pandas==2.2.2
flask==3.1.2
flask-cors==6.0.1
mtcnn==1.0.0
retina-face==0.0.17
requests>=2.32.0
tqdm>=4.67.1
Pillow>=11.0.0


Esta combinación de versiones evita conflictos de dependencias comunes que causan errores al usar DeepFace y OpenCV.

Uso del sistema

Agregar imágenes de referencia en la carpeta faces/.

El nombre del archivo será el nombre de la persona para registrar asistencia.

Ejecutar el script principal:

python main.py


Se abrirá la webcam y el sistema empezará a detectar rostros.

Si se reconoce a alguien, registrará la asistencia automáticamente en attendance.csv.

En la ventana de video se mostrará el nombre de la persona detectada.

Salir del sistema presionando la tecla q.

Notas importantes

El archivo attendance.csv se crea automáticamente si no existe.

Para agregar nuevas personas, solo añade sus fotos a la carpeta faces/ y reinicia el script.

Asegúrate de tener buena iluminación para un reconocimiento más preciso.
