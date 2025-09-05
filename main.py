import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# =============================
# 1. Cargar imágenes de referencia
# =============================
faces_dir = "faces/"
known_faces = {}
for file in os.listdir(faces_dir):
    if file.endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(file)[0]  # El nombre del archivo será el nombre de la persona
        known_faces[name] = os.path.join(faces_dir, file)

# =============================
# 2. Crear archivo de asistencia si no existe
# =============================
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Nombre", "Hora"])
    df.to_csv(attendance_file, index=False)

# =============================
# 3. Función para registrar asistencia
# =============================
def registrar_asistencia(nombre):
    df = pd.read_csv(attendance_file)
    if nombre not in df["Nombre"].values:  # Evitar duplicados
        hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {"Nombre": nombre, "Hora": hora}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"✅ Asistencia registrada: {nombre} a las {hora}")

# =============================
# 4. Captura de video y reconocimiento
# =============================
cap = cv2.VideoCapture(0)  # 0 = webcam por defecto

print("📷 Sistema de asistencia iniciado. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Guardar frame temporal para comparar
    cv2.imwrite("temp.jpg", frame)

    try:
        # Comparar con cada persona de la base de datos
        for nombre, path in known_faces.items():
            result = DeepFace.verify(img1_path=path, img2_path="temp.jpg", enforce_detection=False)
            if result["verified"]:
                registrar_asistencia(nombre)
                cv2.putText(frame, f"Asistencia: {nombre}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                break
    except Exception as e:
        print("⚠️ Error en reconocimiento:", e)

    cv2.imshow("Asistencia Automatizada", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Salir con tecla q
        break

cap.release()
cv2.destroyAllWindows()
