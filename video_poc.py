import cv2
import time

# Ruta del archivo XML del clasificador Haar Cascade entrenado
cascade_path = 'cars.xml'

# Ruta del video de prueba
video_path = 'videos/video1.avi'

# Ruta del video de salida con detección
output_video_path = 'output_video.avi'

# Cargar el clasificador entrenado
trained_cascade = cv2.CascadeClassifier(cascade_path)

# Iniciar la captura de video desde el archivo
cap = cv2.VideoCapture(video_path)

# Obtener las propiedades del video original
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar el objeto VideoWriter para escribir el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    # Leer un fotograma del video
    ret, frame = cap.read()

    # Salir del bucle si no hay más fotogramas
    if not ret:
        break

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar objetos en el fotograma usando el clasificador con ajustes
    objects = trained_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Ajusta el factor de escala
        minNeighbors=1,       # Ajusta el número mínimo de vecinos
        minSize=(10, 10),      # Ajusta el tamaño mínimo del objeto
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Dibujar rectángulos alrededor de los objetos detectados
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Guardar el fotograma en el video de salida
    out.write(frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
