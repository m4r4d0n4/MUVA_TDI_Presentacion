import cv2

import time


# Ruta del archivo XML del clasificador Haar Cascade entrenado
cascade_path = 'cars.xml'

# Ruta del video de prueba
video_path = 'videos/video1.avi'

# Cargar el clasificador entrenado
trained_cascade = cv2.CascadeClassifier(cascade_path)

# Iniciar la captura de video desde el archivo
cap = cv2.VideoCapture(video_path)

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
    
    # Establecer una pausa para ralentizar la reproducción (ajusta el valor según sea necesario)
    time.sleep(0.1)
    # Mostrar el fotograma con los objetos detectados
    cv2.imshow('Detected Objects', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
