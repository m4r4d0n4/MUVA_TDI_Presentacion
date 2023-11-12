import cv2

# Ruta del archivo XML del clasificador Haar Cascade entrenado
cascade_path = 'cars.xml'

# Ruta de la imagen de prueba
image_path = 'images/test5.jpg'

# Cargar el clasificador entrenado
trained_cascade = cv2.CascadeClassifier(cascade_path)

# Cargar la imagen de prueba
image = cv2.imread(image_path)

# Redimensionar la imagen a 320x240
resized_image = cv2.resize(image, (320, 240))

# Convertir la imagen redimensionada a escala de grises
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)


     # Detectar objetos en el fotograma usando el clasificador con ajustes
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
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Mostrar la imagen redimensionada con los objetos detectados
# Guarda la imagen procesada
cv2.imwrite('output_image.jpg', resized_image)
#cv2.imshow('Detected Objects', resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
