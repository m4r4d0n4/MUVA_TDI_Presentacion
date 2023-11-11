import dlib
import glob
import os

# Directorio donde se encuentran las imágenes positivas
positive_images_dir = 'positive_images/'

# Directorio donde se encuentran las imágenes negativas
negative_images_dir = 'negative_images/'

# Directorio de salida para guardar el modelo entrenado
output_model_dir = 'trained_model/'

# Crear el directorio de salida si no existe
if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)

# Crear el archivo de lista de imágenes positivas
positive_images_file = 'positive_images.txt'
with open(positive_images_file, 'w') as f:
    for img_path in glob.glob(os.path.join(positive_images_dir, '*.jpg')):
        f.write(img_path + ' 1 0 0 80 80\n')

# Crear el archivo de lista de imágenes negativas
negative_images_file = 'negative_images.txt'
with open(negative_images_file, 'w') as f:
    for img_path in glob.glob(os.path.join(negative_images_dir, '*.jpg')):
        f.write(img_path + '\n')

# Configurar y entrenar la cascada Haar
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True

# Cargar las imágenes positivas y negativas
train_images = [dlib.load_rgb_image(img_path) for img_path in glob.glob(os.path.join(positive_images_dir, '*.jpg'))]

# Cargar las anotaciones de las imágenes positivas
train_boxes = []
with open(positive_images_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            x, y, width, height = map(int, parts[1:])
            train_boxes.append([dlib.rectangle(left=x, top=y, right=x + width, bottom=y + height)])

# Verificar que la longitud de las listas coincida
if len(train_images) != len(train_boxes):
    raise RuntimeError("La longitud de las listas de imágenes y cajas no coincide.")

# Entrenar el detector
detector = dlib.train_simple_object_detector(train_images, train_boxes, options)

# Guardar el modelo entrenado
detector.save(os.path.join(output_model_dir, 'detector.svm'))

print("Entrenamiento completado. Modelo guardado en:", output_model_dir)
