import cv2
import matplotlib.pyplot as plt
from helpers import (
    aplicar_blackhat,
    binarizar_adaptativo,
    filtrar_caracteres
)

img_path = '../data/img_12.jpg'
img_color = cv2.imread(img_path)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Imagen transformada por blackhat
img_bh = aplicar_blackhat(img_gray, kernel_size=(15, 15))

# Imagen binarizada
thresh = binarizar_adaptativo(img_bh)

# Encontramos la patente
letras_detectadas = filtrar_caracteres(thresh)

# Mostramos los resultados
img_resultado = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
for (x, y, w, h, area) in letras_detectadas:
    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_bh, cmap='gray')
plt.title('Black Hat')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Umbral Adaptativo')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_resultado)
plt.title(f'Segmentación de letras')
plt.axis('off')

plt.tight_layout()
plt.show()

# Si querés ver los recortes individuales de las letras:
if len(letras_detectadas) > 0:
    plt.figure(figsize=(10, 2))
    for i, (x, y, w, h, area) in enumerate(letras_detectadas):
        recorte = img_color[y:y+h, x:x+w]
        plt.subplot(1, len(letras_detectadas), i+1)
        plt.imshow(recorte, cmap='gray')
        plt.axis('off')
    plt.show()