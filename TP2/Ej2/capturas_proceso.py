import cv2
import matplotlib.pyplot as plt
from helpers import (
    normalizar_imagen,
    aplicar_blackhat,
    binarizar_adaptativo,
    filtrar_contornos,
    encontrar_patente,
    cargar_templates,
    leer_caracteres_patente,
    refinar_patente
)

img_path = '../data/img_12.jpg'
img_color = cv2.imread(img_path)

img_color = normalizar_imagen(img_color, ancho_estandar=1500)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Imagen transformada por blackhat
img_bh = aplicar_blackhat(img_gray, kernel_size=(15, 15))

# Imagen binarizada
thresh = binarizar_adaptativo(img_bh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Encontramos la patente
contornos_detectados = filtrar_contornos(thresh)

mis_templates = cargar_templates("templates")

# Mostramos los resultados
img_contornos = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_resultado = img_contornos.copy()

for (x, y, w, h, area) in contornos_detectados:
    cv2.rectangle(img_contornos, (x, y), (x + w, y + h), (0, 255, 0), 2)

letras_detectadas = encontrar_patente(contornos_detectados)

# Refinamos fino la patente para cuando quedan cortadas algunas partes de una letra.
# Pasamos la original en GRIS y las coordenadas.
# La función nos devuelve las coords originales y la imagen del parche limpio para verlo.
letras_detectadas, parche_limpio = refinar_patente(img_gray, letras_detectadas)

texto_final = leer_caracteres_patente(thresh, letras_detectadas, mis_templates)
print(f"LA PATENTE DETECTADA ES: {texto_final}")

cv2.putText(img_resultado, texto_final, (letras_detectadas[0][0], letras_detectadas[0][1] - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
for (x, y, w, h, area) in letras_detectadas:
    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_bh, cmap='gray')
plt.title('Black Hat')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Umbral Adaptativo')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_contornos)
plt.title(f'Segmentación de contornos')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_resultado)
plt.title(f'Segmentación de letras')
plt.axis('off')

plt.tight_layout()
plt.show()

# Para ver los recortes individuales de las letras:
if len(letras_detectadas) > 0:
    plt.figure(figsize=(10, 2))
    for i, (x, y, w, h, area) in enumerate(letras_detectadas):
        recorte = img_color[y:y+h, x:x+w]
        plt.subplot(1, len(letras_detectadas), i+1)
        plt.imshow(recorte, cmap='gray')
        plt.axis('off')
    plt.show()