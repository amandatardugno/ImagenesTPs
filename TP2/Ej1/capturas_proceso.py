import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helpers import (
    aislar_cinta_transportadora,
    encontrar_pastillas
)

img_path = 'data/pills.png'
img = cv2.imread(img_path)

# Obtenemos las coordenadas de la cinta
coordenadas = aislar_cinta_transportadora(img, padding=10)

if coordenadas is None:
    print("No se encontró la cinta transportadora.")
    exit()
x, y, w, h = coordenadas

# Mostramos la segmentación de la cinta transportadora
plt.figure(figsize=(8, 6))

# Matplotlib usa RGB, OpenCV carga en BGR
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detección de Cinta Transportadora')

ax = plt.gca()
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
ax.add_patch(rect)

plt.axis('off')
plt.show(block=False)

img_cinta_recortada = img[y:y+h, x:x+w]

contornos_pastillas = encontrar_pastillas(img_cinta_recortada)

plt.figure(figsize=(8, 6))
cv2.drawContours(img_cinta_recortada, contornos_pastillas, -1, (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(img_cinta_recortada, cv2.COLOR_BGR2RGB))
plt.title('Detección de Pastillas')
plt.show(block=False)

areas = []
perimetros = []

for c in contornos_pastillas:
    area = cv2.contourArea(c)
    # El parámetro True indica que asumimos que el contorno es cerrado
    perimetro = cv2.arcLength(c, True) 
    
    areas.append(area)
    perimetros.append(perimetro)

plt.figure(figsize=(12, 5))

# Histograma de Áreas
plt.subplot(1, 2, 1)
plt.hist(areas, bins=15, color='skyblue', edgecolor='black', alpha=0.8)
plt.title('Distribución de Áreas', fontsize=14)
plt.xlabel('Área (px²)')
plt.ylabel('Frecuencia (Cantidad de Pastillas)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histograma de Perímetros
plt.subplot(1, 2, 2)
plt.hist(perimetros, bins=15, color='salmon', edgecolor='black', alpha=0.8)
plt.title('Distribución de Perímetros', fontsize=14)
plt.xlabel('Perímetro (px)')
plt.ylabel('Frecuencia (Cantidad de Pastillas)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()