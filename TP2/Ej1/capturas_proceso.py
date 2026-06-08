import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helpers import (
    aislar_cinta_transportadora,
    encontrar_pastillas,
    obtener_bordes,
    Pastilla
)

img_path = '../data/pills.png'
img = cv2.imread(img_path)

# Mostramos funcionamiento del Canny
edges = obtener_bordes(img)
plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap='gray')
plt.title('Detección de Bordes con Canny')
plt.axis('off')
plt.show(block=False)


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

img_cinta_recortada = img[y:y+h, x:x+w].copy()

contornos_pastillas = encontrar_pastillas(img_cinta_recortada)

plt.figure(figsize=(8, 6))
cv2.drawContours(img_cinta_recortada, contornos_pastillas, -1, (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(img_cinta_recortada, cv2.COLOR_BGR2RGB))
plt.title('Detección de Pastillas')
plt.show(block=False)

areas = []
perimetros = []
fps = []

for c in contornos_pastillas:
    area = cv2.contourArea(c)
    # El parámetro True indica que asumimos que el contorno es cerrado
    perimetro = cv2.arcLength(c, True) 
    fp = area/perimetro/perimetro
    fps.append(fp)
    areas.append(area)
    perimetros.append(perimetro)

plt.figure(figsize=(11, 7))

# Valor de factor de forma teorico
fp_circulo = 1 / (4 * np.pi)
fp_cuadrado = 1 / 16.0

# Histograma de Áreas
plt.subplot(2, 2, 1)
plt.hist(areas, bins=15, color='skyblue', edgecolor='black', alpha=0.8)
plt.title('Distribución de Áreas', fontsize=14)
plt.xlabel('Área (px²)')
plt.ylabel('Frecuencia (Cantidad de Pastillas)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histograma de Perímetros
plt.subplot(2, 2, 2)
plt.hist(perimetros, bins=15, color='salmon', edgecolor='black', alpha=0.8)
plt.title('Distribución de Perímetros', fontsize=14)
plt.xlabel('Perímetro (px)')
plt.ylabel('Frecuencia (Cantidad de Pastillas)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histograma de Factores de forma
plt.subplot(2, 2, 3)
plt.hist(fps, bins=25, color='purple', edgecolor='black', alpha=0.8)
plt.title('Distribución de Factores de Forma', fontsize=14)
plt.xlabel('Factor de Forma (A/P²)')
plt.ylabel('Frecuencia (Cantidad de Pastillas)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Valor teórico del circulo
plt.axvline(x=fp_circulo, color='red', linestyle='--', alpha=0.7)
plt.text(fp_circulo - 0.001, 0.95, 'Círculo', color='red', 
         ha='right', va='top', rotation=90, fontsize=10,
         transform=plt.gca().get_xaxis_transform())

# Valor teórico del cuadrado
plt.axvline(x=fp_cuadrado, color='blue', linestyle='--', alpha=0.7)
plt.text(fp_cuadrado + 0.001, 0.95, 'Cuadrado', color='blue', 
         ha='right', va='top', rotation=90, fontsize=10,
         transform=plt.gca().get_xaxis_transform())

# Scatter Factor de forma vs area
plt.subplot(2, 2, 4)
plt.scatter(areas, fps, color='green', alpha=0.7, edgecolor='black', s=50)
plt.title('Clustering: Área vs Factor de Forma', fontsize=14)
plt.xlabel('Área (px²)')
plt.ylabel('Factor de Forma (A/P²)')
plt.grid(linestyle='--', alpha=0.7)

# Valor teórico del circulo
plt.axhline(y=fp_circulo, color='red', linestyle='--', alpha=0.7)
plt.text(0.98, fp_circulo - 0.001, 'Círculo', color='red', 
         ha='right', va='top', fontsize=10, 
         transform=plt.gca().get_yaxis_transform())

# Valor teórico del cuadrado
plt.axhline(y=fp_cuadrado, color='blue', linestyle='--', alpha=0.7)
plt.text(0.98, fp_cuadrado + 0.001, 'Cuadrado', color='blue', 
         ha='right', va='bottom', fontsize=10, 
         transform=plt.gca().get_yaxis_transform())

# Clasificacion de formas
img_cinta_limpia = img.copy()[y:y+h, x:x+w]
img_hsv_recortada = cv2.cvtColor(img_cinta_limpia, cv2.COLOR_BGR2HSV)

lista_pastillas = []
diccionario_conteo = {}
for contorno in contornos_pastillas:
    nueva_pastilla = Pastilla(contorno, img_hsv_recortada)
    lista_pastillas.append(nueva_pastilla)

for pastilla in lista_pastillas:
    tipo = pastilla.tipo
    diccionario_conteo[tipo] = diccionario_conteo.get(tipo, 0) + 1
    pastilla.id_numero = diccionario_conteo[tipo]
    pastilla.etiqueta = f"{tipo}{pastilla.id_numero}"


img_resultado = img_cinta_limpia.copy()

for pastilla in lista_pastillas:
    cv2.drawContours(img_resultado, [pastilla.contorno], -1, (0, 255, 0), 2)
    cv2.circle(img_resultado, (pastilla.cx, pastilla.cy), 3, (0, 0, 255), -1)
    posicion_texto = (pastilla.cx - 15, pastilla.cy - 15)
    cv2.putText(img_resultado, pastilla.etiqueta, posicion_texto, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title('Clasificación Final de Pastillas con Etiquetas')
plt.axis('off')
plt.tight_layout()
plt.show(block=False)

# Clustering de color
h_values = []
s_values = []

for c in contornos_pastillas:
    mascara = np.zeros(img_hsv_recortada.shape[:2], dtype=np.uint8)
    cv2.drawContours(mascara, [c], -1, 255, -1)

    promedio = cv2.mean(img_hsv_recortada, mask=mascara)

    h_values.append(promedio[0])
    s_values.append(promedio[1])

plt.figure(figsize=(10, 6))

h_grid, s_grid = np.meshgrid(np.arange(180), np.arange(256))
v_grid = np.ones_like(h_grid) * 255
fondo_hsv = np.dstack((h_grid, s_grid, v_grid)).astype(np.uint8)
fondo_rgb = cv2.cvtColor(fondo_hsv, cv2.COLOR_HSV2RGB)

plt.imshow(
    fondo_rgb,
    origin='lower',
    extent=[0, 179, 0, 255],
    aspect='auto',
    alpha=0.5
)

plt.scatter(
    h_values,
    s_values,
    color='black',
    edgecolor='white',
    s=60,
    alpha=0.9,
    zorder=5
)

plt.title('Clustering de Color: Matiz (H) vs Saturación (S)', fontsize=14)
plt.xlabel('Hue / Matiz (0 - 179)')
plt.ylabel('Saturation / Saturación (0 - 255)')
plt.grid(linestyle='--', alpha=0.4, color='black')
plt.axhline(y=75, color='black', linestyle='--', linewidth=2, alpha=0.8)
plt.xlim(0, 179)
plt.ylim(0, 255)
plt.tight_layout()
plt.show(block=False)
plt.show()