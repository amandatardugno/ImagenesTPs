import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helpers import (
    aislar_cinta_transportadora,
    encontrar_pastillas,
    Pastilla
)

img_path = '../data/pills.png'
img = cv2.imread(img_path)

if img is None:
    print(f"ERROR: No se pudo cargar la imagen en '{img_path}'.")
    exit()

# Segmentamos la cinta transportadora (ROI)
coordenadas = aislar_cinta_transportadora(img, padding=10)
if coordenadas is None:
    print("No se encontró la cinta transportadora.")
    exit()
x, y, w, h = coordenadas

img_cinta = img[y:y+h, x:x+w].copy()
img_hsv = cv2.cvtColor(img_cinta, cv2.COLOR_BGR2HSV)

# Detectamos y segmentamos cada pastilla
contornos_pastillas = encontrar_pastillas(img_cinta)

# Clasificamos cada pastilla y armamos el conteo por tipo
lista_pastillas = []
diccionario_conteo = {}
for contorno in contornos_pastillas:
    pastilla = Pastilla(contorno, img_hsv)
    diccionario_conteo[pastilla.tipo] = diccionario_conteo.get(pastilla.tipo, 0) + 1
    pastilla.id_numero = diccionario_conteo[pastilla.tipo]
    pastilla.etiqueta = f"{pastilla.tipo}{pastilla.id_numero}"
    lista_pastillas.append(pastilla)

# Informamos por consola
print(f"Total de pastillas detectadas: {len(lista_pastillas)}")
print("\nConteo por tipo:")
for tipo in sorted(diccionario_conteo):
    print(f"    {tipo}: {diccionario_conteo[tipo]}")

# Generamos la imagen resultante con cada pastilla etiquetada
img_resultado = img_cinta.copy()
for pastilla in lista_pastillas:
    cv2.drawContours(img_resultado, [pastilla.contorno], -1, (0, 255, 0), 2)
    cv2.circle(img_resultado, (pastilla.cx, pastilla.cy), 3, (0, 0, 255), -1)
    posicion_texto = (pastilla.cx - 15, pastilla.cy - 15)
    cv2.putText(img_resultado, pastilla.etiqueta, posicion_texto,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Imagen final con las pastillas detectadas y etiquetadas
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title('Clasificación final de pastillas con etiquetas')
plt.axis('off')
plt.tight_layout()
plt.show(block=False)

# Distribución de pastillas por tipo
tipos = sorted(diccionario_conteo)
cantidades = [diccionario_conteo[t] for t in tipos]

plt.figure(figsize=(9, 6))
barras = plt.bar(tipos, cantidades, color='skyblue', edgecolor='black')
plt.title('Distribución de pastillas por tipo')
plt.xlabel('Tipo de pastilla')
plt.ylabel('Cantidad')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for barra, cantidad in zip(barras, cantidades):
    plt.text(barra.get_x() + barra.get_width() / 2, cantidad,
             str(cantidad), ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.show(block=False)
plt.show()
