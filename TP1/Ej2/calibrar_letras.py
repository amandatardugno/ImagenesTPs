import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import encontrarRespuestas, bordesExamen, extraerCaracteristicasLetra

# Carga la imagen del examen 4, que tiene balanceadas la cantidad de respuestas.
img = cv2.imread("./Ej2/examen_4.png", cv2.IMREAD_GRAYSCALE)

respuestas_4 = {
    1: 'B', 2: 'C', 3: 'D', 4: 'B', 5: 'A', 
    6: 'A', 7: 'C', 8: 'D', 9: 'B', 10: 'A'
}

# Lista para guardar las características
caracteristicas = []

bordes = bordesExamen(img)

for e in range(1, 11):
    (l, t), (r, b) = bordes[e]
    ejercicio = img[t:b, l:r]
    respuestas = encontrarRespuestas(ejercicio)
    
    # Solo calibramos si detectó exactamente 1 respuesta
    if len(respuestas) == 1:
        x, y, w, h, area = respuestas[0]
        letra = respuestas_4[e]
        # Recortamos la letra con un margen
        margen = 2
        letra_roi = ejercicio[max(0, y-margen) : y+h+margen, max(0, x-margen) : x+w+margen]
    
        # Sacamos los agujeros y el área del agujero más grande
        agujeros, area_agujero = extraerCaracteristicasLetra(letra_roi)
        caracteristicas.append((letra, w, h, area, agujeros, area_agujero))
        print(e,letra,w,h,area, agujeros, area_agujero)

colores = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'orange'}

plt.figure(figsize=(8, 6))
plt.title('Calibración: Agujeros vs Área del Agujero', fontsize=14)

for letra_buscada in ['A', 'B', 'C', 'D']:
    # Filtramos los datos de la letra actual
    datos_letra = [d for d in caracteristicas if d[0] == letra_buscada]
    
    if not datos_letra:
        continue
        
    # Índice 4 es 'agujeros' y el índice 5 es 'area_agujero'
    agujeros = [d[4] for d in datos_letra]
    area_agujero = [d[5] for d in datos_letra]
    c = colores[letra_buscada]
    
    # Hacemos el scatter plot (s=100 aumenta un poco el tamaño de los puntos)
    plt.scatter(agujeros, area_agujero, color=c, alpha=0.7, label=f'Letra {letra_buscada}', s=100)

plt.xlabel('Cantidad de Agujeros')
plt.ylabel('Área del Agujero (px²)')

# Forzamos que el eje X solo muestre números enteros
plt.xticks([0, 1, 2]) 

plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()