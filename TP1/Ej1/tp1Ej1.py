# Librerías a utilizar:
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Función de ecualizado de histograma local.
def ecualizacionHistogramaLocal(img, ventana, final_blur=True):
    """
    img: Imagen de entrada (grayscale).
    ventana: Tupla con tamaño de ventana (debe ser impar, ej: (15, 15)).
    Devuelve: Imagen con ecualizado de histograma local.
    """

    # Obtiene las dimensiones de la imagen y la ventana
    alto_imagen, ancho_imagen = img.shape
    alto_ventana, ancho_ventana = ventana
    
    if alto_ventana % 2 == 0 or ancho_ventana % 2 == 0:
        raise ValueError(f"Error: La ventana {ventana} tiene dimensiones pares. "
                         "Debe ser impar para garantizar un centro único.")

    # Inicializo la imagen de salida como una copia de la imagen original.
    img_salida = img.copy()
    
    # Calculamos el offset de la ventana para el centrado
    off_vertical = alto_ventana // 2
    off_horizontal = ancho_ventana // 2
    
    # Aplicamos padding para manejar los bordes (reflejando la imagen)
    img_pad = cv2.copyMakeBorder(img, off_vertical, off_vertical, off_horizontal, off_horizontal, cv2.BORDER_REFLECT)
    
    # Recorremos cada píxel de la imagen original
    for i in range(alto_imagen):
        for j in range(ancho_imagen):
            # Extraemos la vecindad (ROI) usando las coordenadas con padding
            # La ventana está centrada en (i + off_vertical, j + off_horizontal) en la imagen con pad
            roi = img_pad[i : i + alto_ventana, j : j + ancho_ventana]
            
            # Ecualizamos la ventana
            roi_equalizada = cv2.equalizeHist(roi)
            
            # El valor del píxel central de la ventana ecualizada (i + off_vertical, j + off_horizontal)
            # es el nuevo valor para nuestro píxel (i, j)
            img_salida[i, j] = roi_equalizada[off_vertical, off_horizontal]

    # Aplica medianBlur para reducir el ruido de la imagen.
    if final_blur:
        img_salida = cv2.medianBlur(img_salida, 3)

    return img_salida

# Carga la imagen con detalles escondidos del ejercicio.
imagen_con_detalles_escondidos = cv2.imread('./Ej1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 8))
plt.subplot(1, 3, 1)
plt.title('Imagen Original')
plt.imshow(imagen_con_detalles_escondidos, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Imagen Ecualizada')
plt.imshow(ecualizacionHistogramaLocal(imagen_con_detalles_escondidos, (21,21), final_blur=False), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Imagen Ecualizada con Blur')
plt.imshow(ecualizacionHistogramaLocal(imagen_con_detalles_escondidos, (21,21), final_blur=True), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()


# Definimos 5 casos de ventana locales
casos_ventana = [(5, 5), (21, 21), (51, 51), (5, 51), (51, 5)]

# Creamos la figura general para 2 filas y 3 columnas
plt.figure(figsize=(15, 10))

# Índice para llevar el control de los subplots
subplot_idx = 1

# Iteramos y graficamos los 5 casos locales
for tam in casos_ventana:
    res = ecualizacionHistogramaLocal(imagen_con_detalles_escondidos, tam)

    plt.subplot(2, 3, subplot_idx)
    plt.imshow(res, cmap='gray')
    plt.title(f"Local {tam[0]}x{tam[1]}")
    plt.axis('off')
    
    subplot_idx += 1

# Agregamos Ecualización Global para comparar
res_global = cv2.equalizeHist(imagen_con_detalles_escondidos)
plt.subplot(2, 3, subplot_idx)
plt.imshow(res_global, cmap='gray')
plt.title("Ecualización Global")
plt.axis('off')
plt.tight_layout()
plt.show()