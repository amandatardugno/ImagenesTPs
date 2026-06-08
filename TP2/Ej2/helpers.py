import os
import cv2
import numpy as np

MIN_AREA_THRESHOLD = 50
TAMANIO_PATENTE = (45,65)

def normalizar_imagen(img, ancho_estandar=800):
    """
    Redimensiona la imagen a un ancho fijo manteniendo la proporción (aspect ratio).
    Esto garantiza que los kernels morfológicos y los umbrales de área en píxeles 
    funcionen igual para imágenes de cualquier resolución original.
    """
    alto_original, ancho_original = img.shape[:2]
    
    # Calculamos la proporción de la escala
    proporcion = ancho_estandar / float(ancho_original)
    alto_nuevo = int(alto_original * proporcion)
    
    img_redimensionada = cv2.resize(img, (ancho_estandar, alto_nuevo), interpolation=cv2.INTER_AREA)
    
    return img_redimensionada

def aplicar_blackhat(img_gray, kernel_size=(15, 15)):
    """
    Usa black-hat para extraer las letras negras sobre el fondo blanco de la patente.
    img_gray: Imagen original en escala de grises
    kernel_size: Kernel a usar en la transformación
    Devuelve: Imagen transformada
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    return blackhat

def binarizar_adaptativo(img_gray):
    """
    Usa umbral adaptativo para binarizar la imagen en blanco y negro,
    compensando problemas de iluminación.
    img_gray: Imagen original en escala de grises
    Devuelve: Imagen binarizada
    """
    thresh = cv2.adaptiveThreshold(img_gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 51, -5)
    
    return thresh

def filtrar_contornos(thresh):
    """
    Busca componentes conectadas que parezcan letras y sean 7 alineadas (compatible con la patente).
    thresh: Imagen binarizada
    Devuelve: Lista de bounding boxes [x, y, w, h, area] ordenadas de izq a der de los caracteres.
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    candidatos = []

    # stats contiene [x, y, w, h, area]
    # Arrancamos desde 1 para ignorar el fondo (label 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Filtramos por relación de aspecto (entre 65/45 y 65/35, con margen) y por area
        aspect_ratio = h / float(w)
        
        if 1.5 <= aspect_ratio <= 5.0 and MIN_AREA_THRESHOLD <= area <= 0.05 * stats[0][4]:
            candidatos.append(stats[i])
    
    return candidatos

def encontrar_patente(candidatos):

    patente_detectada = []

    for i in range(len(candidatos)):
        x1, y1, w1, h1, a1 = candidatos[i]
        
        # Iniciamos un grupo con el caracter actual
        grupo_actual = [candidatos[i]]

        for j in range(len(candidatos)):
            if i == j: continue

            x2, y2, w2, h2, a2 = candidatos[j]

            max_h = max(h1, h2)
            max_w = max(w1, w2)

            dist_y = abs(y1 - y2)
            diff_h = abs(h1 - h2)
            dist_x = abs(x1 - x2)

            # Condiciones para decir que están en una patente
            # 1. dist_y: Comparten el renglón (tolerancia 40% de la altura).
            # 2. diff_h: Miden casi lo mismo de alto (tolerancia 10%).
            # 3. dist_x: Pertenecen a la misma patente. La distancia máxima entre la 
            #  primera letra y la última es, técnicamente, 7.5 u 8.5 veces el ancho de una letra.
            if dist_y <= max_h * 0.4 and diff_h <= max_h * 0.1 and dist_x <= max_w * 10:
                grupo_actual.append(candidatos[j])

        # Si hay 7 caracteres que cumplieron, es la patente.
        if len(grupo_actual) == 7:
            # Eliminamos posibles duplicados (convirtiendo a tupla el grupo que entra en la lista)
            grupo_unico = list(set(map(tuple, grupo_actual)))
            
            # Ordenamos de izquierda a derecha
            grupo_unico.sort(key=lambda st: st[0])
            
            patente_detectada = grupo_unico
            break

    return patente_detectada