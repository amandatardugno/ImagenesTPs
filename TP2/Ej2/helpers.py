import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        
        if 1.5 <= aspect_ratio <= 5.0 and MIN_AREA_THRESHOLD <= area:
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
            if dist_y <= max_h * 0.4 and diff_h <= max_h * 0.1 and dist_x <= max_w * 9:
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

def cargar_templates(carpeta="templates"):
    """
    Lee todas las imágenes de la carpeta y las guarda en un diccionario.
    Clave: 'A', Valor: matriz de la imagen.
    """
    diccionario_templates = {}
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".png"):
            # El nombre del archivo es el caracter (ej: A.png)
            caracter = archivo.split('.')[0]
            ruta = os.path.join(carpeta, archivo)
            
            # Cargamos en escala de grises
            img_template = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            diccionario_templates[caracter] = img_template
            
    return diccionario_templates

def leer_caracteres_patente(thresh, letras_detectadas, templates):
    """
    Compara cada letra segmentada con los templates para adivinar el texto.
    """
    texto_patente = ""
    
    # Usamos enumerate para tener el índice 'i' (0 a 6)
    for i, (x, y, w, h, area) in enumerate(letras_detectadas):
        # Recortamos el caracter de la imagen binarizada
        recorte = thresh[y:y+h, x:x+w]
        
        # CLAUSURA LOCAL: Rellenamos huecos/imperfecciones del caracter 
        # antes de compararlo con el template perfecto.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        recorte_relleno = cv2.morphologyEx(recorte, cv2.MORPH_CLOSE, kernel)
        
        recorte_redimensionado = cv2.resize(recorte_relleno, TAMANIO_PATENTE)
        
        mejor_coincidencia = -1
        mejor_letra = "?"

        # Filtramos el diccionario de templates según la posición del caracter
        if i in [0, 1, 5, 6]:
            # Si estamos en la posición 0, 1, 5 o 6, SOLO buscamos letras
            templates_a_comparar = {k: v for k, v in templates.items() if k.isalpha()}
        elif i in [2, 3, 4]:
            # Si estamos en la posición 2, 3 o 4, SOLO buscamos números
            templates_a_comparar = {k: v for k, v in templates.items() if k.isdigit()}
        else:
            # Fallback por si acaso
            templates_a_comparar = templates

        for letra, img_template in templates_a_comparar.items():
            # CCOEFF_NORMED devuelve un valor entre -1 y 1. 1 es idéntico.
            resultado = cv2.matchTemplate(recorte_redimensionado, img_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(resultado)
            
            if max_val > mejor_coincidencia:
                mejor_coincidencia = max_val
                mejor_letra = letra
                
        texto_patente += mejor_letra

    return texto_patente

def refinar_patente(img_gray, letras_detectadas):
    """
    Toma la detección "gruesa" de los 7 caracteres, recorta esa zona original con padding,
    la estandariza a un tamaño fijo para aplicar morfología sin miedo, re-detecta
    y devuelve las coordenadas mapeadas a la imagen original.
    """
    if len(letras_detectadas) != 7:
        return letras_detectadas, None # Si no hay 7, no hacemos el refinamiento

    alto_img, ancho_img = img_gray.shape

    # Sacamos el Bounding Box general que envuelve a las 7 letras
    min_x = min([st[0] for st in letras_detectadas])
    min_y = min([st[1] for st in letras_detectadas])
    max_x = max([st[0] + st[2] for st in letras_detectadas])
    max_y = max([st[1] + st[3] for st in letras_detectadas])
    
    w_grupo = max_x - min_x
    h_grupo = max_y - min_y

    pad_x = int(w_grupo / 14.0)
    pad_y = int(h_grupo * 0.1)

    roi_x = max(0, min_x - pad_x)
    roi_y = max(0, min_y - pad_y)
    roi_w = min(ancho_img - roi_x, w_grupo + 2 * pad_x)
    roi_h = min(alto_img - roi_y, h_grupo + 2 * pad_y)

    recorte_original = img_gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Redimensionamos la zona de la patente a un tamaño fijo
    # 45*7 (letras) + 20*6 (espacios) = 435 aprox.
    # Usamos una altura fija aproximada
    W_FIJO = 450
    H_FIJO = 150
    recorte_norm = cv2.resize(recorte_original, (W_FIJO, H_FIJO), interpolation=cv2.INTER_CUBIC)
    
    img_bh = aplicar_blackhat(recorte_norm, kernel_size=(11, 11))
    thresh = binarizar_adaptativo(img_bh)

    # Clausura segura para unir bordes rotos
    kernel_clausura = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh_limpio = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clausura)
    
    # Apertura pequeña para ruido blanco fino
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_limpio = cv2.morphologyEx(thresh_limpio, cv2.MORPH_OPEN, kernel_apertura)

    letras_refinadas_norm = filtrar_contornos(thresh_limpio)
    letras_refinadas_norm = encontrar_patente(letras_refinadas_norm)
    

    # Coordenadas Refinadas -> Coordenadas Originales
    # Regla de 3 simple para deshacer el resize y sumarle el offset del ROI
    escala_x = roi_w / float(W_FIJO)
    escala_y = roi_h / float(H_FIJO)

    letras_finales = []
    for (x_n, y_n, w_n, h_n, a_n) in letras_refinadas_norm:
        # Deshacemos el resize
        x_local = int(x_n * escala_x)
        y_local = int(y_n * escala_y)
        w_local = int(w_n * escala_x)
        h_local = int(h_n * escala_y)
        
        # Le sumamos la posición del recorte en la imagen original
        x_global = x_local + roi_x
        y_global = y_local + roi_y
        
        # Volvemos a armar la tupla
        letras_finales.append((x_global, y_global, w_local, h_local, a_n))
        
    # Devolvemos las mejores letras encontradas (si falló el fino por algo, devolvemos las originales)
    if len(letras_finales) == 7:
        return letras_finales, thresh_limpio
    else:
        return letras_detectadas, None