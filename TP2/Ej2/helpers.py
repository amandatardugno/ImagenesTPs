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