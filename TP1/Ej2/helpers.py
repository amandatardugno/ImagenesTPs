import cv2
import numpy as np

GLOBAL_THRESHOLD = 150

def bordesExamen(img):
    th=GLOBAL_THRESHOLD

    img_th=img<th

    img_cols = np.sum(img_th,0)
    img_rows = np.sum(img_th,1)

    alto_imagen, ancho_imagen = img.shape

    th_col = int(0.75*alto_imagen)
    th_row = int(0.75*ancho_imagen)

    img_cols_th = img_cols>th_col
    img_rows_th = img_rows>th_row

    left = []
    top = []
    right = []
    bottom = []

    for i in range(0,len(img_cols_th)-1):
        left_col = img_cols_th[i]
        right_col = img_cols_th[i+1]
        if left_col != right_col:
            if left_col and not right_col:
                left.append(i+1)
            else:
                right.append(i)


    for i in range(0,len(img_rows_th)-1):
        top_row = img_rows_th[i]
        bottom_row = img_rows_th[i+1]
        if top_row != bottom_row:
            if top_row and not bottom_row:
                top.append(i+1)
            else:
                bottom.append(i)

    # Bordes de cada recuadro, empezando por el encabezado
    bordes = {
        0:((0,0),(ancho_imagen,bottom[0]))
        }
    
    # Eliminamos las coordenadas fuera de los ejercicios
    del right[0]
    del bottom[0]
    del left[1]
    del right[1]
    del top[-1]
    del left[-1]

    # Agregamos un padding por si el borde no esta tan bien cortado
    p=2

    for e in range(0,10):
        i = e%5
        j = e//5
        bordes[e+1]=((left[j]+p,top[i]+p),(right[j]-p,bottom[i]-p))

    return bordes

def encontrarRespuestas(img):
    # Umbralizamos e invertimos
    # Lo negro pasa a ser blanco (255) y el fondo blanco pasa a negro (0)
    # Esto lo hacemos para que funcione buscar las componentes conectadas
    _, img_th = cv2.threshold(img, GLOBAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Etiquetamos las componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_th, 8, cv2.CV_32S)

    # Buscamos la línea donde se encuentra la respuesta
    # Ignoramos el índice 0 porque es el fondo (background)
    linea=[]
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Si el ancho es más del triple que el alto, debe ser una línea y terminamos este bucle
        if w > (h * 3):
            linea=[x,y,w,h]
            break

    if not linea:
        return []

    respuestas = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        xl, yl, wl, hl = linea

        # Eliminamos ruido (píxeles sueltos o puntos)
        if area < 5:
            continue

        # La respuesta debería estar sobre la linea de respuesta
        # Asumimos que entonces una letra para abajo de la respuesta,
        # debería intersecar a la línea
        if x > xl and x < xl + wl and yl < y + 2*h and yl > y+h:
            respuestas.append((x, y, w, h, area))

    return respuestas

def extraerCaracteristicasLetra(roi_letra):
    """
    Recibe la imagen binarizada de una sola letra recortada.
    Devuelve (cantidad_agujeros, area_agujero_max, area_externa)
    """
    _, roi_letra = cv2.threshold(roi_letra, GLOBAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Buscamos los contornos 
    contours, hierarchy = cv2.findContours(roi_letra, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Por si no llegara a detectar contornos, sale
    if hierarchy is None:
        return 0, 0

    jerarquia = hierarchy[0] # hierarchy devuelve un array de shape (1, N, 4)
    
    cantidad_agujeros = 0
    area_agujero_max = 0
    area_externa = 0

    # jerarquia[i] = [Next, Previous, First_Child, Parent]
    for i, contorno in enumerate(contours):
        padre = jerarquia[i, 3] # El índice 3 es el 'Parent'
        
        if padre == -1:
            area_externa = cv2.contourArea(contorno)
        else:
            # Si tiene un padre (Parent != -1), significa que es un contorno interno (un agujero)
            area_hijo = cv2.contourArea(contorno)
            # Filtramos por si detectó un píxel suelto adentro como agujero
            if area_hijo > 3: 
                cantidad_agujeros += 1
            if area_hijo>area_agujero_max:
                area_agujero_max=area_hijo
                
    return cantidad_agujeros, area_agujero_max, area_externa

def contarPalabras(chars):
    """
    Recibe una lista de stats de caracteres (x, y, w, h) ordenados por x.
    Cuenta cuántas palabras forman separando por gaps horizontales grandes.
    """
    if not chars:
        return 0

    # Usamos el alto promedio como referencia (más uniforme que el ancho).
    # Un espacio entre palabras suele ser ~0.5 del alto del caracter.
    alto_prom = sum(c[3] for c in chars) / len(chars)
    th_gap = alto_prom * 0.45

    palabras = 1
    for i in range(1, len(chars)):
        x_prev, _, w_prev, _ = chars[i-1]
        x_curr = chars[i][0]
        gap = x_curr - (x_prev + w_prev)
        if gap > th_gap:
            palabras += 1
    return palabras


def detectarCamposEncabezado(header_img):
    """
    Recibe la imagen del encabezado del examen.

    Detecta las 3 líneas de los campos (Name, Date, Class) y los caracteres
    que están justo arriba de cada una.

    Devuelve una tupla (lineas, campos):
    - lineas: lista de 3 tuplas (x, y, w, h), ordenadas de izquierda a derecha.
    - campos: lista de 3 listas de chars (x, y, w, h) ordenados por x,
              correspondientes a [Name, Date, Class].

    Si no se detectan al menos 3 líneas, devuelve (None, None).
    """
    _, img_th = cv2.threshold(header_img, GLOBAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img_th, 8, cv2.CV_32S)

    # Separamos las componentes en líneas (mucho más anchas que altas) y caracteres
    lineas = []
    componentes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 5:
            continue
        if w > 3*h and w > 30:
            lineas.append((x, y, w, h))
        else:
            componentes.append((x, y, w, h))

    if len(lineas) < 3:
        return None, None

    # Las 3 líneas más anchas son las de los campos. Las ordenamos de izquierda a derecha.
    lineas.sort(key=lambda l: l[2], reverse=True)
    lineas = lineas[:3]
    lineas.sort(key=lambda l: l[0])

    # Para cada línea, juntamos los caracteres que están justo arriba de la misma
    campos = []
    for xl, yl, wl, hl in lineas:
        chars = []
        for x, y, w, h in componentes:
            cx = x + w/2
            # El centro horizontal del char debe caer sobre la línea
            if cx < xl or cx > xl + wl:
                continue
            # Debe estar arriba de la línea
            if y + h > yl:
                continue
            # Pero no muy lejos
            if yl - (y + h) > 25:
                continue
            chars.append((x, y, w, h))
        chars.sort(key=lambda c: c[0])
        campos.append(chars)

    return lineas, campos


def validarEncabezado(header_img):
    """
    Recibe la imagen del encabezado del examen.

    Valida los campos Name, Date y Class según las restricciones del enunciado.

    Devuelve un dict con el estado (OK/MAL) de cada campo.
    """
    _, campos = detectarCamposEncabezado(header_img)
    if campos is None:
        return {"Name": "MAL", "Date": "MAL", "Class": "MAL"}

    name_chars, date_chars, class_chars = campos

    # Name: al menos 2 palabras y no más de 25 caracteres
    name_ok = 0 < len(name_chars) <= 25 and contarPalabras(name_chars) >= 2
    # Date: 8 caracteres formando una sola palabra
    date_ok = len(date_chars) == 8 and contarPalabras(date_chars) == 1
    # Class: un único caracter
    class_ok = len(class_chars) == 1

    return {
        "Name": "OK" if name_ok else "MAL",
        "Date": "OK" if date_ok else "MAL",
        "Class": "OK" if class_ok else "MAL",
    }


def obtenerName(header_img):
    """
    Recibe la imagen del encabezado del examen.

    Detecta la línea del campo Name y los caracteres sobre ella, y devuelve
    una imagen recortada con ese campo.
    """
    lineas, campos = detectarCamposEncabezado(header_img)
    if lineas is None:
        return None

    xl, yl, wl, hl = lineas[0]
    name_chars = campos[0]

    # Calculamos el bounding box de los caracteres del Name
    alto_header, ancho_header = header_img.shape
    top_line = alto_header
    left_limit = ancho_header
    bottom_line = 0
    right_limit = 0
    for x, y, w, h in name_chars:
        top_line = min(y, top_line)
        left_limit = min(x, left_limit)
        bottom_line = max(y+h, bottom_line)
        right_limit = max(x+w, right_limit)

    # Agregamos un margen proporcional al espacio entre los chars y la línea
    gap = yl - bottom_line
    top_line = max(top_line - gap, 0)
    left_limit -= gap
    right_limit += gap

    return header_img[top_line:yl, left_limit:right_limit]

def identificarRespuestas(img):
    """
    Recibe la imagen de un ejercicio.
    
    Encuentra la respuesta y la identifica en base
    a la cantidad y tamaño de los agujeros.
    
    Devuelve la letra o None si no es A, B, C o D
    """
    respuestas=encontrarRespuestas(img)
    
    if len(respuestas)!=1:
        return None
    
    x, y, w, h, area = respuestas[0]
    
    # Recortamos la letra con un margen
    margen = 2
    letra_roi = img[max(0, y-margen) : y+h+margen, max(0, x-margen) : x+w+margen]

    # Sacamos los agujeros y el área del agujero más grande
    agujeros, area_agujero, area_externa = extraerCaracteristicasLetra(letra_roi)

    if agujeros == 0:
        return 'C'
    elif agujeros == 2:
        return 'B'
    elif agujeros == 1:
        if area_agujero > 0.75*area_externa:
            return 'D'
        else:
            return 'A'
    else:
        return None