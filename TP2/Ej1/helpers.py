import cv2
import numpy as np

SHADOW_THRESHOLD = 50
GLOBAL_THRESHOLD = 120
MIN_AREA_THRESHOLD = 10

def obtener_bordes(img, umbral=GLOBAL_THRESHOLD):
    """
    Devuelve la imagen identificando los bordes por Canny.
    img: Imagen original en color
    Devuelve: Imagen en escala de grises con bordes
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_v = img_hsv[:, :, 2]
    
    img_blur = cv2.GaussianBlur(img_v, (5, 5), 10)
    
    edges = cv2.Canny(img_blur, SHADOW_THRESHOLD, umbral, apertureSize=3)

    return edges


def aislar_cinta_transportadora_con_sombra(img, umbral=SHADOW_THRESHOLD, padding=0):
    """
    Identifica la cinta transportadora por área y devuelve su Bounding Box.
    img: Imagen original en color.
    umbral: Valor de corte para separar la sombra del fondo de la cinta.
    Devuelve: (x, y, w, h)
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Para reconocer la cinta, usamos la sombra entre la cinta y el borde
    _, thresh = cv2.threshold(img_gray, umbral, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contornos:
        return None

    # La cinta debería ser el objeto más grande
    contorno_cinta = max(contornos, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contorno_cinta)

    return (x+padding, y+padding, w-2*padding, h-2*padding)

def aislar_cinta_transportadora(img, umbral=GLOBAL_THRESHOLD, padding=0):
    """
    Identifica la cinta transportadora usando el canal V de HSV y 
    Hough Probabilístico para encontrar las 2 líneas horizontales más largas.
    img: Imagen original en color.
    Devuelve: (x, y, w, h)
    """
    alto_img, ancho_img = img.shape[:2]

    edges=obtener_bordes(img, umbral)

    # minLineLength la hacemos relativa al ancho de la imagen (15% del ancho)
    min_length = int(ancho_img * 0.15)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=min_length)

    if lines is None:
        return None

    lineas_horizontales = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
            
        angulo = abs(np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi)
        
        # Una línea horizontal tiene un ángulo cercano a 0 o 180 grados
        # Damos una tolerancia de +/- 5 grados
        if angulo < 5 or angulo > 175:
            longitud = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Guardamos la longitud y la coordenada Y promedio de la línea
            y_promedio = (y1 + y2) / 2.0
            lineas_horizontales.append({'longitud': longitud, 'x1': x1, 'x2': x2, 'y': y_promedio})
            
    if len(lineas_horizontales) < 2:
        return None

    mitad_imagen = alto_img // 2
    
    # top_lineas tiene a las lineas horizontales arriba de la mitad de la iamgen
    top_lineas = [linea for linea in lineas_horizontales if linea['y'] < mitad_imagen]

    # idem bottom_lines
    bottom_lines = [linea for linea in lineas_horizontales if linea['y'] >= mitad_imagen]
    
    if not top_lineas or not bottom_lines:
            return None

    y_top=int(max(linea['y'] for linea in top_lineas))
    y_bottom=int(min(linea['y'] for linea in bottom_lines))

    x = padding
    y = y_top + padding
    w = ancho_img - 2 * padding
    h = (y_bottom - y_top) - 2 * padding
    
    # Validamos que el alto no sea negativo por un padding excesivo
    if h <= 0 or w <= 0:
        return None
        
    return (x, y, w, h)

def encontrar_pastillas(img, umbral=GLOBAL_THRESHOLD):
    """
    Identifica las pastillas en la imagen.
    img: Imagen original en color
    umbral: Valor de corte para separar el fondo
    Devuelve: lista de contornos
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_gray = img_hsv[:, :, 2]

    _, thresh = cv2.threshold(img_gray, umbral, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_limpios = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area > MIN_AREA_THRESHOLD:
            contornos_limpios.append(c)

    return contornos_limpios