import cv2
import numpy as np

SHADOW_THRESHOLD = 50
GLOBAL_THRESHOLD = 100
MIN_AREA_THRESHOLD = 10

def aislar_cinta_transportadora(img, umbral=SHADOW_THRESHOLD, padding=0):
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

def encontrar_pastillas(img, umbral=GLOBAL_THRESHOLD):
    """
    Identifica las pastillas en la imagen.
    img: Imagen original en color
    umbral: Valor de corte para separar el fondo
    Devuelve: lista de contornos
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_gray = img_hsv[:, :, 2]

    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contornos_limpios = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area > MIN_AREA_THRESHOLD:
            contornos_limpios.append(c)

    return contornos_limpios