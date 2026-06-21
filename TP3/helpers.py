import cv2
import numpy as np

# Rango HSV del fondo verde
VERDE_MIN = np.array([35, 40, 30])
VERDE_MAX = np.array([95, 255, 255])

# Área que puede tener el contorno de un dado
AREA_MIN = 1500
AREA_MAX = 20000

# Factor de forma que puede tener un dado
FACTOR_FORMA_MIN = 0.05
FACTOR_FORMA_MAX = 0.062

# Tamaño del kernel de la clausura que suaviza el borde rugoso de la máscara
KERNEL_CLAUSURA = 5

# Cantidad de dados esperada por tirada
CANTIDAD_DADOS = 5


def segmentar_no_verde(frame):
    """Máscara binaria: el fondo verde queda en 0 y lo demás (dados, mano) en 255."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, VERDE_MIN, VERDE_MAX)
    mascara = cv2.bitwise_not(verde)

    # Clausura para suavizar el borde rugoso (baja el perímetro y "redondea" el
    # factor de forma de los dados hacia el del cuadrado).
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_CLAUSURA, KERNEL_CLAUSURA))
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, se)


def factor_de_forma(contorno):
    """Compacidad del contorno: A / p² (cuadrado teórico = 1/16)."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    if perimetro == 0:
        return 0.0
    return area / (perimetro ** 2)


def detectar_dados(mascara):
    """
    Devuelve los dados de la máscara como lista de bounding boxes (x, y, w, h).

    Filtra los contornos por área y por factor de forma: así descarta el ruido,
    la madera del fondo (área muy grande) y la mano (no es cuadrada).
    """
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dados = []
    for c in contornos:
        area = cv2.contourArea(c)
        if not (AREA_MIN <= area <= AREA_MAX):
            continue
        if not (FACTOR_FORMA_MIN <= factor_de_forma(c) <= FACTOR_FORMA_MAX):
            continue
        dados.append(cv2.boundingRect(c))

    return dados
