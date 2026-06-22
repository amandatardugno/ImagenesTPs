import cv2
import numpy as np

# Rango HSV del fondo verde
VERDE_MIN = np.array([35, 40, 30])
VERDE_MAX = np.array([95, 255, 255])

# Área que puede tener el contorno de un dado
AREA_MIN = 1500
AREA_MAX = 10000

# Factor de forma que puede tener un dado
FACTOR_FORMA_MIN = 0.043
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

    # Clausura para suavizar el borde rugoso
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

def configurar_detector_blobs():
    params = cv2.SimpleBlobDetector_Params()
    
    # Buscamos blobs blancos (255)
    params.filterByColor = True
    params.blobColor = 255
    
    # Evitamos detectar ruido o mucha área
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 500
    
    # Buscamos círculos bien definidos
    params.filterByCircularity = True
    params.minCircularity = 0.7 
    
    # Evitamos óvalos o manchas por el movimiento
    params.filterByInertia = True
    params.minInertiaRatio = 0.6
    
    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    return cv2.SimpleBlobDetector_create(params)

# Lo instanciamos una sola vez globalmente para que no lenteje el video
detector_puntos = configurar_detector_blobs()

def contar_puntos(roi_bgr):
    """
    Toma la ROI (recorte) del dado y devuelve: 
    (cantidad_puntos, keypoints_detectados, mascara_blanca)
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    # Ajuste de color para los puntos blancos: muy baja saturación y alto brillo
    saturacion_max = 80
    brillo_min = 150
    
    # Máscara: Hue[0-180], Sat[0-80], Val[150-255]
    lower_white = np.array([0, 0, brillo_min])
    upper_white = np.array([180, saturacion_max, 255])
    
    mask_blanco = cv2.inRange(hsv, lower_white, upper_white)
    
    # Pasamos la máscara al detector. Al ser imagen binaria, los puntos son 255 (blanco)
    keypoints = detector_puntos.detect(mask_blanco)
    
    return len(keypoints), keypoints, mask_blanco

import math

class Dado:
    def __init__(self, id_dado, bbox):
        self.id = id_dado
        self.bbox = bbox
        self.puntos = 0
        self.centro = self._calcular_centro(bbox)
        
        self.frames_perdido = 0 

    def _calcular_centro(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def actualizar_posicion(self, bbox):
        self.bbox = bbox
        self.centro = self._calcular_centro(bbox)
        self.frames_perdido = 0 

class RastreadorDados:
    def __init__(self, max_frames_perdido=15): # Tolera hasta 15 frames sin verlo
        self.dados_activos = []
        self.proximo_id = 1
        self.max_frames_perdido = max_frames_perdido

    def actualizar(self, bboxes):
        dados_actualizados = []
        dados_matcheados = set()
        
        # Intentamos emparejar las detecciones nuevas con los dados existentes
        for bbox in bboxes:
            x, y, w, h = bbox
            centro_nuevo = (x + w // 2, y + h // 2)
            
            mejor_dado = None
            min_dist = float('inf')
            
            max_distancia = 3 * w 
            
            for dado in self.dados_activos:
                if dado in dados_matcheados:
                    continue # Ya le asignamos una caja a este dado
                
                dist = math.hypot(centro_nuevo[0] - dado.centro[0], centro_nuevo[1] - dado.centro[1])
                if dist < min_dist:
                    min_dist = dist
                    mejor_dado = dado
                    
            if mejor_dado is not None and min_dist < max_distancia:
                mejor_dado.actualizar_posicion(bbox)
                dados_actualizados.append(mejor_dado)
                dados_matcheados.add(mejor_dado)
            else:
                # Si no hay ninguno cerca, es un dado nuevo
                nuevo_dado = Dado(self.proximo_id, bbox)
                dados_actualizados.append(nuevo_dado)
                self.proximo_id += 1
                
        # Dados que no se detectan nuevamente
        for dado in self.dados_activos:
            if dado not in dados_matcheados:
                dado.frames_perdido += 1
                if dado.frames_perdido <= self.max_frames_perdido:
                    dados_actualizados.append(dado)
                    
        self.dados_activos = dados_actualizados
        return self.dados_activos