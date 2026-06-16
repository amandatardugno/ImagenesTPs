import cv2
import numpy as np


def dados_estan_quietos(
    frame_anterior,
    frame_actual,
    umbral_mascara=0.03,
    umbral_visual=8.0
):
    """
    Determina si los dados están quietos comparando dos frames consecutivos.

    La comparación se realiza únicamente sobre las regiones rojas,
    para evitar que movimientos o sombras del fondo afecten el resultado.

    Devuelve:
        quietos: True si los dados parecen estar detenidos.
        cambio_mascara: proporción de píxeles rojos que cambiaron.
        cambio_visual: diferencia media de intensidad dentro de los dados.
    """

    hsv_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2HSV)
    hsv_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2HSV)

    # El rojo ocupa los dos extremos del canal H.
    rojo_bajo_1 = np.array([0, 80, 50])
    rojo_alto_1 = np.array([10, 255, 255])

    rojo_bajo_2 = np.array([170, 80, 50])
    rojo_alto_2 = np.array([179, 255, 255])

    mascara_anterior = (
        cv2.inRange(hsv_anterior, rojo_bajo_1, rojo_alto_1)
        | cv2.inRange(hsv_anterior, rojo_bajo_2, rojo_alto_2)
    )

    mascara_actual = (
        cv2.inRange(hsv_actual, rojo_bajo_1, rojo_alto_1)
        | cv2.inRange(hsv_actual, rojo_bajo_2, rojo_alto_2)
    )

    # Limpiamos puntos aislados
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mascara_anterior = cv2.morphologyEx(
        mascara_anterior,
        cv2.MORPH_OPEN,
        kernel
    )

    mascara_actual = cv2.morphologyEx(
        mascara_actual,
        cv2.MORPH_OPEN,
        kernel
    )

    # Región ocupada por los dados en cualquiera de los dos frames
    mascara_union = cv2.bitwise_or(
        mascara_anterior,
        mascara_actual
    )

    area_union = cv2.countNonZero(mascara_union)

    if area_union == 0:
        return False, 1.0, float("inf")

    # Cuánto cambió la ubicación o forma de las regiones rojas
    diferencia_mascaras = cv2.bitwise_xor(
        mascara_anterior,
        mascara_actual
    )

    cambio_mascara = (
        cv2.countNonZero(diferencia_mascaras)
        / area_union
    )

    # Cuánto cambia el interior de los dados
    gris_anterior = cv2.cvtColor(
        frame_anterior,
        cv2.COLOR_BGR2GRAY
    )

    gris_actual = cv2.cvtColor(
        frame_actual,
        cv2.COLOR_BGR2GRAY
    )

    diferencia_visual = cv2.absdiff(
        gris_anterior,
        gris_actual
    )

    cambio_visual = cv2.mean(
        diferencia_visual,
        mask=mascara_union
    )[0]

    quietos = (
        cambio_mascara <= umbral_mascara
        and cambio_visual <= umbral_visual
    )

    return quietos, cambio_mascara, cambio_visual