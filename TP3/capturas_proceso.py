import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers import (segmentar_no_verde, detectar_dados, contar_puntos, factor_de_forma,
                     RastreadorDados, CANTIDAD_DADOS, VERDE_MIN, VERDE_MAX)

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if new_fig:
        plt.show(block=blocking)


VIDEO_PATH = "data/tirada_3.mp4"
FRAMES_REPOSO = 4      # frames seguidos con 5 dados y lectura de puntos constante
ALTO_DEBUG = 600       # alto fijo de cada panel de debug (px)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print(f"ERROR: no se pudo leer el video '{VIDEO_PATH}'.")
    cap.release()
    exit()

numero_frame = 0
frames_estables = 0
reposo_capturado = False
frame_reposo_idx = None
historial = []   # (frame, n_dados, frames_estables) para la curva de estabilidad

frame_reposo_crudo = None
frame_reposo_marcado = None
valores_finales = []
puntos_anteriores = []

cv2.namedWindow("Debug deteccion de reposo", cv2.WINDOW_NORMAL)
rastreador = RastreadorDados()
puntos_anteriores = {}

while ret:
    numero_frame += 1

    mascara = segmentar_no_verde(frame)
    bboxes_crudos = detectar_dados(mascara)
    
    dados_trackeados = rastreador.actualizar(bboxes_crudos)
    
    n_dados = len(dados_trackeados)
    dibujo = frame.copy()
    
    puntos_actuales = {}

    # Ahora iteramos sobre los objetos de tu clase
    for dado in dados_trackeados:
        x, y, w, h = dado.bbox
        
        if dado.frames_perdido > 0:
            # Si el dado está oculto, usamos su memoria. No evaluamos puntos nuevos.
            color_box = (0, 255, 255) # Amarillo para "En memoria"
            cant_puntos = dado.puntos # Usamos los puntos del frame anterior
            puntos_actuales[dado.id] = cant_puntos
            
            cv2.rectangle(dibujo, (x, y), (x + w, y + h), color_box, 2, cv2.LINE_4)
            cv2.putText(dibujo, f"D{dado.id}-{cant_puntos}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
            continue # Pasamos al siguiente dado sin procesar la imagen
        
        roi_dado = frame[y:y+h, x:x+w]
        cant_puntos, keypoints, _ = contar_puntos(roi_dado)
        
        # Guardamos los puntos en el objeto y en el diccionario actual
        dado.puntos = cant_puntos
        puntos_actuales[dado.id] = cant_puntos

        if cant_puntos >= 1:
            color_box = (0, 255, 0) # Verde
            for kp in keypoints:
                centro = (int(kp.pt[0]) + x, int(kp.pt[1]) + y)
                radio = int(kp.size / 2)
                cv2.circle(dibujo, centro, radio, (255, 0, 0), 2)
        else:
            color_box = (255, 0, 255) # Magenta

        cv2.rectangle(dibujo, (x, y), (x + w, y + h), color_box, 3)
        cv2.putText(dibujo, f"D{dado.id}-{cant_puntos}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_box, 2)

    # Hay reposo si:
    # 1. Hay 5 dados. 
    # 2. Todos tienen 1 o más puntos.
    # 3. Los IDs detectados y sus puntos son iguales a los del frame anterior
    son_validos = (n_dados == CANTIDAD_DADOS) and all(p >= 1 for p in puntos_actuales.values())
    
    if son_validos:
        if puntos_actuales == puntos_anteriores:
            frames_estables += 1
        else:
            frames_estables = 1 
    else:
        frames_estables = 0

    puntos_anteriores = puntos_actuales.copy()
    reposo = frames_estables >= FRAMES_REPOSO
    historial.append((numero_frame, n_dados, frames_estables))

    # Guardamos el primer frame donde se confirma el reposo 
    if reposo and not reposo_capturado:
        reposo_capturado = True
        frame_reposo_idx = numero_frame

        frame_reposo_crudo = frame.copy()
        frame_reposo_marcado = dibujo.copy() 
        valores_finales = puntos_actuales
        
        print(f"\n" + "="*50)
        print(f">> REPOSO CONFIRMADO EN EL FRAME {numero_frame}")
        print(f">> Valores informados:")
        for i, val in enumerate(valores_finales, start=1):
            print(f"   - D{i}: {val}")
        print("="*50 + "\n")
        
        cv2.imwrite("frame_dados_detenidos.jpg", frame_reposo_marcado)

    # Panel de debug
    estado = "REPOSO" if reposo else "ANALIZANDO"
    color = (0, 255, 0) if reposo else (0, 255, 255)
    cv2.putText(dibujo, f"{estado} | dados={n_dados} | estables={frames_estables}/{FRAMES_REPOSO}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    mascara_color = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    escala = ALTO_DEBUG / frame.shape[0]
    tam = (int(frame.shape[1] * escala), ALTO_DEBUG)
    panel = cv2.hconcat([cv2.resize(dibujo, tam), cv2.resize(mascara_color, tam)])
    cv2.imshow("Debug deteccion de reposo", panel)

    tecla = cv2.waitKey(30) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord(" "):     
        cv2.waitKey(0)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

if frame_reposo_crudo is not None:
    frame_rgb = cv2.cvtColor(frame_reposo_crudo, cv2.COLOR_BGR2RGB)
    mascara_final = segmentar_no_verde(frame_reposo_crudo)
    marcado_rgb = cv2.cvtColor(frame_reposo_marcado, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    ax = plt.subplot(131); imshow(frame_rgb, title="Frame de reposo (Original)", color_img=True, new_fig=False)
    plt.subplot(132, sharex=ax, sharey=ax); imshow(mascara_final, title="Máscara no-verde", new_fig=False)
    plt.subplot(133, sharex=ax, sharey=ax); imshow(marcado_rgb, title=f"Resultado: {valores_finales}", color_img=True, new_fig=False)
    plt.tight_layout()
    plt.show(block=True)
else:
    print("No se detectó un frame de reposo en este video")

def mascara_blanco_y_puntos(roi_bgr):
    """ROI de un dado -> (máscara de blanco, ROI con los puntos marcados, cantidad)."""
    cant, keypoints, mask_blanco = contar_puntos(roi_bgr)
    roi_marcado = roi_bgr.copy()
    for kp in keypoints:
        cv2.circle(roi_marcado, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (255, 0, 0), 2)
    return mask_blanco, cv2.cvtColor(roi_marcado, cv2.COLOR_BGR2RGB), cant


if frame_reposo_crudo is not None:
    bboxes_reposo = detectar_dados(segmentar_no_verde(frame_reposo_crudo))

    # Caso elegido para el zoom
    bbox_dado = max(bboxes_reposo,
                    key=lambda b: contar_puntos(frame_reposo_crudo[b[1]:b[1]+b[3], b[0]:b[0]+b[2]])[0])
    x, y, w, h = bbox_dado
    roi_reposo = frame_reposo_crudo[y:y+h, x:x+w]

    # Captura zoom a un dado
    mask_r, roi_r_marcado, cant_r = mascara_blanco_y_puntos(roi_reposo)
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(131); imshow(cv2.cvtColor(roi_reposo, cv2.COLOR_BGR2RGB), title="ROI del dado", color_img=True, new_fig=False)
    plt.subplot(132); imshow(mask_r, title="Máscara de blanco (S<=80, V>=150)", new_fig=False)
    plt.subplot(133); imshow(roi_r_marcado, title=f"Puntos detectados: {cant_r}", color_img=True, new_fig=False)
    plt.suptitle("Conteo de puntos en un dado")
    plt.tight_layout()
    plt.show(block=True)

    # Captura movimiento vs reposo en los puntos 
    idx_mov = max(1, (frame_reposo_idx or 16) - 15)
    cap_mov = cv2.VideoCapture(VIDEO_PATH)
    cap_mov.set(cv2.CAP_PROP_POS_FRAMES, idx_mov - 1)
    ret_mov, frame_mov = cap_mov.read()
    cap_mov.release()

    roi_mov = None
    if ret_mov:
        contornos, _ = cv2.findContours(segmentar_no_verde(frame_mov), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []   
        for c in contornos:
            area = cv2.contourArea(c)
            if 1500 <= area <= 12000 and 0.02 <= factor_de_forma(c) <= 0.07:
                bx, by, bw, bh = cv2.boundingRect(c)
                cp, _, _ = contar_puntos(frame_mov[by:by+bh, bx:bx+bw])
                candidatos.append((cp, (bx, by, bw, bh)))
        if candidatos:
            _, (bx, by, bw, bh) = min(candidatos, key=lambda t: t[0])
            roi_mov = frame_mov[by:by+bh, bx:bx+bw]

    if roi_mov is not None:
        mask_m, roi_m_marcado, cant_m = mascara_blanco_y_puntos(roi_mov)
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(231); imshow(cv2.cvtColor(roi_mov, cv2.COLOR_BGR2RGB), title="Dado en MOVIMIENTO", color_img=True, new_fig=False)
        plt.subplot(232); imshow(mask_m, title="Máscara de blanco", new_fig=False)
        plt.subplot(233); imshow(roi_m_marcado, title=f"Círculos nítidos: {cant_m}", color_img=True, new_fig=False)
        plt.subplot(234); imshow(cv2.cvtColor(roi_reposo, cv2.COLOR_BGR2RGB), title="Dado en REPOSO", color_img=True, new_fig=False)
        plt.subplot(235); imshow(mask_r, title="Máscara de blanco", new_fig=False)
        plt.subplot(236); imshow(roi_r_marcado, title=f"Círculos nítidos: {cant_r}", color_img=True, new_fig=False)
        plt.suptitle("Por qué usamos los puntos: en movimiento se desdibujan y no se detectan")
        plt.tight_layout()
        plt.show(block=True)

    # Captura efecto de la clausura sobre el borde del dado y el factor de forma
    hsv_reposo = cv2.cvtColor(frame_reposo_crudo, cv2.COLOR_BGR2HSV)
    mask_sin = cv2.bitwise_not(cv2.inRange(hsv_reposo, VERDE_MIN, VERDE_MAX))   # sin clausura
    mask_con = segmentar_no_verde(frame_reposo_crudo)                            # con clausura

    def ff_recorte(mascara, bbox, pad=12):
        bx, by, bw, bh = bbox
        sub = mascara[max(0, by-pad):by+bh+pad, max(0, bx-pad):bx+bw+pad]
        contornos, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return sub, None
        return sub, factor_de_forma(max(contornos, key=cv2.contourArea))

    sub_sin, ff_sin = ff_recorte(mask_sin, bbox_dado)
    sub_con, ff_con = ff_recorte(mask_con, bbox_dado)
    plt.figure(figsize=(9, 5))
    ax = plt.subplot(121); imshow(sub_sin, title=f"Sin clausura (factor de forma {ff_sin:.4f})", new_fig=False)
    plt.subplot(122); imshow(sub_con, title=f"Con clausura (factor de forma {ff_con:.4f})", new_fig=False)
    plt.suptitle("La clausura suaviza el borde y sube el factor de forma")
    plt.tight_layout()
    plt.show(block=True)

# Captura curva de estabilidad
if historial:
    frames_x = [h[0] for h in historial]
    n_dados_y = [h[1] for h in historial]
    estables_y = [h[2] for h in historial]

    plt.figure(figsize=(11, 4))
    plt.plot(frames_x, n_dados_y, label="dados detectados")
    plt.plot(frames_x, estables_y, label="frames estables")
    plt.axhline(CANTIDAD_DADOS, color="gray", linestyle="--", linewidth=1, label=f"{CANTIDAD_DADOS} dados")
    if frame_reposo_idx is not None:
        plt.axvline(frame_reposo_idx, color="green", linestyle="--", label=f"reposo (frame {frame_reposo_idx})")
    plt.xlabel("frame"); plt.ylabel("cantidad")
    plt.title("Detección de dados y estabilidad a lo largo del video")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)