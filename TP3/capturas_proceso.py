import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers import segmentar_no_verde, detectar_dados, CANTIDAD_DADOS, contar_puntos, detector_puntos, RastreadorDados

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

    # Guardamos el primer frame donde se confirma el reposo 
    if reposo and not reposo_capturado:
        reposo_capturado = True
        
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