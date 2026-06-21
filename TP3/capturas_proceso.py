import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers import segmentar_no_verde, detectar_dados, CANTIDAD_DADOS

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


VIDEO_PATH = "data/tirada_1.mp4"
FRAMES_REPOSO = 10    # frames seguidos con 5 dados cuadrados para confirmar reposo
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
frame_reposo = None

cv2.namedWindow("Debug deteccion de reposo", cv2.WINDOW_NORMAL)

while ret:
    numero_frame += 1

    mascara = segmentar_no_verde(frame)
    dados = detectar_dados(mascara)
    n_dados = len(dados)

    # Detección de reposo: 5 dados cuadrados durante varios frames seguidos
    if n_dados == CANTIDAD_DADOS:
        frames_estables += 1
    else:
        frames_estables = 0

    reposo = frames_estables >= FRAMES_REPOSO

    # Guardo el primer frame donde se confirma el reposo 
    if reposo and not reposo_capturado:
        frame_reposo = frame.copy()
        reposo_capturado = True
        print(f"\n>> Reposo confirmado en el frame {numero_frame}")
        cv2.imwrite("frame_dados_detenidos.jpg", frame_reposo)

    # Panel de debug
    dibujo = frame.copy()
    for i, (x, y, w, h) in enumerate(dados, start=1):
        cv2.rectangle(dibujo, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv2.putText(dibujo, f"D{i}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    estado = "REPOSO" if reposo else "ANALIZANDO"
    color = (0, 255, 0) if reposo else (0, 255, 255)
    cv2.putText(dibujo, f"{estado} | dados={n_dados} | estables={frames_estables}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    mascara_color = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    escala = ALTO_DEBUG / frame.shape[0]
    tam = (int(frame.shape[1] * escala), ALTO_DEBUG)
    panel = cv2.hconcat([cv2.resize(dibujo, tam), cv2.resize(mascara_color, tam)])
    cv2.imshow("Debug deteccion de reposo", panel)

    print(f"Frame {numero_frame:3d}: dados={n_dados} | estables={frames_estables} | reposo={reposo}")

    tecla = cv2.waitKey(30) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord(" "):     
        cv2.waitKey(0)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

if frame_reposo is not None:
    frame_rgb = cv2.cvtColor(frame_reposo, cv2.COLOR_BGR2RGB)
    mascara = segmentar_no_verde(frame_reposo)
    dados = detectar_dados(mascara)

    marcado = frame_rgb.copy()
    for i, (x, y, w, h) in enumerate(dados, start=1):
        cv2.rectangle(marcado, (x, y), (x + w, y + h), (255, 0, 255), 4)
        cv2.putText(marcado, f"D{i}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    plt.figure()
    ax = plt.subplot(131); imshow(frame_rgb, title="Frame de reposo", color_img=True, new_fig=False)
    plt.subplot(132, sharex=ax, sharey=ax); imshow(mascara, title="Máscara no-verde", new_fig=False)
    plt.subplot(133, sharex=ax, sharey=ax); imshow(marcado, title=f"{len(dados)} dados detectados", color_img=True, new_fig=False)
    plt.show(block=True)
else:
    print("No se detectó un frame de reposo en este video")
