import cv2

from helpers import dados_estan_quietos


video_path = "data/tirada_1.mp4"
captura = cv2.VideoCapture(video_path)

ret, frame_anterior = captura.read()

if not ret:
    print(f"ERROR: No se pudo leer el video '{video_path}'.")
    captura.release()
    exit()

numero_frame = 1
frames_estables = 0
FRAMES_ESTABLES_REQUERIDOS = 10

frame_reposo_mostrado = False

while True:
    ret, frame_actual = captura.read()

    if not ret:
        break

    quietos, cambio_mascara, cambio_visual = dados_estan_quietos(
        frame_anterior,
        frame_actual
    )

    if quietos:
        frames_estables += 1
    else:
        frames_estables = 0

    reposo_confirmado = (
        frames_estables >= FRAMES_ESTABLES_REQUERIDOS
    )

    if reposo_confirmado and not frame_reposo_mostrado:
        captura_reposo = frame_actual.copy()

        cv2.putText(
            captura_reposo,
            f"REPOSO CONFIRMADO - Frame {numero_frame}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
    )

        # Mostramos únicamente el primer frame confirmado como estático
        cv2.imshow(
            "Frame con dados detenidos",
            captura_reposo
        )

        # Se guarda
        cv2.imwrite(
            "frame_dados_detenidos.jpg",
            captura_reposo
        )

        print(
            f"\nFrame de reposo capturado: {numero_frame}"
        )

        frame_reposo_mostrado = True

    # Diferencia visual entre ambos frames
    diferencia = cv2.absdiff(
        frame_anterior,
        frame_actual
    )

    diferencia_gris = cv2.cvtColor(
        diferencia,
        cv2.COLOR_BGR2GRAY
    )

    diferencia_color = cv2.cvtColor(
        diferencia_gris,
        cv2.COLOR_GRAY2BGR
    )

    # Copias para escribir información sin modificar los frames originales
    frame_anterior_debug = frame_anterior.copy()
    frame_actual_debug = frame_actual.copy()

    estado = "QUIETOS" if quietos else "MOVIMIENTO"
    color_estado = (0, 255, 0) if quietos else (0, 0, 255)

    estado_confirmado = (
        "REPOSO CONFIRMADO"
        if reposo_confirmado
        else "ANALIZANDO"
    )

    cv2.putText(
        frame_anterior_debug,
        f"Frame anterior: {numero_frame - 1}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_actual_debug,
        f"Frame actual: {numero_frame}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_actual_debug,
        estado,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_estado,
        3
    )

    cv2.putText(
        frame_actual_debug,
        f"Cambio mascara: {cambio_mascara:.4f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_actual_debug,
        f"Cambio visual: {cambio_visual:.2f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_actual_debug,
        f"Frames estables: {frames_estables}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_actual_debug,
        estado_confirmado,
        (20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color_estado if reposo_confirmado else (0, 255, 255),
        2
    )

    cv2.putText(
        diferencia_color,
        "Diferencia entre frames",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Ajustamos las imágenes para mostrarlas juntas
    ancho_muestra = 640

    proporcion = ancho_muestra / frame_actual.shape[1]
    alto_muestra = int(frame_actual.shape[0] * proporcion)

    frame_anterior_debug = cv2.resize(
        frame_anterior_debug,
        (ancho_muestra, alto_muestra)
    )

    frame_actual_debug = cv2.resize(
        frame_actual_debug,
        (ancho_muestra, alto_muestra)
    )

    diferencia_color = cv2.resize(
        diferencia_color,
        (ancho_muestra, alto_muestra)
    )

    comparacion_frames = cv2.hconcat([
        frame_anterior_debug,
        frame_actual_debug
    ])

    panel_debug = cv2.vconcat([
        comparacion_frames,
        cv2.hconcat([
            diferencia_color,
            diferencia_color
        ])
    ])

    cv2.imshow(
        "Debug deteccion de movimiento",
        panel_debug
    )

    print(
        f"Frame {numero_frame}: "
        f"quietos={quietos} | "
        f"cambio_mascara={cambio_mascara:.4f} | "
        f"cambio_visual={cambio_visual:.2f} | "
        f"frames_estables={frames_estables} | "
        f"reposo_confirmado={reposo_confirmado}"
    )

    tecla = cv2.waitKey(30) & 0xFF

    if tecla == ord("q"):
        break

    # Con espacio se pausa o reanuda el video
    if tecla == ord(" "):
        cv2.waitKey(0)

    frame_anterior = frame_actual.copy()
    numero_frame += 1

captura.release()
cv2.destroyAllWindows()