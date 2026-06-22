import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from helpers import segmentar_no_verde, detectar_dados, contar_puntos, CANTIDAD_DADOS, RastreadorDados

# Configuración general
VIDEOS = [f"data/tirada_{i}.mp4" for i in range(1, 5)]
FRAMES_REPOSO = 4
CARPETA_SALIDA = "outputs"

# Creamos la carpeta de salida automáticamente si no existe
if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)

def procesar_video(video_path):
    print(f"\nProcesando {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir {video_path}")
        return None, None

    # Obtenemos info del video original
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    # Inicializamos variables del tracking
    rastreador = RastreadorDados()
    puntos_anteriores = {}
    frames_estables = 0
    reposo_capturado = False
    
    recorte_final = None
    resultados_tirada = {}
    out = None 

    numero_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        numero_frame += 1

        mascara = segmentar_no_verde(frame)
        bboxes_crudos = detectar_dados(mascara)
        dados_trackeados = rastreador.actualizar(bboxes_crudos)
        n_dados = len(dados_trackeados)
        
        dibujo = frame.copy()
        puntos_actuales = {}

        for dado in dados_trackeados:
            x, y, w, h = dado.bbox
            
            # Si el dado está temporalmente perdido
            if dado.frames_perdido > 0:
                color_box = (0, 255, 255) # Amarillo
                cant_puntos = dado.puntos
                puntos_actuales[dado.id] = cant_puntos
                cv2.rectangle(dibujo, (x, y), (x + w, y + h), color_box, 3)
                cv2.putText(dibujo, f"D{dado.id}-{cant_puntos}", (x, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_box, 2)
                continue

            # Procesamiento normal si está visible
            roi_dado = frame[y:y+h, x:x+w]
            cant_puntos, keypoints, _ = contar_puntos(roi_dado)
            
            dado.puntos = cant_puntos
            puntos_actuales[dado.id] = cant_puntos

            if cant_puntos >= 1:
                color_box = (0, 255, 0) # Verde
                for kp in keypoints:
                    centro = (int(kp.pt[0]) + x, int(kp.pt[1]) + y)
                    radio = int(kp.size / 2)
                    cv2.circle(dibujo, centro, radio, (255, 0, 0), 3) # Puntos azules
            else:
                color_box = (255, 0, 255) # Magenta

            cv2.rectangle(dibujo, (x, y), (x + w, y + h), color_box, 4)
            cv2.putText(dibujo, f"D{dado.id}-{cant_puntos}", (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_box, 3)

        # Lógica de reposo
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

        # Captura del momento exacto del reposo
        if reposo and not reposo_capturado:
            reposo_capturado = True
            resultados_tirada = puntos_actuales.copy()
            
            print(f">> ¡Reposo confirmado en el frame {numero_frame}!")
            for id_dado, pts in sorted(resultados_tirada.items()):
                print(f"   D{id_dado}: {pts} puntos")
                
            xs = [d.bbox[0] for d in dados_trackeados]
            ys = [d.bbox[1] for d in dados_trackeados]
            x_maxs = [d.bbox[0] + d.bbox[2] for d in dados_trackeados]
            y_maxs = [d.bbox[1] + d.bbox[3] for d in dados_trackeados]
            
            pad = 40
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(frame.shape[1], max(x_maxs) + pad)
            y2 = min(frame.shape[0], max(y_maxs) + pad)
            
            recorte_final = dibujo[y1:y2, x1:x2].copy()

        estado = "REPOSO" if reposo else "ANALIZANDO"
        color_texto = (0, 255, 0) if reposo else (0, 255, 255)
        cv2.putText(dibujo, f"{estado} | dados={n_dados} | estables={frames_estables}/{FRAMES_REPOSO}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_texto, 3)

        # Inicializamos el grabador 
        if out is None:
            alto_orig, ancho_orig = dibujo.shape[:2]
            nombre_salida = os.path.join(CARPETA_SALIDA, f"output_{os.path.basename(video_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(nombre_salida, fourcc, fps, (ancho_orig, alto_orig))
            
        out.write(dibujo)

    cap.release()
    if out is not None:
        out.release()
        
    if not reposo_capturado:
        print(">> ADVERTENCIA: Nunca se alcanzó el estado de reposo en este video.")
        
    return recorte_final, resultados_tirada

imagenes_recortadas = []

for video in VIDEOS:
    if not os.path.exists(video):
        print(f"No se encontró el archivo: {video}")
        continue
        
    recorte, resultados = procesar_video(video)
    
    if recorte is not None:
        recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
        imagenes_recortadas.append((video, recorte_rgb, resultados))

# Generamos la visualización final con matplotlib
if imagenes_recortadas:
    print("\nTodos los videos procesados. Generando visualización final...")
    n_imgs = len(imagenes_recortadas)
    fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
    
    if n_imgs == 1:
        axes = [axes]
        
    for ax, (video_path, img, res) in zip(axes, imagenes_recortadas):
        ax.imshow(img)
        ax.axis('off')
        
        res_str = ", ".join([f"D{k}:{v}" for k, v in sorted(res.items())])
        titulo = f"{os.path.basename(video_path)}\n{res_str}"
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    
    # Guardamos el mosaico de matplotlib en la carpeta outputs
    ruta_grafico = os.path.join(CARPETA_SALIDA, "mosaico_resultados.png")
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f">> Mosaico final guardado con éxito en: {ruta_grafico}")
    
    plt.show()
else:
    print("\nNo se pudo extraer el reposo de ningún video.")

print("\n¡Estadio cerrado! Todo el procesamiento del TP3 está guardado en /outputs.")