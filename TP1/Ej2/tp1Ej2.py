import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


respuestasCorrectas = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']

num_examenes = 5

# Configuramos dimensiones de la imagen de salida
alto_fila = 100
ancho_imagen = 1000
alto_imagen = alto_fila * num_examenes

# Creamos un lienzo blanco (fondo) en formato BGR (3 canales)
canvas = np.ones((alto_imagen, ancho_imagen, 3), dtype=np.uint8) * 255

font = cv2.FONT_HERSHEY_SIMPLEX

# Procesamos los exámenes del 1 al 5
for n in range(1, num_examenes+1):
    print(f"\n===== examen_{n}.png =====")
    img = cv2.imread(f"./Ej2/examen_{n}.png", cv2.IMREAD_GRAYSCALE)
    bordes = bordesExamen(img)

    # Validamos los campos del encabezado
    (l, t), (r, b) = bordes[0]
    encabezado = img[t:b, l:r]
    estado_encabezado = validarEncabezado(encabezado)
    print(f"Name: {estado_encabezado['Name']}")
    print(f"Date: {estado_encabezado['Date']}")
    print(f"Class: {estado_encabezado['Class']}")

    aciertos = 0
    for e in range(1, 11):
        (l, t), (r, b) = bordes[e]
        ejercicio = img[t:b, l:r]

        respuesta = identificarRespuestas(ejercicio)

        if respuesta == respuestasCorrectas[e-1]:
            estado = "OK"
            aciertos += 1
        else:
            estado = "MAL"

        print(f"Pregunta {e}: {estado}")
    if aciertos >= 6:
        print(f"Resultado: APROBADO ({aciertos} aciertos)")
        color_borde = (0, 200, 0) # Verde en BGR
        texto_estado = "APROBADO"
    else:        
        print(f"Resultado: DESAPROBADO ({aciertos} aciertos)")
        color_borde = (0, 0, 200) # Rojo en BGR
        texto_estado = "DESAPROBADO"
    
    y_start = (n-1) * alto_fila

    # Dibujamos el recuadro de la fila con un poco de margen
    cv2.rectangle(canvas, (10, y_start + 10), (ancho_imagen - 10, y_start + alto_fila - 10), color_borde, 2)
    
    crop = obtenerName(encabezado)

    # Lo convertimos el crop a BGR para que coincida con los canales del canvas
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    h_c, w_c = crop_bgr.shape[:2]
    
    # Ajustamos el tamaño del recorte a 50px de alto manteniendo proporción
    h_nuevo = 50
    w_nuevo = int(w_c * (h_nuevo / h_c))
    crop_resized = cv2.resize(crop_bgr, (w_nuevo, h_nuevo))
    
    x_offset_crop = 20
    y_offset_crop = y_start + 25
    
    # Reemplazamos los píxeles en el canvas con el recorte
    canvas[y_offset_crop:y_offset_crop+h_nuevo, x_offset_crop:x_offset_crop+w_nuevo] = crop_resized
    
    # Escribimos el resultado en texto
    x_offset_texto = x_offset_crop + w_nuevo + 20
    texto_final = f"tiene {aciertos} aciertos. Esta {texto_estado}."
    cv2.putText(canvas, texto_final, (x_offset_texto, y_start + 55), font, 0.8, color_borde, 2)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)) # Matplotlib usa RGB así que hay que pasar de BGR a RGB
plt.title('Reporte Final de Calificaciones', fontsize=24)
plt.axis('off')
plt.tight_layout()
plt.show()