"""
Script de visualizacion paso a paso del pipeline del Ej2.

Cada seccion llama a una funcion del pipeline (helpers + main) y muestra
con matplotlib que hace, para ir sacando capturas y armar el reporte final.
Se usa examen_5.png como caso de ejemplo (Name: JUAN PEREZ, 10/10 correctas).

Correr desde la raiz del repo:
    python Ej2/capturas_proceso.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helpers import (
    bordesExamen,
    detectarCamposEncabezado,
    validarEncabezado,
    contarPalabras,
    obtenerName,
    encontrarRespuestas,
    extraerCaracteristicasLetra,
    identificarRespuestas,
    GLOBAL_THRESHOLD,
)

EXAMEN = "./Ej2/examen_4.png"
RESPUESTAS_CORRECTAS = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']

# Entrada
img = cv2.imread(EXAMEN, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 10))
plt.imshow(img, cmap='gray')
plt.title('0. Imagen de entrada (examen_4.png)')
plt.axis('off')
plt.tight_layout()
plt.show()


# bordesExamen: detecta el encabezado y las 10 celdas de preguntas
bordes = bordesExamen(img)

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(img, cmap='gray')
for i, ((l, t), (r, b)) in bordes.items():
    color = 'red' if i == 0 else 'cyan'
    ax.add_patch(patches.Rectangle((l, t), r-l, b-t,
                                   linewidth=1.5, edgecolor=color, facecolor='none'))
    ax.text(l+4, t+14, str(i), color=color, fontsize=10, fontweight='bold')
ax.set_title('1. bordesExamen: rojo = encabezado (0), cyan = 10 preguntas (1..10)')
ax.axis('off')
plt.tight_layout()
plt.show()


# detectarCamposEncabezado: encuentra las 3 lineas y agrupa chars por campo
(l, t), (r, b) = bordes[0]
encabezado = img[t:b, l:r]
lineas, campos = detectarCamposEncabezado(encabezado)

colores = ['red', 'green', 'blue']
nombres = ['Name', 'Date', 'Class']

fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(encabezado, cmap='gray')
for i, ((xl, yl, wl, hl), chars) in enumerate(zip(lineas, campos)):
    # Linea del campo
    ax.add_patch(patches.Rectangle((xl, yl-1), wl, 3,
                                   edgecolor=colores[i], facecolor='none', linewidth=2))
    # Bounding box de cada caracter detectado sobre la linea
    for x, y, w, h in chars:
        ax.add_patch(patches.Rectangle((x, y), w, h,
                                       edgecolor=colores[i], facecolor='none', linewidth=0.8))
    ax.text(xl, yl+14, nombres[i], color=colores[i], fontsize=11, fontweight='bold')
ax.set_title('2. detectarCamposEncabezado: 3 lineas + chars agrupados por campo')
ax.axis('off')
plt.tight_layout()
plt.show()


# validarEncabezado: aplica las reglas del enunciado y devuelve OK/MAL
estado = validarEncabezado(encabezado)

fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(encabezado, cmap='gray')
for i, ((xl, yl, wl, hl), chars) in enumerate(zip(lineas, campos)):
    color = 'green' if estado[nombres[i]] == 'OK' else 'red'
    ax.add_patch(patches.Rectangle((xl, yl-1), wl, 3,
                                   edgecolor=color, facecolor='none', linewidth=2))
    ax.text(xl, yl+14, f"{nombres[i]}: {estado[nombres[i]]}",
            color=color, fontsize=11, fontweight='bold')
ax.set_title('3. validarEncabezado: estado OK/MAL por campo')
ax.axis('off')
plt.tight_layout()
plt.show()


# contarPalabras: separa palabras midiendo gaps horizontales grandes
name_chars = campos[0]
alto_prom = sum(c[3] for c in name_chars) / len(name_chars)
th_gap = alto_prom * 0.45
palabras = contarPalabras(name_chars)

fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(encabezado, cmap='gray')
for i, (x, y, w, h) in enumerate(name_chars):
    # Bounding box del caracter
    ax.add_patch(patches.Rectangle((x, y), w, h,
                                   edgecolor='blue', facecolor='none', linewidth=0.8))
    if i > 0:
        x_prev, _, w_prev, _ = name_chars[i-1]
        gap = x - (x_prev + w_prev)
        if gap > th_gap:
            # Gap grande: separador de palabras (rojo)
            ax.add_patch(patches.Rectangle((x_prev+w_prev, y), gap, h,
                                           edgecolor='red', facecolor='red', alpha=0.4))
ax.set_title(f"4. contarPalabras: {palabras} palabra(s) en Name — gaps en rojo superan {th_gap:.1f}px (alto_prom*0.45)")
ax.axis('off')
plt.tight_layout()
plt.show()

# obtenerName: recorta solo el campo Name del encabezado
name_crop = obtenerName(encabezado)

fig, axs = plt.subplots(1, 2, figsize=(14, 3))
axs[0].imshow(encabezado, cmap='gray')
axs[0].set_title('Encabezado original')
axs[0].axis('off')
axs[1].imshow(name_crop, cmap='gray')
axs[1].set_title('5. obtenerName: crop del campo Name')
axs[1].axis('off')
plt.tight_layout()
plt.show()

# encontrarRespuestas: detecta la letra marcada sobre la linea del ejercicio
# Se usa la pregunta 1 como ejemplo
(l, t), (r, b) = bordes[1]
ejercicio = img[t:b, l:r]
respuestas = encontrarRespuestas(ejercicio)

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(ejercicio, cmap='gray')
for x, y, w, h, area in respuestas:
    ax.add_patch(patches.Rectangle((x, y), w, h,
                                   edgecolor='red', facecolor='none', linewidth=2))
ax.set_title('6. encontrarRespuestas: componentes sobre la linea de respuesta')
ax.axis('off')
plt.tight_layout()
plt.show()


# extraerCaracteristicasLetra: cuenta los agujeros de la letra
x, y, w, h, area = respuestas[0]
margen = 2
letra_roi = ejercicio[max(0, y-margen):y+h+margen, max(0, x-margen):x+w+margen]
agujeros, area_agujero, _ = extraerCaracteristicasLetra(letra_roi)

# Para visualizar, dibujamos los contornos hijos (agujeros) en rojo sobre la letra
_, letra_th = cv2.threshold(letra_roi, GLOBAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(letra_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
letra_color = cv2.cvtColor(letra_roi, cv2.COLOR_GRAY2RGB)
if hierarchy is not None:
    for i, c in enumerate(contours):
        if hierarchy[0, i, 3] != -1: 
            cv2.drawContours(letra_color, [c], -1, (255, 0, 0), 1)

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(letra_roi, cmap='gray')
axs[0].set_title('Letra recortada')
axs[0].axis('off')
axs[1].imshow(letra_color)
axs[1].set_title(f'Agujeros: {agujeros}, area max: {area_agujero:.0f}')
axs[1].axis('off')
plt.suptitle('7. extraerCaracteristicasLetra')
plt.tight_layout()
plt.show()


# identificarRespuestas + comparacion con las respuestas correctas
fig, axs = plt.subplots(2, 5, figsize=(15, 5))
aciertos = 0
for e in range(1, 11):
    (l, t), (r, b) = bordes[e]
    ejercicio = img[t:b, l:r]
    respuesta = identificarRespuestas(ejercicio)
    correcto = respuesta == RESPUESTAS_CORRECTAS[e-1]
    if correcto:
        aciertos += 1

    color = 'green' if correcto else 'red'
    estado_preg = 'OK' if correcto else 'MAL'
    ax = axs[(e-1)//5, (e-1)%5]
    ax.imshow(ejercicio, cmap='gray')
    ax.set_title(f'P{e}: {respuesta} ({estado_preg})', color=color, fontsize=11)
    ax.axis('off')
plt.suptitle(f'8. identificarRespuestas + comparacion: {aciertos}/10 aciertos', fontsize=14)
plt.tight_layout()
plt.show()


# Reporte final
num_examenes = 5
alto_fila = 100
ancho_canvas = 1000
canvas = np.ones((alto_fila * num_examenes, ancho_canvas, 3), dtype=np.uint8) * 255
font = cv2.FONT_HERSHEY_SIMPLEX

for n in range(1, num_examenes+1):
    img_n = cv2.imread(f"./Ej2/examen_{n}.png", cv2.IMREAD_GRAYSCALE)
    bordes_n = bordesExamen(img_n)

    aciertos_n = 0
    for e in range(1, 11):
        (l, t), (r, b) = bordes_n[e]
        ejercicio = img_n[t:b, l:r]
        if identificarRespuestas(ejercicio) == RESPUESTAS_CORRECTAS[e-1]:
            aciertos_n += 1

    if aciertos_n >= 6:
        color_borde = (0, 200, 0)  # verde BGR
        texto_estado = "APROBADO"
    else:
        color_borde = (0, 0, 200)  # rojo BGR
        texto_estado = "DESAPROBADO"

    y_start = (n-1) * alto_fila
    cv2.rectangle(canvas, (10, y_start + 10),
                  (ancho_canvas - 10, y_start + alto_fila - 10), color_borde, 2)

    (l, t), (r, b) = bordes_n[0]
    encabezado_n = img_n[t:b, l:r]
    crop = obtenerName(encabezado_n)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    h_c, w_c = crop_bgr.shape[:2]
    h_nuevo = 50
    w_nuevo = int(w_c * (h_nuevo / h_c))
    crop_resized = cv2.resize(crop_bgr, (w_nuevo, h_nuevo))

    x_off = 20
    y_off = y_start + 25
    canvas[y_off:y_off+h_nuevo, x_off:x_off+w_nuevo] = crop_resized
    cv2.putText(canvas, f"tiene {aciertos_n} aciertos. Esta {texto_estado}.",
                (x_off + w_nuevo + 20, y_start + 55), font, 0.8, color_borde, 2)

plt.show()
