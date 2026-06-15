import cv2
import matplotlib.pyplot as plt
from helpers import (
    normalizar_imagen,
    aplicar_blackhat,
    binarizar_adaptativo,
    filtrar_contornos,
    encontrar_patente,
    cargar_templates,
    cargar_templates,
    leer_caracteres_patente,
    refinar_patente
)

mis_templates = cargar_templates("templates")

# Procesamos las imágenes
for id_imagen in range(1, 13):
    img_path = f"../data/img_{id_imagen}.jpg"
    img_color = cv2.imread(img_path)

    print(f"\nProcesando img_{id_imagen}.jpg")

    if img_color is None:
        print(f"ERROR: No se pudo cargar la imagen en '{img_path}'.")
        continue

    # Preprocesamiento
    img_color = normalizar_imagen(img_color, ancho_estandar=1500)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    img_bh = aplicar_blackhat(img_gray, kernel_size=(15, 15))
    thresh = binarizar_adaptativo(img_bh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detectamos candidatos y buscamos el grupo de 7 caracteres
    contornos_detectados = filtrar_contornos(thresh)
    letras_detectadas = encontrar_patente(contornos_detectados)

    if len(letras_detectadas) != 7:
        print("No se pudo detectar una patente con 7 caracteres.")
        continue

    # Refinamos la segmentación de los caracteres
    letras_detectadas, _ = refinar_patente(
        img_gray,
        letras_detectadas
    )

    if len(letras_detectadas) != 7:
        print("No se pudieron segmentar correctamente los 7 caracteres.")
        continue

    # Reconocimiento adicional mediante Template Matching
    texto_patente = leer_caracteres_patente(
        thresh,
        letras_detectadas,
        mis_templates
    )

    print("Patente detectada correctamente.")
    print(f"Caracteres segmentados: {len(letras_detectadas)}")
    print(f"Texto reconocido: {texto_patente}")

    # Imagen con todos los candidatos detectados
    img_candidatos = img_color.copy()

    for x, y, w, h, area in contornos_detectados:
        cv2.rectangle(
            img_candidatos,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Calculamos la región completa de la patente
    min_x = min(x for x, y, w, h, area in letras_detectadas)
    min_y = min(y for x, y, w, h, area in letras_detectadas)
    max_x = max(x + w for x, y, w, h, area in letras_detectadas)
    max_y = max(y + h for x, y, w, h, area in letras_detectadas)

    padding_x = 15
    padding_y = 10

    x1 = max(0, min_x - padding_x)
    y1 = max(0, min_y - padding_y)
    x2 = min(img_color.shape[1], max_x + padding_x)
    y2 = min(img_color.shape[0], max_y + padding_y)

    patente_color = img_color[y1:y2, x1:x2].copy()

    # Imagen final con la patente y sus caracteres señalados
    img_resultado = img_color.copy()

    cv2.rectangle(
        img_resultado,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    for x, y, w, h, area in letras_detectadas:
        cv2.rectangle(
            img_resultado,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        img_resultado,
        texto_patente,
        (x1, max(30, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Mostramos todas las etapas en una única figura
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(
        3,
        14,
        height_ratios=[3, 3, 1.5]
    )

    # Imagen original
    ax_original = fig.add_subplot(grid[0, 0:4])
    ax_original.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    ax_original.set_title("Imagen original")
    ax_original.axis("off")

    # Transformación Black Hat
    ax_blackhat = fig.add_subplot(grid[0, 4:7])
    ax_blackhat.imshow(img_bh, cmap="gray")
    ax_blackhat.set_title("Black Hat")
    ax_blackhat.axis("off")

    # Umbralización adaptativa
    ax_umbral = fig.add_subplot(grid[0, 7:10])
    ax_umbral.imshow(thresh, cmap="gray")
    ax_umbral.set_title("Umbral adaptativo")
    ax_umbral.axis("off")

    # Componentes candidatos
    ax_candidatos = fig.add_subplot(grid[0, 10:14])
    ax_candidatos.imshow(
        cv2.cvtColor(img_candidatos, cv2.COLOR_BGR2RGB)
    )
    ax_candidatos.set_title(
        f"Candidatos detectados: {len(contornos_detectados)}"
    )
    ax_candidatos.axis("off")

    # Resultado final
    ax_resultado = fig.add_subplot(grid[1, 0:9])
    ax_resultado.imshow(
        cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
    )
    ax_resultado.set_title(
        f"Patente reconocida: {texto_patente}"
    )
    ax_resultado.axis("off")

    # Patente completa segmentada
    ax_patente = fig.add_subplot(grid[1, 9:14])
    ax_patente.imshow(
        cv2.cvtColor(patente_color, cv2.COLOR_BGR2RGB)
    )
    ax_patente.set_title("Patente segmentada")
    ax_patente.axis("off")

    # Caracteres individuales
    for i, (x, y, w, h, area) in enumerate(letras_detectadas):
        ax_caracter = fig.add_subplot(grid[2, i * 2:(i + 1) * 2])

        recorte = thresh[y:y + h, x:x + w]

        ax_caracter.imshow(recorte, cmap="gray")
        ax_caracter.set_title(f"Carácter {i + 1}")
        ax_caracter.axis("off")

    fig.suptitle(
        f"Procesamiento de img_{id_imagen}.jpg",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()
    plt.close(fig)