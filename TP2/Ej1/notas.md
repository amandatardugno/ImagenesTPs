## Consideraciones importantes hasta ahora:
### Segmentación de la Cinta
* Primero se intentó utilizar el fondo de la cinta para que sea negro y las pastillas blancas al binarizar
* Como esto incluía la sombra (de color más oscuro que la cinta), luego daba problemas en el reconocimiento de pastillas. Por lo tanto, se usó la sombra que da el borde sobre la cinta para segmentar la cinta, generando un área limpia para el posterior análisis.
* Se incluyó un padding a la ROI para evitar bordes difusos.
### Reconocimiento de pastillas
* Al utilizar la conversión estándar (`COLOR_BGR2GRAY`), las pastillas de color azul oscuro se mimetizaban con el fondo y se perdían. Esto ocurre porque la fórmula clásica de luminancia asigna un peso de apenas 11.4% al canal azul ($Y = 0.299R + 0.587G + 0.114B$).
* Para evitar la pérdida de información del color, se optó por convertir la imagen al espacio HSV y trabajar exclusivamente con el canal V (Value). Este canal captura el brillo máximo de los tres colores originales (R, G, B), garantizando que cualquier pastilla de color vibrante (incluido el azul) resalte fuertemente sobre el fondo oscuro de la cinta.
* Se aplicó un umbral sobre el canal V. Se iteró cuál era el mejor umbral para el corte, ya que valores bajos incluían la parte superior de la cinta, más iluminada, en los contornos.
* Tras la detección de contornos (`cv2.findContours`), se implementó un filtrado iterativo para descartar falsos positivos. Se eliminaron todos los componentes conexos cuya área fuera inferior a un umbral mínimo (ej. 10 píxeles).

### Segmentación de la Cinta (Actualización)

  - Originalmente se utilizaba la sombra que da el borde sobre la cinta para segmentarla mediante umbralado. Sin embargo, este método nos pareció muy dependiente de las condiciones de iluminación. Cualquier cambio sutil en la intensidad o dirección de la luz altera el tamaño, y podría incluso eliminar la sombra, haciendo que el Bounding Box calculado sea inestable o impreciso.
  - Para lograr una mayor robustez, cambiamos el enfoque para detectar características geométricas (líneas rectas horizontales) en lugar de depender de intensidades de píxeles:
    1.  Trabajamos sobre el canal V (Value) del espacio HSV, aplicamos un filtro Gaussiano para reducir el ruido y luego el algoritmo de Canny para obtener un mapa binario de los bordes.
    2.  Utilizamos `cv2.HoughLinesP` con el largo de la línea relativo a la imagen (`minLineLength` > 15% del ancho de la imagen) para ignorar texturas pequeñas o pastillas. Pensamos usar un `maxLineGap` alto para conectar tramos de línea fragmentados, pero no fue necesario para la correcta detección de las líneas.
  - Las líneas detectadas se filtran descartando aquellas que no sean horizontales (tolerancia de $\pm5^\circ$ respecto a 0° o 180°).
    - Para asegurar el recorte perfecto de la cinta, se dividen las líneas encontradas en la mitad superior e inferior de la imagen. Se selecciona la línea más baja de la mitad superior (`max(y)`) y la línea más alta de la mitad inferior (`min(y)`), garantizando así capturar los bordes internos de la cinta.
  - Se mantiene la aplicación de un *padding* final a las coordenadas (x, y, w, h) para asegurar un área limpia para la posterior detección de pastillas.


### Clasificación por Color
* Para la detección del color, se continuó aprovechando las ventajas del espacio **HSV**.
* Se aisló cada pastilla iterando sobre los contornos detectados y generando una máscara binaria específica para cada una.
* Utilizando esta máscara, se calculó el valor promedio de los canales H (Matiz) y S (Saturación) únicamente sobre los píxeles que pertenecen a la pastilla, ignorando cualquier interferencia del fondo oscuro.
* **Pastillas Blancas:** Se determinó que el color blanco no se define por un matiz en el canal H, sino por una falta de saturación. Por ende, se aplicó un primer filtro lógico sobre el canal S: si la saturación promedio es muy baja, se clasifica directamente como "Blanca".
* **Pastillas de Color:** Para las pastillas con saturación alta, se evaluó el canal H (cuyos valores en OpenCV van de 0 a 179) para asignarlas a sus respectivas clases (Roja, Naranja, Azul, Rosa). Se contempló la particularidad de que el color rojo se sitúa en los dos extremos del espectro (cerca de 0 y de 179).
* Para evitar el uso de valores arbitrarios (*hardcoding* a ciegas), se desarrolló una función de visualización (*Scatter Plot* de H vs S) renderizada sobre un fondo con el espectro de color HSV real. Esto permitió visualizar la formación natural de *clusters* y definir empíricamente los umbrales de corte con alta precisión.

### Clasificación por Forma
* Se identificó la geometría utilizando la relación matemática entre el área y el perímetro del contorno cerrado para calcular el **Factor de Forma**: $FP = \frac{\text{Área}}{\text{Perímetro}^2}$.
* Se graficaron histogramas y un gráfico de dispersión (Área vs Factor de Forma) para contrastar los resultados obtenidos frente a los valores teóricos de referencia para el círculo ($\approx 0.079$) y el cuadrado ($0.0625$).
* Observando los *clusters* generados en este espacio geométrico, se definieron los umbrales de corte para catalogar definitivamente cada píldora en tres categorías: Redonda, Cuadrada o Alargada.

### Estructuración de Datos y Etiquetado
* Para manejar los datos extraídos de manera limpia y robusta, se adoptó un enfoque de Programación Orientada a Objetos (POO). 
* Se construyó la clase `Pastilla`, la cual encapsula e inicializa internamente todas las propiedades físicas (área, perímetro, factor de forma, centroide), así como su clasificación final combinada de color y forma geométrica (ej. "RR" para Roja Redonda).
* Dentro de la instanciación de esta clase, se calcularon los **momentos espaciales** de OpenCV (`cv2.moments`) para encontrar el centroide $(x, y)$ del contorno. Con esta coordenada exacta se logró posicionar dinámicamente el texto de la etiqueta final sobre la imagen. 
* El conteo total y la asignación de IDs se resolvió en el flujo principal implementando un diccionario dinámico, lo cual permitió asignar un número secuencial exacto a las pastillas procesadas y cumplir con el formato de salida requerido por el enunciado (ej. RR1, RR2, CB1).