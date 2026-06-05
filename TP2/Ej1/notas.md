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