## Consideraciones importantes hasta ahora:
### Segmentación de la Cinta
* Primero se intentó utilizar el fondo de la cinta para que sea negro y las pastillas blancas al binarizar
* Como esto incluía la sombra (de color más oscuro que la cinta), luego daba problemas en el reconocimiento de pastillas. Por lo tanto, se usó la sombra que da el borde sobre la cinta para segmentar la cinta, generando un área limpia para el posterior análisis.
* Se incluyó un padding a la ROI para evitar bordes difusos.
### Reconocimiento de pastillas
### Reconocimiento de pastillas
* Al utilizar la conversión estándar (`COLOR_BGR2GRAY`), las pastillas de color azul oscuro se mimetizaban con el fondo y se perdían. Esto ocurre porque la fórmula clásica de luminancia asigna un peso de apenas 11.4% al canal azul ($Y = 0.299R + 0.587G + 0.114B$).
* Para evitar la pérdida de información del color, se optó por convertir la imagen al espacio HSV y trabajar exclusivamente con el canal V (Value). Este canal captura el brillo máximo de los tres colores originales (R, G, B), garantizando que cualquier pastilla de color vibrante (incluido el azul) resalte fuertemente sobre el fondo oscuro de la cinta.
* Se aplicó un umbral sobre el canal V. Se iteró cuál era el mejor umbral para el corte, ya que valores bajos incluían la parte superior de la cinta, más iluminada, en los contornos.
* Tras la detección de contornos (`cv2.findContours`), se implementó un filtrado iterativo para descartar falsos positivos. Se eliminaron todos los componentes conexos cuya área fuera inferior a un umbral mínimo (ej. 10 píxeles).