## Consideraciones importantes hasta ahora:

### Preprocesamiento y Resaltado de Caracteres

* Inicialmente se evaluó la posibilidad de binarizar la imagen directamente o usar la transformación Top Hat. Sin embargo, dado que las patentes del Mercosur son caracteres negros sobre un fondo blanco, la operación ideal es el Black Hat. Esta transformación extrae elementos oscuros sobre un fondo claro.
* Se decidió aplicar el Black Hat sobre la imagen original en escala de grises antes de binarizar. Si se binarizaba primero, las variaciones de iluminación dificultaban la detección de los caracteres. Al aplicar Black Hat en grises, se aplana la iluminación y se dejan los caracteres listos y limpios.
* Al observar distintas imágenes, se encontró que por diferentes resoluciones, había algunas patentes que no reconocía. Lo asociamos a un problema de usar kernels y algunas dimensiones fijas, haciendo que se comporte distinto según la imagen (particularmente, la imagen 7 se procesaba muy lentamente y encontraba muchísimos contornos, a la vez que subdividía y no lograba encontrar las letras de la patente).
* Un umbral global fallaba estrepitosamente en imágenes con sombras o iluminación dispareja sobre la patente (ejemplo de la imagen 12). Se optó por `cv2.adaptiveThreshold` (umbralización local), el cual calcula el umbral basándose en el vecindario de cada píxel, logrando separar las letras del fondo sin afectar la iluminación.
* Se evaluó incluir una Apertura (para limpiar ruido) o Clausura (para rellenar letras huecas). Sin embargo, el pipeline de Black Hat + Umbral Adaptativo demostró ser lo suficientemente robusto. Se omitieron estas operaciones para mantener el código más rápido y evitar el riesgo de deformar los caracteres (por ejemplo, partir una letra por la mitad o fusionar dos letras cercanas).

### Detección y Filtrado de Candidatos (Letras)

* Inicialmente se pensó en agrupar toda la patente en un solo gran rectángulo blanco mediante dilataciones/clausuras, pero se prefirió resolver el problema detectando los caracteres individuales usando `cv2.connectedComponentsWithStats`. Esto provee automáticamente las bounding boxes y el área de cada componente en un solo paso.
* Basándonos en las medidas oficiales de la patente del Mercosur, se sabe que los caracteres son más altos que anchos, con una relación de aspecto bastante particular. Se le aplicó un filtro a ésta (`aspect_ratio = h / w`) entre `1.5` y `5.0`, lo que elimina manchas asimétricas, líneas horizontales del paragolpes o contornos del fondo.
* Se descartaron componentes menores a 50 píxeles de área y componentes masivamente grandes (mayores al 5% del fondo) para descartar ruido y reflejos grandes del auto o el fondo.

### Agrupamiento Lógico y Segmentación Final

* Aún filtrando por forma y área, partes del auto (como logos, letras de la marca del auto, ranuras de la parrilla, o letras del fondo) superaban los filtros individuales.
* Para garantizar que las letras detectadas sean realmente la patente, se implementó un algoritmo de agrupamiento por renglón:
    1.  Partimos de que las letras de una misma patente comparten el mismo renglón. Se exigió que la diferencia en la coordenada Y entre letras no supere el 40% de su altura. Es bastante permisivo para considerar patentes medianamente torcidas.
    2.  Se exigió que la diferencia de altura entre caracteres no supere el 10% (deberían ser todos iguales pero damos margen).
    3.  Como el formato es "AB 123 CD", hay espacios grandes entre bloques de caracteres, pero reglamentados. La tolerancia en el eje X se ajustó a 10 veces el ancho máximo de letra, permitiendo que el algoritmo entienda que la "A" (primera letra) y la "D" (última) pertenecen a la misma patente a pesar del espacio central. El valor por reglamento da entre 7.5 y 8.5 anchos la diferencia entre la primera y última letra, pero damos margen.
* La condición final de éxito exige que el grupo formado tenga exactamente 7 elementos (`len(grupo_actual) == 7`). Una vez encontrado el grupo válido, se ordena de izquierda a derecha usando su coordenada X, dejándolo listo para mostrar segmentado por caracter en orden.

### Reconocimiento Óptico de Caracteres (OCR) mediante Template Matching

* Para la transcripción de las patentes segmentadas a texto, se descartó el uso de librerías de IA o externas (como PyTesseract) priorizando una solución pura en OpenCV mediante la función `cv2.matchTemplate`. Esto permite un control mayor sobre el proceso de reconocimiento.
* El algoritmo de Template Matching es extremadamente sensible a las diferencias de escala (compara píxel a píxel). Para solucionar esto, se desarrolló un script auxiliar que genera imágenes "template" perfectas de cada letra (A-Z) y número (0-9) en blanco y negro, fijándolas a un tamaño estandarizado de 45x65 píxeles (proporcional al tamaño reglamentario). Durante la ejecución del pipeline principal, cada caracter recortado de la patente se redimensiona a este mismo tamaño exacto antes de la comparación.
* Al recortar los caracteres de la imagen binarizada, se observó que algunos presentaban discontinuidades o huecos internos en el trazo debido a desgastes en la pintura de la chapa o reflejos puntuales. Para asegurar una correlación alta con los templates sólidos, se aplicó una operación morfológica de Clausura (kernel de 3x3) únicamente sobre el recorte del caracter. Esto actuó como un filtro de soldadura, rellenando imperfecciones y engrosando el trazo.
* Uno de los errores más comunes en el OCR clásico es la confusión de caracteres visualmente similares (ej. "B" con "8", o "O" con "0"). Para erradicar este problema, se implementó una lógica de filtrado posicional aprovechando que la patente del Mercosur tiene un patrón estricto: Letra-Letra-Número-Número-Número-Letra-Letra. 
  - Usando el índice iterativo de la detección, si la posición correspondía a una letra (0, 1, 5, 6), el recorte solo se comparaba contra el subconjunto de templates de letras (`.isalpha()`).
  - Si correspondía a un número (2, 3, 4), se comparaba exclusivamente contra números (`.isdigit()`).
  Esta validación contextual eliminó los falsos positivos y redujo la cantidad de comparaciones matemáticas necesarias por caracter, optimizando el rendimiento general del algoritmo.

### Refinamiento Estructural: Detección Refinada de los caracteres

* Durante las pruebas iniciales, se detectó que algunos caracteres quedaban fragmentados tras la binarización (por ejemplo, el desgaste en la pintura hacía que una "T" o una "I" se detectara como dos manchas separadas). La solución clásica sería aplicar una operación morfológica de Clausura a toda la imagen para rellenar o unir estos trazos. Sin embargo, esto se descartó categóricamente a nivel global: aplicar una clausura sobre toda la imagen provocaba que, en las fotos donde el auto estaba más alejado, el espacio entre las letras fuera muy poco para el kernel, fusionando múltiples caracteres en un solo bloque insegmentable o con los bordes de la patente.
* Para solucionar los caracteres rotos sin afectar la separación entre ellos, diseñamos una arquitectura de segmentación en dos fases aislando el problema en un entorno controlado:
  1. Primero se realiza la detección general tolerando imperfecciones. Una vez encontradas las coordenadas aproximadas del grupo de caracteres, se calcula una Bounding Box global que envuelve a toda la patente, añadiendo un margen (padding) de seguridad proporcional (1/14 del ancho y 0.1 del alto).
  2. Se recorta esta región de la imagen original en escala de grises y se redimensiona por interpolación a un tamaño estandarizado estricto (W=450px, H=150px). Este tamaño se calculó estimando el ancho relativo de 7 caracteres más los 6 espacios de la norma Mercosur aprox.
  3. Al operar sobre este parche de tamaño fijo, se elimina la varianza de escala. Ahora sí, fue seguro aplicar un Black Hat, binarizar (de nuevo) y realizar una Clausura con un kernel específico (9x9). Como la patente siempre mide lo mismo en este recorte estandarizado, el kernel garantiza soldar roturas finas (de 3 a 4 píxeles) dentro de una letra partida, siendo físicamente incapaz de cruzar los amplios espacios que separan a dos letras distintas.
  4. Finalmente, se redetectan los contornos sobre el parche limpio y las nuevas coordenadas se transforman de regreso al sistema de coordenadas de la imagen original de alta resolución.

### Problemas aún sin resolver:
* La imagen 7 tiene su última letra como una T despintada. El código como está, la recorta como si fuera una I.
* Se intentó usar una clausura y una apertura luego del binarizado para corregir este problema, pero hacía que no se detecte la patente de la imagen 11.
* Intentamos detectar las patentes comparando con un template de la fuente usada legalmente (FE-Scrift). Detecta en muchos casos bien, pero en muchos otros falla. ¿Probamos pyTesseract a ver si soluciona?