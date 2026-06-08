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

### Problemas aún sin resolver:
* La imagen 7 tiene su última letra como una T despintada. El código como está, la recorta como si fuera una I.
* Se intentó usar una clausura y una apertura luego del binarizado para corregir este problema, pero hacía que no se detecte la patente de la imagen 11.
* Intentamos detectar las patentes comparando con un template de la fuente usada legalmente (FE-Scrift). Detecta en muchos casos bien, pero en muchos otros falla. ¿Lo omitimos, lo hacemos con algunos errores, o probamos otra manera?