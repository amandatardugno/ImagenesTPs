## Consideraciones importantes hasta ahora:
### Cómo implementar el ecualizado local

Primero la imagen se dividía en una cuadrícula estática. Se calculaba y aplicaba el histograma a todo un bloque a la vez, lo que generaba "artefactos de bloque" (efecto de tablero de ajedrez), ya que las estadísticas de un bloque no tenían continuidad con el contiguo.

Lo cambiamos a una ventana deslizante: Al usar una ventana deslizante con paso de 1 píxel, reevaluamos la vecindad para cada píxel individualmente. El píxel central adopta el valor de esa ecualización local, logrando transiciones de contraste completamente suaves y continuas a lo largo de toda la imagen.

### El uso del padding:
Para poder procesar los píxeles que están en los extremos de la imagen, la ventana de vecindad necesita "salirse" de los límites originales, o bien no analizar el borde. Elegimos generar con padding un margen artificial. Usar un reflejo de la propia imagen (BORDER_REFLECT) en lugar de rellenar con ceros (negro) evita alterar la distribución estadística del histograma local en los márgenes, previniendo que los bordes de la imagen se oscurezcan de forma antinatural.

### Dimensiones impares de la ventana:

Las dimensiones de la ventana (ej. $15 \times 15$) deben ser estrictamente impares para garantizar la existencia de un centro geométrico en la ventana. Si fuera par (ej. $4 \times 4$), la asignación del valor ecualizado sufriría un desplazamiento espacial hacia arriba a la izqueirda, desalineando sutilmente la imagen resultante respecto a la original. Eso nos obligaría a calcular el ecualizado del píxel de manera asimétrica.

### Influencia del tamaño de la ventana:

* Ventanas pequeñas: Maximizan la sensibilidad a los cambios locales, resaltando detalles topológicos muy finos. Son muy susceptibles a amplificar el ruido de alta frecuencia (si una zona es casi plana, el algoritmo forzará artificialmente un contraste alto sobre el ruido de la imagen).
* Ventanas grandes: Proporcionan una mejor representación del contraste a nivel macro, mitigando el ruido. Contrapartida: Reducen la resolución espacial de la ecualización. Cerca de bordes fuertemente contrastados, los detalles más sutiles que están ocultos se "diluyen" en el cálculo del histograma global de la ventana, perdiendo el beneficio del realce local (genera un efecto de "halo").
* Ventanas asimétricas: Si la ventana no es cuadrada (ej. $3 \times 31$), el comportamiento del filtro se vuelve anisotrópico. Los efectos mencionados arriba se manifiestan en su dimensión correspondiente: en el eje de la dimensión pequeña (3) el realce será ruidoso pero muy reactivo a cambios locales, mientras que en el eje de la dimensión grande (31) el contraste se promediará, suavizando los detalles en esa dirección geométrica.