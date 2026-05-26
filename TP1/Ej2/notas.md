## Consideraciones importantes hasta ahora:
### Segmentación de Celdas
* Se utilizó la función `bordesExamen` para identificar la estructura del examen.
* Se aplicó umbralización y sumas de proyecciones horizontales y verticales para detectar las líneas negras de la grilla.
* Se segmentó el encabezado (donde se encuentra el nombre) y los 10 recuadros de los ejercicios de forma individual para su procesamiento posterior.

### Localización de la Línea de Respuesta
* Inicialmente se intentó obtener la posición de la línea de respuesta contando espacios en blanco entre los renglones de las opciones (sabiendo que siempre hay 4). Sin embargo, se descartó este método debido a que el ancho de las respuestas y el ancho de la línea de puntos era variable y en ocasiones se mezclaba con el texto de la pregunta, perdiendo robustez.
* Se utilizó `connectedComponentsWithStats` sobre el recorte de cada ejercicio. Se identificó la línea de respuesta mediante descriptores de forma: se buscó un componente conexo con una relación de aspecto muy ancha ($W >> H$). Esta línea se utilizó como "ancla" para buscar la respuesta.

### Extracción de la Letra de Respuesta
* Una vez localizada la línea de respuesta, se definió una región de interés (ROI) inmediatamente por encima de ella.
* Se identificaron los componentes conexos en esa región. Se aplicó un filtro de área para eliminar ruidos pequeños o trazos accidentales, conservando únicamente el componente con tamaño compatible con una letra manuscrita.

### Clasificación de la Respuesta (A, B, C, D)
* Se intentó clasificar la letra basándose únicamente en el ancho, alto y área del Bounding Box. Se descartó rápidamente al notar en los *scatter plots* de calibración que las dimensiones de las letras eran extremadamente similares entre sí, lo que generaba un error ante variaciones mínimas de escritura.
* Se implementó una clasificación híbrida (Topológica + Geométrica) mediante `cv2.findContours` con jerarquía (`RETR_TREE`):
    * Criterio Topológico: Se contó la cantidad de contornos hijos (agujeros). Si tiene 2 agujeros se clasifica como **B**. Si tiene 0 agujeros se clasifica como **C**.
    * Criterio Geométrico (Desempate): Si tiene 1 solo agujero (caso de **A** y **D**), se calcula el área de dicho agujero. Tras calibrar con un examen patrón, se determinó que el agujero de la **D** es significativamente más grande que el de la **A**, permitiendo un desempate robusto.
* Limitación: Este criterio asume que la respuesta es válida (A, B, C o D). Si un alumno respondiera con una letra externa (ej. E o F), el programa la clasificaría incorrectamente (como **C** al poseer 0 agujeros para la E o la F).

### Reporte de Resultados
* Se generó un lienzo (canvas) en formato **BGR** para permitir la superposición de colores (Rojo/Verde) y recortes en escala de grises.
* Se concatenó el recorte del campo "Name" con el resultado del examen y la cantidad de aciertos para cada alumno.
* Se definió el umbral de aprobación en 6 aciertos (60%).