# TP 1 Procesamiento de Imágenes

Primer proyecto de la materia Procesamiento de Imágenes de la Tecnicatura Universitaria en Inteligencia Artificial de la Universidad Nacional de Rosario.

El trabajo práctico busca implementar un ecualizado local del histograma en su primer ejercicio, y una corrección automática de exámenes en su segundo ejercicio.

# Integrantes
Amanda Tardugno 
Lucas Díaz Celauro 
Adriano Marzol Perini

---

## Estructura del proyecto

```text
TP1/
  Ej1/
    tp1Ej1.py                           # Script principal: ecualización local de histograma
    Imagen_con_detalles_escondidos.tif  # Imagen de entrada del ejercicio
    notas.md                            # Notas de desarrollo del ejercicio
  Ej2/
    tp1Ej2.py                           # Script principal: corrección automática de los 5 exámenes
    helpers.py                          # Funciones reutilizables del ejercicio 2
    calibrar_letras.py                  # Script auxiliar de calibración (agujeros vs área por letra)
    capturas_proceso.py                 # Script de visualización paso a paso del pipeline (para el reporte)
    examen_1.png ... examen_5.png       # 5 exámenes resueltos de ejemplo
    notas.md                            # Notas de desarrollo del ejercicio
  TUIA_PDI_TP1_2026_C1.pdf              # Enunciado completo del trabajo práctico
  README.md
  requirements.txt                      # Librerías requeridas
```

## Uso

Todos los scripts se ejecutan desde la raíz del proyecto (las rutas de los archivos de entrada son relativas).

### Ejercicio 1 — Ecualización local de histograma

```bash
python Ej1/tp1Ej1.py
```

Carga `Imagen_con_detalles_escondidos.tif` y muestra tres versiones lado a lado: la imagen original, la ecualizada globalmente y la ecualizada localmente (ventana deslizante de 21×21).

### Ejercicio 2 — Corrección automática de exámenes

**Script principal:**

```bash
python Ej2/tp1Ej2.py
```

Recorre los 5 exámenes (`examen_1.png` ... `examen_5.png`) y, por cada uno:

- valida los campos del encabezado (`Name`, `Date`, `Class`) e imprime OK/MAL,
- identifica la letra marcada en cada una de las 10 preguntas y la compara con las respuestas correctas,
- cuenta los aciertos y decide aprobado (≥6) o desaprobado.

Al final muestra un reporte visual con los nombres recortados de cada alumno, marcados en verde (aprobado) o rojo (desaprobado).

**Script de visualización paso a paso (para armar el reporte del TP):**

```bash
python Ej2/capturas_proceso.py
```

Abre una sucesión de figuras de matplotlib que muestran lo que hace cada función del pipeline sobre `examen_4.png`: detección de bordes, detección de las líneas del encabezado, validación de campos, conteo de palabras con los gaps resaltados, recorte del Name, detección de la letra marcada, análisis de agujeros y clasificación de las 10 preguntas.

**Script auxiliar de calibración:**

```bash
python Ej2/calibrar_letras.py
```

Usa `examen_4.png` (que tiene las cuatro letras A/B/C/D representadas) para graficar un scatter plot de *cantidad de agujeros* vs *área del agujero más grande* por letra. Sirve para decidir los umbrales usados en `identificarRespuestas`.


## Instalación del proyecto y creación del entorno virtual

Antes de instalar las dependencias, es necesario crear y activar un entorno virtual.  
A continuación se detalla el procedimiento tanto para Linux como para Windows.

### Crear entorno virtual e instalar dependencias en Linux (Ubuntu / Debian)

Instalar `venv`:

```bash
sudo apt install python3-venv -y
python3 -m venv .venv
```

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
````

Activar el entorno virtual:

```bash
source .venv/bin/activate
```

Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

---

### Crear entorno virtual e instalar dependencias en Windows

Desde la carpeta del proyecto, crear el entorno virtual:

```bash
python -m venv .venv
```

o, según la instalación:

```bash
py -3 -m venv .venv
```

Activar el entorno virtual:

Usando PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

O usando CMD:

```cmd
.venv\Scripts\activate.bat
```

Instalar dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
```