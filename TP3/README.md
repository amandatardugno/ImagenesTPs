# TP3 Procesamiento de Imágenes: Reconocimiento y Conteo de Dados

Tercer proyecto de la materia Procesamiento de Imágenes de la Tecnicatura Universitaria en Inteligencia Artificial de la Universidad Nacional de Rosario.

El proyecto implementa un pipeline automatizado de procesamiento de imágenes capaz de procesar videos de tiradas de dados sobre un tapete verde, determinar el frame exacto de reposo absoluto de la tirada, realizar el seguimiento continuo (tracking) de la identidad de cada dado resistiendo colisiones, e informar los valores finales obtenidos.

## Integrantes
* Amanda Tardugno
* Lucas Díaz Celauro

---

## Estructura del Proyecto

```text
TP3/
  data/
    tirada_*.mp4                        # Video de entrada de cada tirada
  outputs/
    output_tirada_*.mp4                 # Video procesado de cada tirada
    mosaico_resultados.png              # Imagen resumen con el recorte de reposo de cada video
  helpers.py                            # Clases de Tracking (Dado, RastreadorDados) y funciones de segmentación
  solution.py                           # Script principal de procesamiento y exportación
  capturas_proceso.py                   # Script de visualización paso a paso del pipeline (para el reporte)
  InformeTP.pdf                         # Informe final del Trabajo Práctico
  README.md                             # Documentación general del proyecto
```

## Uso

Todos los scripts se ejecutan desde la raíz del proyecto de este trabajo (las rutas de los archivos de entrada son relativas).

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
