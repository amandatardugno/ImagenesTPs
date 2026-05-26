# TP 2 Procesamiento de Imágenes

Segundo proyecto de la materia Procesamiento de Imágenes de la Tecnicatura Universitaria en Inteligencia Artificial de la Universidad Nacional de Rosario.

El trabajo práctico busca implementar el reconocimiento de pastillas en una cinta transportadora en su primer ejercicio, y el reconocimiento automático de patentes argentians en su segundo ejercicio.

# Integrantes
Amanda Tardugno 
Lucas Díaz Celauro 

---

## Estructura del proyecto

```text
TP2/
  data/
    pills.png                           # Imagen de pastillas del primer ejercicio
    img_1.jpg ... img_12.jpg            # Imágenes de patentes del segundo ejercicio
  notas.md                              # Notas de desarrollo del ejercicio
  TUIA_PDI_TP2_2026_C1.pdf              # Enunciado completo del trabajo práctico
  README.md
  requirements.txt                      # Librerías requeridas para este trabajo
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