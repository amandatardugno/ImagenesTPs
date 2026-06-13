import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from helpers import TAMANIO_PATENTE

# Creamos la carpeta si no existe
os.makedirs('templates', exist_ok=True)

caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Cargamos la tipografía con el archivo .ttf de FE-Schrift
# (la letra oficial de las patentes Mercosur). 
# Si no, usa el default.
try:
    font = ImageFont.truetype("templates/FE.TTF", 100)
except:
    font = ImageFont.load_default()
    print("No se encontró la fuente, usando default.")

for char in caracteres:
    img_pil = Image.new('L', (150, 150), color=0)
    draw = ImageDraw.Draw(img_pil)
    
    # Dibujamos el caracter
    draw.text((25, 10), char, font=font, fill=255)
    
    img_cv = np.array(img_pil)
    
    # Buscamos el "Bounding Box" exacto de la letra para quitar los bordes negros
    coordenadas = cv2.findNonZero(img_cv)
    x, y, w, h = cv2.boundingRect(coordenadas)
    recorte = img_cv[y:y+h, x:x+w]
    
    # Redimensionamos al tamaño de la patente
    template_final = cv2.resize(recorte, TAMANIO_PATENTE)
    
    # Guardamos cada caracter
    ruta = f"templates/{char}.png"
    cv2.imwrite(ruta, template_final)
    print(f"Generado {ruta}")