import os
import glob
import cv2 as cv
import numpy as np


EXTENSIONES = ("*.jpg",)


def cargar_imagen(ruta: str, flags=cv.IMREAD_COLOR) -> np.ndarray:
    imagen = cv.imread(ruta, flags)
    if imagen is None:
        print(f"[ERROR] No se pudo cargar la imagen en la ruta: {ruta}")
    return imagen


# ---------------------------------------------------------------------------
# TRANSFORMACIONES (todas presentes en union.py)
# ---------------------------------------------------------------------------
def exponentialTransformation(img, c, gamma):
    temp = c * np.power(img / 255.0, gamma)
    return np.clip(temp * 255.0, 0, 255).astype(np.uint8)


def generalTransformation(img, alfa, beta):
    return np.clip(alfa * (img.astype(np.float32) - 128) + 128 + beta, 0, 255).astype(np.uint8)


def affineTransformation(img, rot_angle, tras_x=0, tras_y=0):
    rows, cols = img.shape[:2]
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rot_angle, 1)
    M[0, 2] = tras_x
    M[1, 2] = tras_y
    return cv.warpAffine(img, M, (cols, rows))


def filtro_gaussiano(img):
    return cv.GaussianBlur(img, (5, 5), 0)


def filtro_mediana(img):
    return cv.medianBlur(img, 5)


def filtro_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv.filter2D(img, -1, kernel)


def anadir_ruido_gaussiano(img, media=0, sigma=15):
    ruido = np.random.normal(media, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def espejo_horizontal(img):
    return cv.flip(img, 1)


# ---------------------------------------------------------------------------
# AUMENTO DE DATOS EN MEMORIA (solo train)
# ---------------------------------------------------------------------------
def aumentar_datos_train(img: np.ndarray) -> list:
    return [
        img,
        affineTransformation(img, 45.0, 0, 0),
        affineTransformation(img, -45.0, 0, 0),
        affineTransformation(img, 15.0, 20, 0),
        affineTransformation(img, -15.0, -20, 10),
        exponentialTransformation(img, 1.0, 0.5),
        exponentialTransformation(img, 1.0, 2.0),
        exponentialTransformation(img, 1.0, 0.7),
        generalTransformation(img, 1.5, 0),
        generalTransformation(img, 0.7, 30),
        filtro_gaussiano(img),
        filtro_sharpen(img),
        anadir_ruido_gaussiano(img),
        espejo_horizontal(img),
    ]


# ---------------------------------------------------------------------------
# VISUALIZACIÓN DEL AUMENTO
# ---------------------------------------------------------------------------
def mostrar_ejemplos_aumento(ruta_imagen: str, titulo: str = "Ejemplos de aumento"):
    import matplotlib.pyplot as plt

    img = cargar_imagen(ruta_imagen)
    if img is None:
        return

    variaciones = aumentar_datos_train(img)
    etiquetas = [
        "Original",
        "Rot +45°", "Rot -45°",
        "Rot +15° + trasl",
        "Rot -15° + trasl",
        "Gamma 0.5", "Gamma 2.0", "Gamma 0.7",
        "Alfa 1.5", "Alfa 0.7 + beta 30",
        "Gaussiano", "Sharpen",
        "Ruido gaussiano",
        "Espejo horizontal",
    ]

    n = len(variaciones)
    cols = 5
    filas = (n + cols - 1) // cols
    fig, axes = plt.subplots(filas, cols, figsize=(3 * cols, 3 * filas))
    axes = axes.ravel()

    for i, (im, etq) in enumerate(zip(variaciones, etiquetas)):
        axes[i].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        axes[i].set_title(etq, fontsize=9)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{titulo}\n({os.path.basename(ruta_imagen)})")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------------------------
def cargar_dataset(ruta_dataset: str = "dataset", clases_filtro: list = None, usar_subtipos: bool = False):
    registros = []
    if not os.path.isdir(ruta_dataset):
        print(f"[ERROR] No existe el directorio del dataset: {ruta_dataset}")
        return registros

    carpetas = sorted([
        d for d in os.listdir(ruta_dataset)
        if os.path.isdir(os.path.join(ruta_dataset, d))
    ])

    for carpeta in carpetas:
        if clases_filtro is not None and carpeta.lower() not in [c.lower() for c in clases_filtro]:
            continue

        ruta_clase = os.path.join(ruta_dataset, carpeta)

        if not usar_subtipos:
            archivos = []
            for ext in EXTENSIONES:
                archivos.extend(glob.glob(os.path.join(ruta_clase, "**", ext), recursive=True))
            for ruta in sorted(set(archivos)):
                if os.path.isfile(ruta):
                    registros.append((ruta, carpeta))
        else:
            subcarpetas = sorted([
                sd for sd in os.listdir(ruta_clase)
                if os.path.isdir(os.path.join(ruta_clase, sd))
            ])
            for sub in subcarpetas:
                ruta_sub = os.path.join(ruta_clase, sub)
                archivos = []
                for ext in EXTENSIONES:
                    archivos.extend(glob.glob(os.path.join(ruta_sub, "**", ext), recursive=True))
                for ruta in sorted(set(archivos)):
                    if os.path.isfile(ruta):
                        registros.append((ruta, sub))

    return registros