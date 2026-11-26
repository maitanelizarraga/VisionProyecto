import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def segmentar_hoja(img: np.ndarray, tamano_base: tuple = (256, 256), mostrar: bool = False):
    img_res = cv.resize(img, tamano_base, interpolation=cv.INTER_AREA)
    gauss = cv.GaussianBlur(img_res, (5, 5), 0)
    gris = cv.cvtColor(gauss, cv.COLOR_BGR2GRAY)

    _, umbral = cv.threshold(gris, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    esquinas = [umbral[0, 0], umbral[0, -1], umbral[-1, 0], umbral[-1, -1]]
    if np.mean(esquinas) > 127:
        umbral = 255 - umbral

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    umbral_morf = cv.dilate(umbral, kernel, iterations=2)
    umbral_morf = cv.erode(umbral_morf, kernel, iterations=2)

    contornos, _ = cv.findContours(umbral_morf, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contornos) == 0:
        mascara = np.ones(gris.shape, dtype=np.uint8) * 255
        cnt = np.array([[[0, 0]], [[gris.shape[1]-1, 0]],
                        [[gris.shape[1]-1, gris.shape[0]-1]], [[0, gris.shape[0]-1]]])
    else:
        cnt = max(contornos, key=cv.contourArea)
        mascara = np.zeros(gris.shape, dtype=np.uint8)
        cv.drawContours(mascara, [cnt], -1, 255, -1)

    if mostrar:
        # ---- VISUALIZACIÓN DEL PIPELINE DE SEGMENTACIÓN ----
        etapas = [
            (cv.cvtColor(img_res, cv.COLOR_BGR2RGB), "1. Original (resized)"),
            (cv.cvtColor(gauss, cv.COLOR_BGR2RGB), "2. GaussianBlur 5x5"),
            (gris, "3. Escala de grises"),
            (umbral, "4. Otsu (posible inversión)"),
            (umbral_morf, "5. Dilate + Erode (3x3)"),
            (mascara, "6. Máscara final"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (im, tit) in zip(axes.ravel(), etapas):
            ax.imshow(im, cmap="gray" if len(im.shape) == 2 else None)
            ax.set_title(tit)
            ax.axis("off")
        plt.suptitle("Pipeline de segmentación de hoja")
        plt.tight_layout()
        plt.show()

    return img_res, gris, mascara, cnt