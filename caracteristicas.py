import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from segmentacion import segmentar_hoja


def calculateBrightnessContrast(img: np.ndarray) -> tuple:
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return np.mean(img_gray), np.std(img_gray)


def extraer_caracteristicas(img: np.ndarray, mostrar: bool = False):
    img_res, gris, mascara, c = segmentar_hoja(img)

    # --- 1. GEOMETRÍA Y MOMENTOS ---
    M = cv.moments(c)
    cx = (M['m10'] / M['m00']) if M['m00'] != 0 else 0.0
    cy = (M['m01'] / M['m00']) if M['m00'] != 0 else 0.0
    area = max(cv.contourArea(c), 1.0)
    perimetro = max(cv.arcLength(c, closed=True), 1.0)
    compacidad = (4.0 * np.pi * area) / (perimetro ** 2)

    hu = cv.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # --- 2. BRILLO, CONTRASTE E HISTOGRAMAS RGB ---
    img_enmascarada = cv.bitwise_and(img_res, img_res, mask=mascara)
    brillo, contraste = calculateBrightnessContrast(img_enmascarada)

    b, g, r = cv.split(img_res)
    hist_b = cv.calcHist([b], [0], mascara, [16], [0, 256]).flatten() / area
    hist_g = cv.calcHist([g], [0], mascara, [16], [0, 256]).flatten() / area
    hist_r = cv.calcHist([r], [0], mascara, [16], [0, 256]).flatten() / area

    # --- 3. BORDES SOBEL Y CANNY ---
    sx = cv.convertScaleAbs(cv.Sobel(gris, cv.CV_64F, 1, 0, ksize=3))
    sy = cv.convertScaleAbs(cv.Sobel(gris, cv.CV_64F, 0, 1, ksize=3))
    sobel_total = cv.addWeighted(sx, 0.5, sy, 0.5, 0)
    sobel_hoja = cv.bitwise_and(sobel_total, sobel_total, mask=mascara)
    media_sobel = np.sum(sobel_hoja) / area

    canny_medio = cv.Canny(gris, 100, 200)
    canny_hoja = cv.bitwise_and(canny_medio, canny_medio, mask=mascara)
    densidad_canny = np.sum(canny_hoja > 0) / area

    # --- 4. PUNTOS CLAVE ORB ---
    orb = cv.ORB_create()
    puntos_orb, _ = orb.detectAndCompute(gris, mascara)
    num_puntos_orb = float(len(puntos_orb)) / area

    vector = np.concatenate([
        np.array([area, perimetro, compacidad, cx, cy,
                  brillo, contraste, media_sobel, densidad_canny, num_puntos_orb],
                 dtype=np.float32),
        hu_log.astype(np.float32),
        hist_b.astype(np.float32),
        hist_g.astype(np.float32),
        hist_r.astype(np.float32),
    ])

    img_geom = img_res.copy()
    cv.drawContours(img_geom, [c], -1, (0, 255, 0), 2)
    cv.circle(img_geom, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    if mostrar:
        # ---- VISUALIZACIÓN DE CARACTERÍSTICAS ----
        fig = plt.figure(figsize=(16, 10))

        # Fila 1: imagen, máscara, contorno+centroide
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(cv.cvtColor(img_res, cv.COLOR_BGR2RGB)); ax1.set_title("Original"); ax1.axis("off")
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(mascara, cmap="gray"); ax2.set_title("Máscara"); ax2.axis("off")
        ax3 = plt.subplot(2, 4, 3)
        ax3.imshow(cv.cvtColor(img_geom, cv.COLOR_BGR2RGB)); ax3.set_title("Contorno + centroide"); ax3.axis("off")
        ax4 = plt.subplot(2, 4, 4)
        ax4.imshow(sobel_hoja, cmap="gray"); ax4.set_title(f"Sobel (media={media_sobel:.1f})"); ax4.axis("off")

        # Fila 2: Canny, ORB, histograma RGB
        ax5 = plt.subplot(2, 4, 5)
        ax5.imshow(canny_hoja, cmap="gray"); ax5.set_title(f"Canny (dens={densidad_canny:.3f})"); ax5.axis("off")
        ax6 = plt.subplot(2, 4, 6)
        ax6.imshow(cv.drawKeypoints(img_res, puntos_orb, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)[..., ::-1])
        ax6.set_title(f"ORB ({len(puntos_orb)} pts)"); ax6.axis("off")

        ax7 = plt.subplot(2, 4, 7)
        ax7.plot(hist_b, color="blue", label="B")
        ax7.plot(hist_g, color="green", label="G")
        ax7.plot(hist_r, color="red", label="R")
        ax7.set_title("Histograma RGB (16 bins)"); ax7.legend(); ax7.set_xlim([0, 16])

        ax8 = plt.subplot(2, 4, 8)
        ax8.bar(range(len(hu_log)), hu_log, color="purple")
        ax8.set_title("Momentos de Hu (log)"); ax8.set_xlabel("i")

        plt.suptitle(f"Características extraídas — Área={area:.0f}, Perím={perimetro:.0f}, "
                     f"Compacidad={compacidad:.2f}, Brillo={brillo:.1f}, Contraste={contraste:.1f}")
        plt.tight_layout()
        plt.show()

    return vector, img_geom