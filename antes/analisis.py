import cv2 as cv




def histgrama(imagen):
        #calculamos el brillo y contraste con la función
        brightness, contrast = fp.calculateBrightnessContrast(imagen)
        print(f"Brillo: {brightness:.2f} y contraste :{contrast:.2f} ")

        #comprobamos con la versión manual
        brightness, contrast = fp.calculateBrightnessContrastManual(imagen)
        print(f"Brillo: {brightness:.2f} y contraste :{contrast:.2f} ")

        #separamos los canales de color
        b,g,r = cv.split(imagen)
        print(f"Azul: {b}, Verde: {g} y Rojo: {r}")

        #historiogramas de intensidad
        #calcular vector de 256 posibles valores (num de intensidad) analizando todos los pixeles
        hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv.calcHist([r], [0], None, [256], [0, 256])

        #representa el histograma
        plt.figure("Histogramas RGB")

        plt.plot(hist_b, color='blue', label='Azul')
        plt.plot(hist_g, color='green', label='Verde')
        plt.plot(hist_r, color='red', label='Rojo')

        #características del histograma (límites en ejes, las leyendas y título)
        plt.xlim([0, 256])
        plt.xlabel("Intensidad")
        plt.ylabel("Número de píxeles")
        plt.title("Histogramas de los canales RGB")
        plt.legend()

        plt.show()    

