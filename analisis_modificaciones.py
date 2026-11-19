import cv2 as cv


#Generamos una función que nos ayude a modificar las imágenes para comprobar si el modelo ha aprendido o ha memorizado
def modificacion_zonas(imagen):
    alto, ancho = imagen.shape[:2] #obtenemos las características ya que desconocemos el tamaño de cada foto
    
    alto_roi = alto // 10
    ancho_roi = ancho // 10
    
    roi1 = imagen[0:alto_roi, ancho-ancho_roi:ancho]#obtenemos como primer fragmento la esquina superior izquierda

    cv.imshow("ROI 1", roi1)
    cv.waitKey(0)

    roi2 = imagen[alto-alto_roi:alto, 0:ancho_roi] #obtenemos como segundo fragmento la esquina inferior derecha
    
    cv.imshow("ROI 2 ", roi2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return roi1, roi2


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


def modificacion_transformacion(imagen):