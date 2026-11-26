import cv2 as cv

'''
Generamos funciones que nos ayude a modificar las imágenes para comprobar si el modelo ha 
aprendido o ha memorizado ya que, de cada tipo no encontramos muchas imágenes, por lo que si 
las modificamos, consegumos más material con el que trabajar
'''
#Con esta función solo obtenemos pequeñas regiones de cada imagen
def zonas(imagen):
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

#Con esta, giramos cada imagen, primero 45º y después -45º
def giro(imagen):
    trans1=fp.affineTransformation(imagen, 45.0,0, 0)
    cv.imshow("TRANS 1", trans1)
    cv.waitKey(0)

    trans2=fp.affineTransformation(imagen, -45.0,0, 0)
    cv.imshow("TRANS 2", trans1)
    cv.waitKey(0)

#Aplicamos diferentes filtros también para cambiar las imágenes
def filtros(imagen):
    media = cv.blur(imagen, (5, 5)) #los 5 son el tamaño de la matriz de filtro
    gauss = cv.GaussianBlur(imagen, (5, 5), 0) #el 0 es la desviación estandar
    mediana = cv.medianBlur(imagen, 5)
    kernel_sharpen = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpen = cv.filter2D(imagen, -1, kernel_sharpen)

    cv.imshow("Original", imagen)
    cv.waitKey(0)
    cv.imshow("Media", media)
    cv.waitKey(0)
    cv.imshow("Gaussiano", gauss)
    cv.waitKey(0)
    cv.imshow("Mediana", mediana)
    cv.waitKey(0)
    cv.imshow("Sharpen", sharpen)
    cv.waitKey(0)

    return media, gauss, mediana, sharpen

#cambiamos la intensidad, el brillo y el contraste
def exponencial(imagen):
     #γ = 0.5, c = 1.0
    imagen_exp1 = fp.exponentialTransformation(imagen, 1.0, 0.5)
    cv.imshow("Exponencial1", imagen_exp1)
    cv.waitKey(0)

    #γ = 2.0, c = 1.0
    imagen_exp2 = fp.exponentialTransformation(imagen, 1.0, 2.0) # + zonas oscuras
    cv.imshow("Exponencial2", imagen_exp2)
    cv.waitKey(0)

    #γ = 2.0, c = 2.0
    imagen_exp3 = fp.exponentialTransformation(imagen, 2.0, 2.0) # + intensidad y brillo
    cv.imshow("Exponencial3", imagen_exp3)
    cv.waitKey(0)

    print("lineal:") 
    #α = 1, β = 50
    imagen_lin1 = fp.generalTransformation(imagen, 1, 50)
    cv.imshow("Lineal1", imagen_lin1)
    cv.waitKey(0)

    #α = 1.5, β = 50
    imagen_lin2 = fp.generalTransformation(imagen, 1.5, 50) # + contraste
    cv.imshow("Lineal2", imagen_lin2)
    cv.waitKey(0)

    #α = 1.5, β = 0
    imagen_lin3 = fp.generalTransformation(imagen, 1.5, 0) # - brillo general
    cv.imshow("Lineal3", imagen_lin3)
    cv.waitKey(0)
