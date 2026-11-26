import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  

def ejercicio1():
    print("ejercicio 1")
    ##---------------------------OBJETIVOS
    # leemos la imagen
    imagen = cv.imread("practica03_data\practica03_data\wirebond-mask.tif")
    cv.imshow("Original", imagen)
    
    # filtro laplaciano
    laplacian = cv.Laplacian(imagen, cv.CV_64F) # CV_64F es para detectar gradientes negativos
    laplacian = cv.convertScaleAbs(laplacian) #para el plot
    cv.imshow("laplaciano", laplacian)
    cv.waitKey(0)
    
    # filtro sobel
    sobel_x = cv.Sobel(imagen, cv.CV_64F, 1, 0, ksize=3)
    sobel_x = cv.convertScaleAbs(sobel_x)
    cv.imshow("sobelX", sobel_x)
    cv.waitKey(0)
    
    #grad Y
    sobel_y = cv.Sobel(imagen, cv.CV_64F, 0, 1, ksize=3)
    sobel_y = cv.convertScaleAbs(sobel_y)
    cv.imshow("sobelY", sobel_y)
    cv.waitKey(0)
    
    #comb ejes para mapa de gradiente total
    sobel_total = cv.addWeighted(sobel_x, 0.8, sobel_y, 0.8, 0) # los 0.8 es cuanto de peso le damos a ese filtro y el 0 es el brillo añadido de cada pixel 
    cv.imshow("sobelTot", sobel_total)
    cv.waitKey(0)

    ##---------------------------ACTIVIDAD
    def obtener_sobel_total(img, ksize):
        sobel_x = cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize))
        sobel_y = cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize))
        return cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    sobel_k3 = obtener_sobel_total(imagen, ksize=3)
    cv.imshow("sobel_k3", sobel_k3)
    cv.waitKey(0)
    sobel_k5 = obtener_sobel_total(imagen, ksize=5)
    cv.imshow("sobel_k5", sobel_k5)
    cv.waitKey(0)
    sobel_k7 = obtener_sobel_total(imagen, ksize=7)
    cv.imshow("sobel_k7", sobel_k7)
    cv.waitKey(0)

    canny_bajo = cv.Canny(imagen, 30, 90)     #umbrales bajos (muy sensible)
    cv.imshow("canny_bajo", canny_bajo)
    cv.waitKey(0)
    canny_medio = cv.Canny(imagen, 100, 200)  #umbrales medios (equilibrado)
    cv.imshow("canny_medio", canny_medio)
    cv.waitKey(0)
    canny_alto = cv.Canny(imagen, 180, 240)   #umbrales altos (solo bordes fuertes)
    cv.imshow("canny_alto", canny_alto)
    cv.waitKey(0)

    imagenes = [
        (imagen, "Original"),
        (laplacian, "Laplacian"),
        (sobel_k3, "Sobel (ksize=3)"),
        (sobel_k5, "Sobel (ksize=5)"),
        (sobel_k7, "Sobel (ksize=7)"),
        (canny_bajo, "Canny (30, 90) - Sensible"),
        (canny_medio, "Canny (100, 200) - Medio"),
        (canny_alto, "Canny (180, 240) - Restrictivo"),
    ]
    
    plt.figure(figsize=(16, 8))
    for i, (img, titulo) in enumerate(imagenes, 1):
        plt.subplot(2, 4, i)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
        plt.title(titulo, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    '''
    LAPLACIAN:
    Detecta perfectamente todos los bordes y cambios de intensidad en todas las direcciones de la imagen
    de forma simétrica y sin calcular componentes direccionales. Sin embargo, si la imagen tuviera mucho
    ruido, no nos funcionaría igual de bien.

    SOBEL:
    La parte positiva de este filtro, es la posibilidad de regular el filtro de ruido y el grosor del borde
    con el tamaño del kernel, por lo que es mucho más útil para imagenes que necesitan procesarse en tiempo real
    o momentos que se necesite una respuesta rápida, es decir, cuando no haga falta una precisión de pixel.

    CANNY:
    Este filtro es el más preciso de los tres, ya que solo genera contornos de un solo pixel, lo que permite
    filtrar bordes que sean muy suaves o ruido, y gracias a que permite ajustar los umbrales. Este es el idóneo
    para situaciones como tareas de visión artificial exigentes, controles de calidad, o situaciones de mucha exigencia.
    '''



def ejercicio2():
    print("ejercicio 2")
    #---------------------------PROCEDIMIENTO
    imagen = cv.imread(r"practica03_data\practica03_data\checkerboard-clean.tif") #esta puesta la r pq si no detecta comandos como salto de linea,...
    cv.imshow("imagen", imagen)
    cv.waitKey(0)

    # border binario
    gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    edges = cv.Canny(gray, 50, 150)
    
    cv.imshow("bor. canny", edges)
    cv.waitKey(0)

    #detecta lineas infinitas y las cuenta
    def hough(img_original, img_edges, umbral):
        lines = cv.HoughLines(img_edges, rho=1, theta=np.pi/180, threshold=umbral)
        img_out = img_original.copy()
        
        num_lineas = 0
        if lines is not None:
            num_lineas = len(lines)
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                #linea recta infinita en el lienzo
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img_out, num_lineas

    #proc base
    hough_std, total_lineas_std = hough(imagen, edges, umbral=100)
    cv.imshow("hough standar", hough_std)
    cv.waitKey(0)

    ##---------------------------ACTIVIDAD
    hough_u50, count_50 = hough(imagen, edges, umbral=50)
    hough_u100, count_100 = hough(imagen, edges, umbral=100)
    hough_u150, count_150 = hough(imagen, edges, umbral=150)

    lines_p = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10) #el rho es resolución en distancia del acumulador pixeles, el theta resolución angular del acumulador radianes, threshold min. de votos para linea valida, el minlenght descarta seg. cortos y el linegap para distancia max para unir puntos
    hough_p = imagen.copy()
    count_p = 0
    if lines_p is not None:
        count_p = len(lines_p)
        for line in lines_p:
            x1, y1, x2, y2 = line.ravel() #lineravel hace q si o si se obtenga esos valores sin importar el tamaño del array
            cv.line(hough_p, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow("hough prob", hough_p)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #visualizacion y comparación
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(edges, cmap='gray')
    plt.title('Bordes canny')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv.cvtColor(hough_u50, cv.COLOR_BGR2RGB))
    plt.title(f'hough stan (umbral=50)\nLíneas: {count_50}')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv.cvtColor(hough_u100, cv.COLOR_BGR2RGB))
    plt.title(f'hough stan (umbral=100)\nLíneas: {count_100}')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv.cvtColor(hough_u150, cv.COLOR_BGR2RGB))
    plt.title(f'hough stan (umbral=150)\nLíneas: {count_150}')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv.cvtColor(hough_p, cv.COLOR_BGR2RGB))
    plt.title(f'hough prob. : {count_p}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    #imagen propia
    imagen = cv.imread("practica03_data/practica03_data/laberinto.jpg") #esta puesta la r pq si no detecta comandos como salto de linea,...
    cv.imshow("imagen", imagen)
    cv.waitKey(0)

    # border binario
    gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    edges = cv.Canny(gray, 50, 150)
    
    cv.imshow("bor. canny", edges)
    cv.waitKey(0)

    #detecta lineas infinitas y las cuenta
    def hough(img_original, img_edges, umbral):
        lines = cv.HoughLines(img_edges, rho=1, theta=np.pi/180, threshold=umbral)
        img_out = img_original.copy()
        
        num_lineas = 0
        if lines is not None:
            num_lineas = len(lines)
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                #linea recta infinita en el lienzo
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img_out, num_lineas

    #proc base
    hough_std, total_lineas_std = hough(imagen, edges, umbral=100)
    cv.imshow("hough standar", hough_std)
    cv.waitKey(0)

    ##---------------------------ACTIVIDAD
    hough_u50, count_50 = hough(imagen, edges, umbral=50)
    hough_u100, count_100 = hough(imagen, edges, umbral=100)
    hough_u150, count_150 = hough(imagen, edges, umbral=150)

    lines_p = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=5, minLineLength=30, maxLineGap=10) #el rho es resolución en distancia del acumulador pixeles, el theta resolución angular del acumulador radianes, threshold min. de votos para linea valida, el minlenght descarta seg. cortos y el linegap para distancia max para unir puntos
    hough_p = imagen.copy()
    count_p = 0
    if lines_p is not None:
        count_p = len(lines_p)
        for line in lines_p:
            x1, y1, x2, y2 = line.ravel() #lineravel hace q si o si se obtenga esos valores sin importar el tamaño del array
            cv.line(hough_p, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow("hough prob", hough_p)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #visualizacion y comparación
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(edges, cmap='gray')
    plt.title('Bordes canny')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv.cvtColor(hough_u50, cv.COLOR_BGR2RGB))
    plt.title(f'hough stan (umbral=50)\nLíneas: {count_50}')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv.cvtColor(hough_u100, cv.COLOR_BGR2RGB))
    plt.title(f'hough stan (umbral=100)\nLíneas: {count_100}')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv.cvtColor(hough_u150, cv.COLOR_BGR2RGB))
    plt.title(f'hough stan (umbral=150)\nLíneas: {count_150}')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv.cvtColor(hough_p, cv.COLOR_BGR2RGB))
    plt.title(f'hough prob. : {count_p}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

def ejercicio1():
    print("EJERCICIO 1")
    ##---------------------------PROCEDIMIENTO
    #leemos la imagen
    imagen = cv.imread("twotone-circles.tiF")
    cv.imshow("IMAGEN", imagen)
    cv.waitKey(0)

    #grises
    gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    cv.imshow('GRIS', gris)
    cv.waitKey(0)

    #umbr binaria
    _, umbral = cv.threshold(gris, 127, 255, cv.THRESH_BINARY)
    cv.imshow('UMBRAL', umbral)
    cv.waitKey(0)

    #detect contornos
    contornos, jerarquia = cv.findContours(umbral, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    imgcontornos = imagen.copy()
    cv.drawContours(imgcontornos, contornos, -1, (0, 255, 0), 2)
    cv.imshow('CONTORNOS', imgcontornos)
    cv.waitKey(0)

    ##---------------------------ACTIVIDAD
    # VARIACIÓN DE PARÁMETROS
    #cont exteriores
    contornos_ext, _ = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #cont exteriores e interiores
    contornos_tree, _ = cv.findContours(umbral, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print("Recuperación de contornos")
    print(f"RETR_EXTERNAL: {len(contornos_ext)} contornos (externas).")
    print(f"RETR_TREE: {len(contornos_tree)} contornos (con huecos).\n")

    img_ext = imagen.copy()
    cv.drawContours(img_ext, contornos_ext, -1, (0, 0, 255), 2)  #rojo para externos
    cv.imshow('ext', img_ext)
    cv.waitKey(0)

    img_tree = imagen.copy()
    cv.drawContours(img_tree, contornos_tree, -1, (0, 255, 0), 2) #verde para jerarquía
    cv.imshow('int', img_tree)
    cv.waitKey(0)
    
    #puntos borde
    contornos_none, _ = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contornos_simple, _ = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    print("Aprox de puntos")
    if len(contornos_none) > 0 and len(contornos_simple) > 0:
        pts_none = len(contornos_none[0])
        pts_simple = len(contornos_simple[0])
        print(f"CHAIN_APPROX_NONE:   {pts_none} puntos almacenados.")
        print(f"CHAIN_APPROX_SIMPLE: {pts_simple} puntos almacenados.")
        reduccion = ((pts_none - pts_simple) / pts_none) * 100
        print(f"Reducción de puntos:  {reduccion:.1f}% menos información de memoria.\n")

    # ANÁLISIS GEOMÉTRICO
    img_geom = imagen.copy()  

    for i, c in enumerate(contornos):
        #momentos y centroide
        M = cv.moments(c)
        
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        #area y perímetro
        area = cv.contourArea(c)
        perimetro = cv.arcLength(c, closed=True)

        print(f"Contorno #{i + 1}:")
        print(f"  - Centroide (Cx, Cy): ({cx}, {cy})")
        print(f"  - Área:               {area:.2f} ")
        print(f"  - Perímetro:          {perimetro:.2f} \n")

        #contorno verde
        cv.drawContours(img_geom, [c], -1, (0, 255, 0), 2)

        #centr rojo y etiqueta
        cv.circle(img_geom, (cx, cy), 5, (0, 0, 255), -1)
        cv.putText(img_geom, f"C{i+1} ({cx},{cy})", (cx + 10, cy - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv.imshow('geometrico', img_geom)
    cv.waitKey(0)
    cv.destroyAllWindows()


def ejercicio2():
    print("EJERCICIO 2")
    ##---------------------------PROCEDIMIENTO
    #lee imagen
    imagen = cv.imread("woman-and-baby.tif")

    #def roi
    alto, ancho = imagen.shape[:2]
    rect = (int(ancho * 0.05), int(alto * 0.05), int(ancho * 0.9), int(alto * 0.9))
    img_roi = imagen.copy()
    x, y, w, h = rect
    cv.rectangle(img_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("roi", img_roi)
    cv.waitKey(0)

    #grabCut
    #masc salida e init modelos de mezcla de gaussianas (GMM)
    mask = np.zeros(imagen.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    #init por rectángulo
    iteraciones = 5
    cv.grabCut(imagen, mask, rect, bgdModel, fgdModel, iteraciones, cv.GC_INIT_WITH_RECT)

    #extracción objeto
    # 0 = Fondo seguro (GC_BGD), 1 = Primer plano seguro (GC_FGD)
    # 2 = Fondo probable (GC_PR_BGD), 3 = Primer plano probable (GC_PR_FGD)
    #mantenemos 1 y 3 (primer plano)
    mask2 = np.where((mask == 1) | (mask == 3), 1, 0).astype('uint8')

    #multi la imagen original
    objeto_segmentado = imagen * mask2[:, :, np.newaxis]
    cv.imshow("masc binaria GrabCut", mask2 * 255)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow("obj segmentado", objeto_segmentado)
    cv.waitKey(0)
    cv.destroyAllWindows()


    ##---------------------------ACTIVIDAD
    alto, ancho = imagen.shape[:2]
    #def otros rectangulos
    rectangulos = {
        "ajust": (int(ancho * 0.15), int(alto * 0.10), int(ancho * 0.70), int(alto * 0.85)),
        "muy Amplio": (int(ancho * 0.01), int(alto * 0.01), int(ancho * 0.98), int(alto * 0.98)),
        "desplazado": (int(ancho * 0.35), int(alto * 0.20), int(ancho * 0.50), int(alto * 0.60))
    }

    for nombre, rect in rectangulos.items():
        # Inicializar máscara y modelos GMM
        mask = np.zeros(imagen.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Aplicar GrabCut con la configuración del rectángulo actual
        cv.grabCut(imagen, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

        # Máscara final (1 = Primer plano seguro, 3 = Primer plano probable)
        mask2 = np.where((mask == 1) | (mask == 3), 1, 0).astype('uint8')
        objeto_segmentado = imagen * mask2[:, :, np.newaxis]

        # Dibujar la caja delimitadora en una copia de la imagen
        img_bbox = imagen.copy()
        x, y, w, h = rect
        cv.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar ROI y Resultado
        cv.imshow(f"ROI ({nombre})", img_bbox)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow(f"Segmentacion ({nombre})", objeto_segmentado)
        cv.waitKey(0)
        cv.destroyAllWindows()

    #img propia
    #lee imagen
    imagen = cv.imread("img.jpg")

    #def roi
    alto, ancho = imagen.shape[:2]
    rect = (int(ancho * 0.05), int(alto * 0.05), int(ancho * 0.9), int(alto * 0.9))
    img_roi = imagen.copy()
    x, y, w, h = rect
    cv.rectangle(img_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("roi", img_roi)
    cv.waitKey(0)

    #grabCut
    #masc salida e init modelos de mezcla de gaussianas (GMM)
    mask = np.zeros(imagen.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    #init por rectángulo
    iteraciones = 5
    cv.grabCut(imagen, mask, rect, bgdModel, fgdModel, iteraciones, cv.GC_INIT_WITH_RECT)

    #extracción objeto
    # 0 = Fondo seguro (GC_BGD), 1 = Primer plano seguro (GC_FGD)
    # 2 = Fondo probable (GC_PR_BGD), 3 = Primer plano probable (GC_PR_FGD)
    #mantenemos 1 y 3 (primer plano)
    mask2 = np.where((mask == 1) | (mask == 3), 1, 0).astype('uint8')

    #multi la imagen original
    objeto_segmentado = imagen * mask2[:, :, np.newaxis]
    cv.imshow("masc binaria GrabCut", mask2 * 255)
    cv.imshow("obj segmentado", objeto_segmentado)
    cv.waitKey(0)
    cv.destroyAllWindows()

    '''
    VENTAJAS DE LA INTERACTIVIDAD DE GRABCUT
    Uno de los principales beneficios que nos aporta el uso de esta herramienta
    es el ahorro de tiempo que genera debido a la sustitución de tener que trazar 
    manualmente pixel a pixel por un simple rectángulo definido inicialmente.

    Además, nos permite corregir fallos con trazos manuales que recalibran los 
    modelos de color.

    Por otro lado, nos ayuda a definir correctamente los bordes de una imagen entre
    el objeto y el respectivo fondo, siendo más flexible que una umbralización binaria tradicional.
    '''

def ejercicio3():
    print("EJERCICIO 3")
    ##---------------------------PROCEDIMIENTO
    #leemos la imagen
    imagen = cv.imread("balls-with-reflections.tif")
    cv.imshow("IMAGEN", imagen)
    cv.waitKey(0)

    gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    #umbr binaria Otsu
    _, umbral = cv.threshold(gris, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #elem estructurante op morfológicas
    kernel = np.ones((3, 3), np.uint8)

    #def fondo seguro
    fondo_seguro = cv.dilate(umbral, kernel, iterations=3)

    #def 1 plano seg
    primer_plano_seguro = cv.erode(umbral, kernel, iterations=3)

    #reg desconocidas
    desconocido = cv.subtract(fondo_seguro, primer_plano_seguro)

    #watershed
    _, marcadores = cv.connectedComponents(primer_plano_seguro)

    #sum 1 a todas las etiquetas para fondo 1 y no 0
    marcadores = marcadores + 1

    #0 a desconocido para watershed
    marcadores[desconocido == 255] = 0

    #watershed imagen original
    marcadores = cv.watershed(imagen, marcadores)

    #limit del algoritmo -1
    resultado = imagen.copy()
    resultado[marcadores == -1] = [0, 0, 255]  #fronteras rojo


    cv.imshow("Otsu", umbral)
    cv.waitKey(0)
    cv.imshow("fondo seguro (Dilatacion)", fondo_seguro)
    cv.waitKey(0)
    cv.imshow("1 plano seguro (Erosion)", primer_plano_seguro)
    cv.waitKey(0)
    cv.imshow("reg desconocidas", desconocido)
    cv.waitKey(0)
    cv.imshow("watershed", resultado)
    cv.waitKey(0)

    
    tamanios_kernel = [1, 3, 7]

    for k in tamanios_kernel:
        #elem estructurante
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))

        #fond seg
        fondo_seguro = cv.dilate(umbral, kernel, iterations=2)

        #1 plano seg
        primer_plano_seguro = cv.erode(umbral, kernel, iterations=2)

        #reg desconocidas
        desconocido = cv.subtract(fondo_seguro, primer_plano_seguro)

        #matriz marcadores componentes conexas
        num_marcadores, marcadores = cv.connectedComponents(primer_plano_seguro)
        
        #fondo = 1, desconocido = 0
        marcadores = marcadores + 1
        marcadores[desconocido == 255] = 0

        #watershed
        resultado = imagen.copy()
        marcadores = cv.watershed(resultado, marcadores)
        resultado[marcadores == -1] = [0, 0, 255]  # Dibujar bordes detectados en rojo

        print(f"Kernel {k}")
        print(f"Objetos en primer plano: {num_marcadores - 1}\n")

        cv.imshow(f"1 plano seguro (Kernel {k})", primer_plano_seguro)
        cv.imshow(f"reg desconocidas (Kernel {k})", desconocido)
        cv.imshow(f"watershed (Kernel {k})", resultado)
        cv.waitKey(0)

    cv.destroyAllWindows()

    '''
    Con el kernel pequeño vemos que no se separan objetos que se estan tocando, 
    generando así pocos núcleos que unen muchos objetos.

    El kernel optimo, el de 3x3, se generan núcleos bien definidos para cada objeto, 
    quitando la unión entre las esferas sin eliminarlas.

    Con el más grande(7x7) se llegan a borrar los objetos más pequeños y los límites 
    se distorsionan, no llegando a generar un buen resultado.

    Hemos visto entonces que con watershed se pueden separar elementos adyacentes con mejores resultados
    que empleando el resto de técnicas vistas en los ejercicios. Es por ello que el findContours funciona 
    correctamente en objetos aislados con buen contraste, pero, para objetos más solapados, esta herramienta 
    tiene mejores resultados.
    '''


import cv2 as cv
import numpy as np


def mostrar(nombre, imagen):
    cv.imshow(nombre, imagen)
    cv.waitKey(0)
    cv.destroyAllWindows()


def ejercicio1():
    print("EJERCICIO 1")
    # --------------------------- PROCEDIMIENTO
    imagen = cv.imread("glass-caster-RGB.tif")

    #escala grises
    gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    mostrar("original", imagen)
    mostrar("gris", gris)

    #detector SIFT
    sift = cv.SIFT_create()

    #puntos clave
    puntos = sift.detect(gris, None)
    imagen_puntos = cv.drawKeypoints(imagen, puntos, None, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    mostrar("puntos clave SIFT", imagen_puntos)

    #puntos clave + descriptores
    puntos, descrip = sift.detectAndCompute(gris, None)
    imagen_detallada = cv.drawKeypoints(imagen,puntos,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    mostrar("puntos SIFT", imagen_detallada)

    print(f"num puntos clave SIFT: {len(puntos)}")
    print(f"tamaño descrip: {descrip.shape}")
    print(f"tamaño memoria: {descrip.nbytes}")

    # --------------------------- ACTIVIDAD
    #SIFT con otra imagen
    imagen1 = cv.imread("SIFT_img1.tif")

    gris1 = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY)
    puntos1, descrip1 = sift.detectAndCompute(gris1, None)

    imagen_1 = cv.drawKeypoints(imagen1, puntos1, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    mostrar("puntos SIFT ref", imagen_1)
    print(f"puntos clave: {len(puntos1)}")

    #emparejamiento de descriptor FLANN
    descrip = np.asarray(descrip, dtype=np.float32)
    descrip1 = np.asarray(descrip1, dtype=np.float32)

    index = dict(algorithm=1, trees=5)  # KD-tree
    busq = dict(checks=50)
    flann = cv.FlannBasedMatcher(index, busq)

    coinc = flann.knnMatch(descrip1, descrip, k=2)
    buenas = []
    for pareja in coinc:
        if len(pareja) == 2:
            m, n = pareja
            if m.distance < 0.7 * n.distance:
                buenas.append(m)

    print(f"coincidencias buenas FLANN: {len(buenas)}")
    imagen_coinc = cv.drawMatches(imagen1, puntos1, imagen, puntos, buenas, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    mostrar("emparejamiento SIFT con FLANN", imagen_coinc)

def ejercicio2():
    print("EJERCICIO 2:")
    # --------------------------- PROCEDIMIENTO

    imagen = cv.imread("glass-caster-RGB.tif")
    gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    # ORB puntos + descrip
    orb = cv.ORB_create()
    puntos_orb, descrip_orb = orb.detectAndCompute(gris, None)
    imagen_orb = cv.drawKeypoints(imagen, puntos_orb, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    mostrar("puntos ORB", imagen_orb)

    # FAST puntos pero no descrip
    fast = cv.FastFeatureDetector_create()
    puntos_fast = fast.detect(gris, None)
    imagen_fast = cv.drawKeypoints(imagen, puntos_fast, None)
    mostrar("puntos FAST", imagen_fast)

    # BRIEF en puntos FAST.
    descrip_brief = None
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    puntos_brief, descrip_brief = brief.compute(gris, puntos_fast)
    imagen_brief = cv.drawKeypoints(imagen, puntos_brief, None)
    mostrar("puntos FAST + BRIEF", imagen_brief)

    # --------------------------- ACTIVIDAD
    print("COMPARACIÓN:")
    print(f"SIFT: {len(cv.SIFT_create().detect(gris, None))} puntos clave")
    print(f"ORB:  {len(puntos_orb)} puntos clave")
    print(f"FAST: {len(puntos_fast)} puntos clave")

    print(f"descrip ORB: {descrip_orb.shape}")
    print(f"tamaño de los descrip ORB: {descrip_orb.nbytes} bytes")

    print(f"descrip BRIEF: {descrip_brief.shape}")
    print(f"tamaño de los descrip BRIEF: {descrip_brief.nbytes} bytes")

    print("FAST detecta puntos, pero no crea descrip por sí solo.")
    print("ORB combina detección y descripción de forma eficiente.")
    print("SIFT suele ofrecer descrip más robustos, con mayor coste computacional.")

    '''
    SIFT y ORB son más completos que FAST y BRIEF, los cuales son más sencillos y ligeros e ideales 
    para aplicaiones con recursos limitados o cuando hay necesidad de rapidez.

    '''

def ejercicio3():
    print("EJERCICIO 3:")
    # --------------------------- PROCEDIMIENTO

    imagen = cv.imread("circuitboard.tif")
    plantilla = cv.imread("template.tif")


    gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    gris_plantilla = cv.cvtColor(plantilla, cv.COLOR_BGR2GRAY)

    alto, ancho = gris_plantilla.shape[:2]
    alto_imagen, ancho_imagen = gris.shape[:2]

    mostrar("Imagen principal", imagen)
    mostrar("Plantilla", plantilla)

    # --------------------------- ACTIVIDAD

    alto, ancho = gris_plantilla.shape[:2]

    # TM_CCOEFF_NORMED
    resultado = cv.matchTemplate(gris, gris_plantilla, cv.TM_CCOEFF_NORMED)

    mejor_valor = cv.minMaxLoc(resultado)[1]
    umbral = 0.8
    ys, xs = np.where(resultado >= umbral)

    print("TM_CCOEFF_NORMED")
    print(f"mejor valor: {mejor_valor:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc. encontradas: {len(xs)}")

    # Dibujamos los rectángulos
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(img_resultado, (x, y), (x + ancho, y + alto),
                     (0, 255, 0), 2)

    cv.imshow("TM_CCOEFF_NORMED", img_resultado)
    cv.waitKey(0)

    # TM_CCOEFF_NORMED otro umbral
    resultado = cv.matchTemplate(gris, gris_plantilla, cv.TM_CCOEFF_NORMED)

    mejor_valor = cv.minMaxLoc(resultado)[1]
    umbral = 0.5
    ys, xs = np.where(resultado >= umbral)

    print("TM_CCOEFF_NORMED")
    print(f"mejor valor: {mejor_valor:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc. encontradas: {len(xs)}")

    # Dibujamos los rectángulos
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(img_resultado, (x, y), (x + ancho, y + alto),
                     (0, 255, 0), 2)

    cv.imshow("TM_CCOEFF_NORMED", img_resultado)
    cv.waitKey(0)


    # TM_CCORR_NORMED
    resultado = cv.matchTemplate(gris, gris_plantilla, cv.TM_CCORR_NORMED)

    mejor_valor = cv.minMaxLoc(resultado)[1]
    umbral = 0.95
    ys, xs = np.where(resultado >= umbral)

    print("TM_CCORR_NORMED")
    print(f"mejor valor: {mejor_valor:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc. encontradas: {len(xs)}")

    # Dibujamos los rectángulos
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(img_resultado, (x, y), (x + ancho, y + alto),
                     (0, 255, 0), 2)

    cv.imshow("TM_CCORR_NORMED", img_resultado)
    cv.waitKey(0)

    # TM_CCORR_NORMED otro umbral
    resultado = cv.matchTemplate(gris, gris_plantilla, cv.TM_CCORR_NORMED)

    mejor_valor = cv.minMaxLoc(resultado)[1]
    umbral = 1
    ys, xs = np.where(resultado >= umbral)

    print("TM_CCORR_NORMED")
    print(f"mejor valor: {mejor_valor:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc. encontradas: {len(xs)}")

    # Dibujamos los rectángulos
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(img_resultado, (x, y), (x + ancho, y + alto),
                     (0, 255, 0), 2)

    cv.imshow("TM_CCORR_NORMED", img_resultado)
    cv.waitKey(0)


    # TM_SQDIFF_NORMED
    resultado = cv.matchTemplate(gris, gris_plantilla, cv.TM_SQDIFF_NORMED)

    mejor_valor = cv.minMaxLoc(resultado)[0]
    umbral = 0.1
    ys, xs = np.where(resultado <= umbral)

    print("TM_SQDIFF_NORMED")
    print(f"mejor valor: {mejor_valor:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc. encontradas: {len(xs)}")

    # Dibujamos los rectángulos
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(img_resultado, (x, y), (x + ancho, y + alto),
                     (0, 255, 0), 2)

    cv.imshow("TM_SQDIFF_NORMED", img_resultado)
    cv.waitKey(0)

    cv.destroyAllWindows()

    # TM_SQDIFF_NORMED otro umbral
    resultado = cv.matchTemplate(gris, gris_plantilla, cv.TM_SQDIFF_NORMED)

    mejor_valor = cv.minMaxLoc(resultado)[0]
    umbral = 0.05
    ys, xs = np.where(resultado <= umbral)

    print("TM_SQDIFF_NORMED")
    print(f"mejor valor: {mejor_valor:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc. encontradas: {len(xs)}")

    # Dibujamos los rectángulos
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(img_resultado, (x, y), (x + ancho, y + alto),
                     (0, 255, 0), 2)

    cv.imshow("TM_SQDIFF_NORMED", img_resultado)
    cv.waitKey(0)

    cv.destroyAllWindows()

    
    #plantilla escalada al 30 %
    plantilla_escalada = cv.resize(
        gris_plantilla,
        None,
        fx=0.3,
        fy=0.3
    )

    mostrar("Plantilla escalada", plantilla_escalada)

    resultado_escalado = cv.matchTemplate(
        gris,
        plantilla_escalada,
        cv.TM_CCOEFF_NORMED
    )

    mejor_valor_escalado = cv.minMaxLoc(resultado_escalado)[1]
    umbral = 0.8

    ys, xs = np.where(resultado_escalado >= umbral)

    print("PLANTILLA ESCALADA")
    print(f"mejor coincidencia: {mejor_valor_escalado:.4f}")
    print(f"umbral utilizado: {umbral}")
    print(f"coinc: {len(xs)}")

    alto_escalado, ancho_escalado = plantilla_escalada.shape[:2]
    img_resultado = imagen.copy()

    for x, y in zip(xs, ys):
        cv.rectangle(
            img_resultado,
            (x, y),
            (x + ancho_escalado, y + alto_escalado),
            (0, 255, 0),
            2
        )

    mostrar("coinc con plantilla escalada", img_resultado)

    '''
    SIFT y ORB funcionan mejor con transformaciones geométricas ya que no comparan
    todos los pixeles directamente si no que detectan los puntos clave y describen la zona de alrededor.
    '''
    
