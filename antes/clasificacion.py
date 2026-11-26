import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#hojas muy diferentes 3: aesculus, amelanchier, chamaecyparis
#hojas parecidas 3: amelanchier, betula, salix
#hojas con 2 tipos: chamaecyparis

def mostrar(nombre, imagen):
    cv.imshow(nombre, imagen)
    cv.waitKey(0)
    cv.destroyAllWindows()

def cargar_imagen(ruta, nombre):
    imagen = cv.imread(ruta)
    return imagen


def ejercicio1():
    print("EJERCICIO 1")
    # --------------------------- PROCEDIMIENTO. ENTRENAMIENTO
    img = cv.imread("data/digits.png")
    mostrar("Imagen de entrenamiento", img)

    # PREPARACIÓN
    # 50 filas
    filas = np.vsplit(img, 50)

    #50 columnas
    digitos = []

    for fila in filas:
        columnas = np.hsplit(fila, 50)

        for digito in columnas:
            digitos.append(digito)

    mostrar("Primer digito", digitos[0])

    #vectorizacion
    muestras = []

    for digito in digitos:
        vector = digito.flatten()
        muestras.append(vector)

    muestras = np.array(muestras)

    print("Número de muestras:", len(muestras))
    print("Tamaño de cada vector:", muestras[0].shape)
    print("Dimensiones de la matriz:", muestras.shape)

    # ETIQUETADO
    etiquetas = []

    for i in range(10):
        for j in range(250):
            etiquetas.append(i)

    etiquetas = np.array(etiquetas)

    print("Número de muestras:", len(muestras))
    print("Dimensiones de las muestras:", muestras.shape)
    print("Dimensiones de las etiquetas:", etiquetas.shape)

    # ENTRENAMIENTO KNN
    knn = cv.ml.KNearest_create()

    muestras = np.float32(muestras)
    etiquetas = np.float32(etiquetas).reshape(-1, 1)

    knn.train(muestras, cv.ml.ROW_SAMPLE, etiquetas)

    print("Clasificador KNN entrenado correctamente")


    # --------------------------- PROCEDIMIENTO. CLASIFICACIÓN
    img_test = cv.imread("data/test_digits.png")
    mostrar("Imagen de prueba", img_test)

    # PREPARACIÓN DE PRUEBAS
    #50 filas
    filas_test = np.vsplit(img_test, 50)

    #50 columnas
    filas_test = np.vsplit(img_test, 50)
    digitos_test = []

    for fila in filas_test:
        digitos_test.append(fila)

    print("Número de dígitos de prueba:", len(digitos_test))

    #originales para analizar errores
    digitos_test_originales = digitos_test.copy()

    #vectorizacion
    muestras_test = []

    for digito in digitos_test:
        vector = digito.flatten()
        muestras_test.append(vector)

    muestras_test = np.array(muestras_test)

    print("Número de muestras de prueba:", len(muestras_test))
    print("Dimensiones de las muestras de prueba:", muestras_test.shape)

    muestras_test = np.float32(muestras_test)


    # ETIQUETAS REALES
    etiquetas_reales = []

    for i in range(10):
        for j in range(5):
            etiquetas_reales.append(i)

    etiquetas_reales = np.array(etiquetas_reales)


    # --------------------------- CLASIFICACIÓN
    ret, predicciones, vecinos, distancias = knn.findNearest(
        muestras_test, k=5
    )

    predicciones = predicciones.flatten().astype(int)

    print("Clasificación realizada correctamente")
    print("Etiquetas predichas:", predicciones)

    #eval
    aciertos = np.sum(predicciones == etiquetas_reales)

    accuracy = aciertos / len(etiquetas_reales)

    print("Número de aciertos:", aciertos)
    print("Número de errores:", len(etiquetas_reales) - aciertos)
    print("Accuracy:", accuracy)
    print("Porcentaje de aciertos:", accuracy * 100, "%")

    #etiqueta real y predicción
    plt.figure(figsize=(12, 5))
    for i in range(min(10, len(digitos_test_originales))):
        plt.subplot(2, 5, i + 1)
        digito = digitos_test_originales[i]
        if len(digito.shape) == 3:
            digito = cv.cvtColor(digito, cv.COLOR_BGR2GRAY)
        plt.imshow(digito, cmap="gray")
        real = etiquetas_reales[i]
        predicho = predicciones[i]
        color_titulo = "green" if real == predicho else "red"
        plt.title(f"Real: {real}\nPredicción: {predicho}", color=color_titulo)
        plt.axis("off")
    plt.suptitle("Clasificación de dígitos: verde = acierto, rojo = error")
    plt.tight_layout()
    plt.show()

    # --------------------------- ACTIVIDAD
    # VARIACIÓN DEL NÚMERO DE VECINOS K
    valores_k = [1, 3, 5]
    resultados_k = {}

    print("\nVARIACIÓN DEL NÚMERO DE VECINOS")
    for k in valores_k:

        #clasificar muestras
        _, predicciones_k, _, _ = knn.findNearest(
            muestras_test, k
        )

        predicciones_k = predicciones_k.flatten().astype(int)

        #num aciertos
        aciertos = np.sum(predicciones_k == etiquetas_reales)

        #porcent precisión
        precision = (aciertos / len(etiquetas_reales)) * 100

        resultados_k[k] = precision

        print(f"\nK = {k}")
        print(f"Aciertos: {aciertos}/{len(etiquetas_reales)}")
        print(f"Errores: {len(etiquetas_reales) - aciertos}")
        print(f"Porcentaje de aciertos: {precision:.2f}%")

    print("\nRESUMEN DE RESULTADOS")

    for k, precision in resultados_k.items():
        print(f"K = {k}: {precision:.2f}%")


    # --------------------------- ANÁLISIS DE ERRORES

    # K = 3 analizar los errores
    _, predicciones_errores, _, _ = knn.findNearest(muestras_test, 3)

    predicciones_errores = predicciones_errores.flatten().astype(int)

    #posiciones donde pred incorrecta
    errores = np.where(predicciones_errores != etiquetas_reales)[0]

    print("\nANÁLISIS DE ERRORES")
    print("Número total de errores:", len(errores))

    for i in errores:
        print(f"Muestra {i}: etiqueta real = {etiquetas_reales[i]}, "
              f"predicción = {predicciones_errores[i]}")

    if len(errores) > 0:
        cantidad = min(10, len(errores))
        plt.figure(figsize=(12, 5))
        for posicion, indice_error in enumerate(errores[:cantidad]):
            plt.subplot(2, 5, posicion + 1)
            digito = digitos_test_originales[indice_error]
            if len(digito.shape) == 3:
                digito = cv.cvtColor(digito, cv.COLOR_BGR2GRAY)
            plt.imshow(digito, cmap="gray")
            plt.title(f"Real: {etiquetas_reales[indice_error]}\n"
                      f"Pred: {predicciones_errores[indice_error]}", color="red")
            plt.axis("off")
        plt.suptitle("Ejemplos de dígitos clasificados incorrectamente")
        plt.tight_layout()
        plt.show()


    # --------------------------- REFLEXIÓN

    '''
    REFLEXIÓN
    El modelo KNN clasifica cada uno de los dígitos buscando lo más parecido del entrenamiento y 
    asignandole la clase más frecuente entre sus k vecinos.

    Con k=1, depende solo del mas cercano, siendo más sensible al ruido pero si es demasiado alto,
    puede ser igaul de malo por considerar valores muy lejanos y de otras clases, por lo que el objetivo
    sería encontar el mejor valor medio.

    
    Los errores pueden deberse a que algunos dígitos tienen formas parecidas o están escritos de manera diferente.
    '''

def ejercicio2():
    print("\nEJERCICIO 2")
    # --------------------------- PROCEDIMIENTO. PREPARACIÓN DEL DATASET
    #datos con 2 clases y características.
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )

    #datos train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    #ajustar etiquetas 
    y_train = np.where(y_train == 0, -1, 1).astype(np.int32)
    y_test = np.where(y_test == 0, -1, 1).astype(np.int32)

    X_train = np.float32(X_train)
    X_test = np.float32(X_test)
    y_train = np.int32(y_train).reshape(-1, 1)

    # --------------------------- PROCEDIMIENTO. ENTRENAMIENTO Y EVALUACIÓN

    def entrenar_svm(kernel, C=1.0, gamma=0.5):
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(kernel)

        #param clasificador
        svm.setC(C)
        svm.setGamma(gamma)

        #param kernel polinómico
        svm.setDegree(3)
        svm.setCoef0(0.0)

        svm.setTermCriteria(
            (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,
            1000, 1e-6)
        )

        svm.train(X_train, cv.ml.ROW_SAMPLE, y_train)

        return svm

    def evaluar_y_dibujar(svm, nombre):
        _, predicciones = svm.predict(X_test)
        predicciones = predicciones.flatten().astype(int)
        precision = accuracy_score(y_test, predicciones) * 100

        print(f"{nombre}: {precision:.2f}% de acierto")

        # Crear una cuadrícula para representar la frontera de decisión
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        puntos = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        _, predicciones_malla = svm.predict(puntos)
        Z = predicciones_malla.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        plt.scatter(
            X_train[:, 0], X_train[:, 1],
            c=y_train.flatten(), cmap="coolwarm", edgecolors="black",
            label="Entrenamiento"
        )
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c=y_test, cmap="coolwarm", marker="x", s=50,
            label="Prueba"
        )
        plt.title(f"Frontera de decisión - {nombre}")
        plt.xlabel("Característica x")
        plt.ylabel("Característica y")
        plt.legend()
        plt.show()

        return precision

    #train inicial kernel RBF
    svm = entrenar_svm(cv.ml.SVM_RBF)
    print("Clasificador SVM RBF entrenado correctamente")
    precision = evaluar_y_dibujar(svm, "Kernel RBF")

    # --------------------------- ACTIVIDAD
    # COMPARACIÓN DE KERNELS
    print("\nCOMPARACIÓN DE KERNELS")

    kernels = {
        "LINEAR": cv.ml.SVM_LINEAR,
        "POLY": cv.ml.SVM_POLY,
        "SIGMOID": cv.ml.SVM_SIGMOID
    }
    resultados_kernels = {}

    for nombre, kernel in kernels.items():
        svm_kernel = entrenar_svm(kernel)
        resultados_kernels[nombre] = evaluar_y_dibujar(
            svm_kernel, f"Kernel {nombre}"
        )

    print("\nResumen de kernels:")
    for nombre, precision in resultados_kernels.items():
        print(f"{nombre}: {precision:.2f}%")

    # VARIACIÓN DE LOS HIPERPARÁMETROS C Y GAMMA
    print("\nVARIACIÓN DE C Y GAMMA")

    valores_C = [0.1, 1, 10]
    valores_gamma = [0.1, 0.5, 1]

    for C in valores_C:
        for gamma in valores_gamma:
            svm_parametros = entrenar_svm(cv.ml.SVM_RBF, C, gamma)
            _, predicciones = svm_parametros.predict(X_test)
            predicciones = predicciones.flatten().astype(int)
            precision = accuracy_score(y_test, predicciones) * 100

            print(
                f"C = {C}, gamma = {gamma}: "
                f"{precision:.2f}% de acierto"
            )

    '''
    REFLEXIÓN
    En este ejercicio podemos ver como c controla cuanto se penalizan los errores
    de clasificación, por lo que, cuando tiene un valor alto, intenta clasificar correctamente 
    más muestras y overfitear mientras que un c bajo permite más errores.

    Por otro lado, gamma controla la influencia en el kernes de cada muestra. Cuando este tiene un valor alto,
    las fronteras que genera son más complejas mientras que con valores más pequeños son más suaves.
    '''

def ejercicio3():
    print("\nEJERCICIO 3")
    # --------------------------- PROCEDIMIENTO
    clasificador = cv.CascadeClassifier(r"data\haarcascade_frontalface_default.xml")

    # Imágenes indicadas en el enunciado
    imagenes = ["data/foto1.jpg", "data/foto2.jpg", "data/foto3.jpg"]

    # --------------------------- ACTIVIDAD
    # Variación de scaleFactor y minNeighbors
    configuraciones = [
        (1.01, 3),
        (1.05, 3),
        (1.1, 3),
        (1.1, 5)
    ]

    for ruta in imagenes:
        img = cargar_imagen(ruta, os.path.basename(ruta))

        if img is None:
            continue

        gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for scale_factor, min_neighbors in configuraciones:
            #detet cara
            rostros = clasificador.detectMultiScale(
                gris,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )

            print(
                f"{os.path.basename(ruta)} - scaleFactor={scale_factor}, "
                f"minNeighbors={min_neighbors}: {len(rostros)} rostro(s)"
            )

            #rectangulo
            resultado = img.copy()

            for (x, y, w, h) in rostros:
                cv.rectangle(
                    resultado,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            #num de caras
            cv.putText(
                resultado,
                f"Rostros detectados: {len(rostros)}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            alto_resultado, ancho_resultado = resultado.shape[:2]
            escala = min(1.0, 1100 / ancho_resultado, 750 / alto_resultado)

            if escala < 1.0:
                resultado_mostrado = cv.resize(
                    resultado,
                    (int(ancho_resultado * escala), int(alto_resultado * escala)),
                    interpolation=cv.INTER_AREA
                )
            else:
                resultado_mostrado = resultado

            mostrar("Rostros detectados", resultado_mostrado)

    # --------------------------- REFLEXIÓN
    '''
    REFLEXIÓN
    Cuando el scaleFactor es cercano a 1, se analizan más escalas y se pueden detectar caras de
    distintos tamaños, aunque eso necesitaría de más tiempo.

    Un minNeighbors bajo aumenta la sensibilidad, al igual que los falsos positivos. Un valor alto pide
    más coincidencias y asegura obtener menos detecciones incorrectas aunque puede dejar de ver caras dificiles.
    '''

def ejercicio4():
    print("\nEJERCICIO 4")
    # --------------------------- PROCEDIMIENTO. ENTRENAMIENTO
    ruta_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","archive")
    imagenes_entrenamiento = []
    etiquetas_entrenamiento = []
    imagenes_prueba = []
    etiquetas_prueba = []
    dimensiones_imagen = None

    for sujeto in range(1, 41):
        carpeta = os.path.join(ruta_dataset, f"s{sujeto}")

        if not os.path.isdir(carpeta):
            continue

        archivos = (
            glob.glob(os.path.join(carpeta, "*.pgm")) 
        )
        archivos = sorted(
            archivos,
            key=lambda ruta: int(os.path.splitext(os.path.basename(ruta))[0])
            if os.path.splitext(os.path.basename(ruta))[0].isdigit()
            else os.path.basename(ruta)
        )

        if len(archivos) == 0:
            continue

        #7 imágenes para train
        for indice, ruta_imagen in enumerate(archivos):
            datos = np.fromfile(ruta_imagen, dtype=np.uint8)
            imagen = cv.imdecode(datos, cv.IMREAD_GRAYSCALE)
            if imagen is None:
                continue

            if dimensiones_imagen is None:
                dimensiones_imagen = imagen.shape

            vector = imagen.flatten()

            if indice < 7:
                imagenes_entrenamiento.append(vector)
                etiquetas_entrenamiento.append(sujeto)
            else:
                imagenes_prueba.append(vector)
                etiquetas_prueba.append(sujeto)


    X_train = np.array(imagenes_entrenamiento, dtype=np.float32)
    y_train = np.array(etiquetas_entrenamiento)
    X_test = np.array(imagenes_prueba, dtype=np.float32)
    y_test = np.array(etiquetas_prueba)

    print("Imágenes de entrenamiento:", X_train.shape)
    print("Imágenes de prueba:", X_test.shape)

    # --------------------------- IMPLEMENTACIÓN DE EIGENFACES (PCA)
    #num de componentes al num de muestras
    valores_componentes = [10, 20, 50, 100]
    resultados_pca = {}

    numero_componentes = min(50, X_train.shape[0] - 1, X_train.shape[1])

    pca_visualizacion = PCA(n_components=numero_componentes, whiten=True)
    pca_visualizacion.fit(X_train)

    alto, ancho = dimensiones_imagen

    plt.figure(figsize=(10, 5))
    for i in range(min(5, numero_componentes)):
        eigenface = pca_visualizacion.components_[i].reshape(alto, ancho)
        plt.subplot(1, 5, i + 1)
        plt.imshow(eigenface, cmap="gray")
        plt.title(f"EigenFace {i + 1}")
        plt.axis("off")
    plt.suptitle("Primeras EigenFaces")
    plt.show()

    # --------------------------- PROCEDIMIENTO. RECONOCIMIENTO Y VISUALIZACIÓN

    # Entrenar KNN con img en espacio PCA
    for componentes in valores_componentes:
        if componentes > min(X_train.shape[0] - 1, X_train.shape[1]):
            continue

        pca = PCA(n_components=componentes, whiten=True)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train_pca, y_train)

        predicciones = knn.predict(X_test_pca)
        precision = accuracy_score(y_test, predicciones) * 100
        resultados_pca[componentes] = precision

        print(
            f"Componentes PCA = {componentes}: "
            f"{precision:.2f}% de acierto"
        )

    # --------------------------- ACTIVIDAD

    print("\nVARIACIÓN DEL NÚMERO DE COMPONENTES")
    for componentes, precision in resultados_pca.items():
        print(f"{componentes} componentes: {precision:.2f}%")

    #ult modelo valido para ejemplo
    if len(resultados_pca) > 0:
        mejor_numero_componentes = max(
            resultados_pca, key=resultados_pca.get
        )

        pca = PCA(n_components=mejor_numero_componentes, whiten=True)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train_pca, y_train)

        #pred imágenes de prueba y mostrar su imagen más cercana
        numero_ejemplos = min(5, len(X_test))
        predicciones_ejemplo = knn.predict(X_test_pca[:numero_ejemplos])

        plt.figure(figsize=(15, 6))
        for indice in range(numero_ejemplos):
            prediccion = predicciones_ejemplo[indice]

            #buscar la imagen de train más cercana en el espacio PCA
            distancias = np.linalg.norm(
                X_train_pca - X_test_pca[indice], axis=1
            )
            indice_cercano = np.argmin(distancias)

            imagen_prueba = X_test[indice].reshape(alto, ancho)
            imagen_cercana = X_train[indice_cercano].reshape(alto, ancho)
            color_titulo = "green" if y_test[indice] == prediccion else "red"

            plt.subplot(2, numero_ejemplos, indice + 1)
            plt.imshow(imagen_prueba, cmap="gray")
            plt.title(f"Prueba\nReal: {y_test[indice]}")
            plt.axis("off")

            plt.subplot(2, numero_ejemplos, numero_ejemplos + indice + 1)
            plt.imshow(imagen_cercana, cmap="gray")
            plt.title(f"Predicción: {prediccion}", color=color_titulo)
            plt.axis("off")

            print(
                f"Imagen de prueba {indice}: persona real = {y_test[indice]}, "
                f"persona reconocida = {prediccion}"
            )

        plt.suptitle("Reconocimiento facial: fila superior = prueba, "
                     "fila inferior = imagen de entrenamiento más cercana")
        plt.tight_layout()
        plt.show()

    # --------------------------- REFLEXIÓN
    '''
    REFLEXIÓN
    Al usar PCA, las dimensiones se reducen y se mantienen las direcciones de mayor variabilidad,
    aunque con pocos componentes se pierde mucha información, con demasiados el ruido se puede llegar
    a mantener y, así, el coste.

    FisherFaces usa LDA, buscando las direcciones que separan mejor las clases y maximizando así la separación
    y minimizando la variabilidad. 

    PCA no utiliza las etiquetas al calcular sus componentes, mientras que LDA sí las utiliza.
    '''
    
def ejercicio5():
    print("\nEJERCICIO 5")
    # --------------------------- PROCEDIMIENTO. SEGMENTACIÓN POR COLOR
    img = cargar_imagen("data/oceano.jpg", "oceano.jpg")

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imagen_float = np.float32(img_rgb)

    alto, ancho, canales = imagen_float.shape

    #vectorizacion
    muestras = imagen_float.reshape((-1, 3))

    print("Dimensiones de la imagen:", imagen_float.shape)
    print("Dimensiones de las muestras:", muestras.shape)

    # --------------------------- ACTIVIDAD
    # VARIACIÓN DEL NÚMERO DE CLUSTERS K
    valores_k = [2, 4, 8]

    for K in valores_k:
        print(f"\nSegmentación con K = {K}")

        #parada de k-Means
        criterio = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

        intentos = 3
        flags = cv.KMEANS_PP_CENTERS

        compactacion, etiquetas, centros = cv.kmeans(
            muestras,
            K,
            None,
            criterio,
            intentos,
            flags
        )

        #centros a enteros para colores
        centros = np.uint8(centros)

        #píxel por el centroide
        imagen_segmentada = centros[etiquetas.flatten()]
        imagen_segmentada = imagen_segmentada.reshape((alto, ancho, canales))

        print("Compactación:", compactacion)
        print("Centroides de color:", centros)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Imagen original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(imagen_segmentada)
        plt.title(f"Imagen segmentada - K = {K}")
        plt.axis("off")

        plt.show()

    # --------------------------- REFLEXIÓN

    '''
    REFLEXIÓN
    Si utilizamos un número bajo de clusters, en la imagen aparecen menos colores y detalles.
    Si aumentamos ese número, hay más precisión en los colores y detalles, aunque la segmentación es menos
    simplificada. En BagOfWords, los centroides se pueden representar como palabras, formando el vocab de la
    imagen y cada descriptor se asignaría al centroide más cercano

    '''


