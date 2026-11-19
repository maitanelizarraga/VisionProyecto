import cv2 as cv
import analisis_modificaciones

def mostrar_imagen(ruta, nombre):
    imagen = cv.imread(ruta)
    cv.imshow(nombre, imagen)
    print(imagen.shape)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return imagen



def main():
    print("Hello from visionproyecto!")


if __name__ == "__main__":
    main()
