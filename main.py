import cv2 as cv
import matplotlib.pyplot as plt

from clasificadores import clasificador_todos


def main():
    print("=== PROYECTO: CLASIFICACIÓN DE HOJAS ===")
    print("Ejecutando los 3 clasificadores en orden...\n")
    clasificador_todos()
    cv.destroyAllWindows()
    plt.close("all")
    print("\n=== FIN ===")


if __name__ == "__main__":
    main()