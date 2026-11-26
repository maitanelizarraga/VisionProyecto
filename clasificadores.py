import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset_loader import (
    cargar_dataset, cargar_imagen, aumentar_datos_train, mostrar_ejemplos_aumento
)
from caracteristicas import extraer_caracteristicas


# Acumulador global para la tabla resumen final
RESUMEN_GLOBAL = []


def normalizar_zscore(X_train, X_test):
    media = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - media) / std, (X_test - media) / std


def evaluar_knn(X_train, y_train, X_test, y_test):
    mejor_acc, mejor_k, mejores_preds, accs = -1.0, 1, None, {}
    for k in [1, 3, 5]:
        if k > len(X_train):
            continue
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        accs[f"KNN k={k}"] = acc
        if acc > mejor_acc:
            mejor_acc, mejor_k, mejores_preds = acc, k, preds
    return f"KNN (k={mejor_k})", mejor_acc, mejores_preds, accs


def evaluar_pca_knn(X_train, y_train, X_test, y_test):
    mejor_acc, mejor_comp, mejores_preds, accs = -1.0, None, None, {}
    for comp in [5, 10, 20]:
        if comp > min(X_train.shape[0] - 1, X_train.shape[1]):
            continue
        pca = PCA(n_components=comp, whiten=True)
        knn = KNeighborsClassifier(n_neighbors=1).fit(pca.fit_transform(X_train), y_train)
        preds = knn.predict(pca.transform(X_test))
        acc = accuracy_score(y_test, preds) * 100
        accs[f"PCA({comp})+KNN"] = acc
        if acc > mejor_acc:
            mejor_acc, mejor_comp, mejores_preds = acc, comp, preds
    return f"PCA ({mejor_comp}) + KNN", mejor_acc, mejores_preds, accs


def evaluar_svm(X_train, y_train, X_test, y_test):
    mejor_acc, mejor_kernel, mejores_preds, accs = -1.0, "", None, {}
    for nombre, kernel in {"LINEAR": "linear", "RBF": "rbf", "POLY": "poly"}.items():
        svm = SVC(kernel=kernel, C=1.0, gamma=0.05, degree=3).fit(X_train, y_train)
        preds = svm.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        accs[f"SVM {nombre}"] = acc
        if acc > mejor_acc:
            mejor_acc, mejor_kernel, mejores_preds = acc, nombre, preds
    return f"SVM ({mejor_kernel})", mejor_acc, mejores_preds, accs


def _entrenar_y_evaluar(rutas_train, y_train_raw, X_test, y_test, usar_augmentation):
    """Entrena los 3 modelos y devuelve (mejor_modelo, mejor_acc, preds, dict_accs)."""
    X_train, y_train = [], []
    for ruta, etq in zip(rutas_train, y_train_raw):
        img = cargar_imagen(ruta)
        if img is None:
            continue
        variaciones = aumentar_datos_train(img) if usar_augmentation else [img]
        for im in variaciones:
            vec, _ = extraer_caracteristicas(im)
            X_train.append(vec)
            y_train.append(etq)

    X_train = np.float32(np.array(X_train))
    y_train = np.array(y_train, dtype=np.int32)

    X_train, X_test_n = normalizar_zscore(X_train, X_test.copy())
    X_train = np.float32(X_train)
    X_test_n = np.float32(X_test_n)

    r_knn = evaluar_knn(X_train, y_train, X_test_n, y_test)
    r_pca = evaluar_pca_knn(X_train, y_train, X_test_n, y_test)
    r_svm = evaluar_svm(X_train, y_train, X_test_n, y_test)

    resultados = [r_knn, r_pca, r_svm]
    mejor_modelo, mejor_acc, mejores_preds = max(
        [(m, a, p) for m, a, p, _ in resultados], key=lambda r: r[1]
    )
    accs_flat = {}
    for _, _, _, accs in resultados:
        accs_flat.update(accs)
    return mejor_modelo, mejor_acc, mejores_preds, accs_flat, len(X_train)


def mostrar_tabla_comparativa(nombre_exp, sin_aug, con_aug):
    """sin_aug y con_aug son dicts modelo -> precisión."""
    modelos = list(con_aug.keys())
    print("\n" + "=" * 75)
    print(f" TABLA COMPARATIVA: SIN vs CON AUMENTO — {nombre_exp}")
    print("=" * 75)
    print(f"{'Modelo':<18} {'Sin aumento':>14} {'Con aumento':>14} {'Mejora':>10}")
    print("-" * 75)
    for m in modelos:
        s = sin_aug.get(m, 0.0)
        c = con_aug.get(m, 0.0)
        mejora = c - s
        signo = "+" if mejora >= 0 else ""
        print(f"{m:<18} {s:>13.2f}% {c:>13.2f}% {signo}{mejora:>8.2f}%")
    print("-" * 75)
    media_s = np.mean(list(sin_aug.values()))
    media_c = np.mean(list(con_aug.values()))
    print(f"{'MEDIA':<18} {media_s:>13.2f}% {media_c:>13.2f}% {media_c - media_s:>+9.2f}%")
    print("=" * 75)

    # Visualización matplotlib de la tabla
    fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(modelos)))
    ax.axis("off")
    filas = [[m, f"{sin_aug.get(m, 0):.2f}%", f"{con_aug.get(m, 0):.2f}%",
              f"{con_aug.get(m, 0) - sin_aug.get(m, 0):+.2f}%"] for m in modelos]
    filas.append(["MEDIA", f"{media_s:.2f}%", f"{media_c:.2f}%", f"{media_c - media_s:+.2f}%"])
    tabla = ax.table(
        cellText=filas,
        colLabels=["Modelo", "Sin aumento", "Con aumento", "Mejora"],
        loc="center", cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)
    plt.title(f"Tabla comparativa: sin vs con aumento\n{nombre_exp}", pad=20)
    plt.tight_layout()
    plt.show()


def ejecutar_experimento(nombre, clases_filtro=None, usar_subtipos=False):
    print("\n" + "=" * 75)
    print(f" {nombre}")
    print("=" * 75)

    registros = cargar_dataset("dataset", clases_filtro, usar_subtipos)
    if len(registros) == 0:
        print("[AVISO] No se encontraron imágenes.")
        return

    clases = sorted(list(set(etq for _, etq in registros)))
    mapa = {n: i for i, n in enumerate(clases)}
    conteo = {c: sum(1 for _, e in registros if e == c) for c in clases}
    for c in clases:
        print(f"  - {c}: {conteo[c]} imágenes")

    if len(clases) < 2:
        print("[ERROR] Se necesitan al menos 2 clases/subtipos.")
        return

    rutas = [r for r, _ in registros]
    etiquetas = np.array([mapa[e] for _, e in registros], dtype=np.int32)

    ejemplos_vistos = set()
    for ruta, etq in registros:
        if etq in ejemplos_vistos:
            continue
        ejemplos_vistos.add(etq)
        mostrar_ejemplos_aumento(ruta, titulo=f"Aumento — clase '{etq}' — {nombre}")
        if len(ejemplos_vistos) >= 2:
            break

    rutas_train, rutas_test, y_train_raw, y_test_raw = train_test_split(
        rutas, etiquetas, test_size=0.30, random_state=42, stratify=etiquetas
    )

    # ---- X_test una sola vez (no se aumenta nunca) ----
    X_test, y_test, imgs_vis = [], [], []
    for ruta, etq in zip(rutas_test, y_test_raw):
        img = cargar_imagen(ruta)
        if img is None:
            continue
        vec, img_geom = extraer_caracteristicas(img)
        X_test.append(vec)
        y_test.append(etq)
        imgs_vis.append(img_geom)
    X_test = np.float32(np.array(X_test))
    y_test = np.array(y_test, dtype=np.int32)

    # ---- 1) SIN AUMENTO ----
    print("\n--- Evaluación SIN aumento de datos ---")
    mejor_m_s, mejor_acc_s, preds_s, accs_s, n_train_s = _entrenar_y_evaluar(
        rutas_train, y_train_raw, X_test, y_test, usar_augmentation=False
    )
    print(f"  Mejor: {mejor_m_s} -> {mejor_acc_s:.2f}% (train={n_train_s})")

    # ---- 2) CON AUMENTO ----
    print("\n--- Evaluación CON aumento de datos (14×) ---")
    mejor_m_c, mejor_acc_c, preds_c, accs_c, n_train_c = _entrenar_y_evaluar(
        rutas_train, y_train_raw, X_test, y_test, usar_augmentation=True
    )
    print(f"  Mejor: {mejor_m_c} -> {mejor_acc_c:.2f}% (train={n_train_c})")

    # ---- 3) Tabla comparativa + figura ----
    mostrar_tabla_comparativa(nombre, accs_s, accs_c)

    # ---- 4) Guardar en resumen global ----
    RESUMEN_GLOBAL.append({
        "nombre": nombre,
        "sin_aug": accs_s,
        "con_aug": accs_c,
        "mejor_s": (mejor_m_s, mejor_acc_s),
        "mejor_c": (mejor_m_c, mejor_acc_c),
        "n_train_s": n_train_s,
        "n_train_c": n_train_c,
        "n_test": len(y_test),
    })

    # A partir de aquí usamos el modelo CON aumento para las visualizaciones
    mejor_modelo, mejor_acc, mejores_preds = mejor_m_c, mejor_acc_c, preds_c
    errores = np.where(mejores_preds != y_test)[0]
    print(f"\n>>> MEJOR MODELO (con aumento): {mejor_modelo} con {mejor_acc:.2f}% <<<")
    print(f"Aciertos: {len(y_test) - len(errores)}/{len(y_test)} | Errores: {len(errores)}")

    # ==================== VIS: distribución de clases ====================
    plt.figure(figsize=(10, 4))
    plt.bar(list(conteo.keys()), list(conteo.values()), color="steelblue")
    plt.title(f"Distribución del dataset — {nombre}")
    plt.ylabel("Nº imágenes")
    plt.xticks(rotation=25)
    for i, (k, v) in enumerate(conteo.items()):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    # ==================== VIS: comparativa de todos los modelos (con aumento) ====================
    nombres_m = list(accs_c.keys())
    valores = list(accs_c.values())
    colores = ["green" if v == max(valores) else "steelblue" for v in valores]
    plt.figure(figsize=(11, 4))
    plt.barh(nombres_m, valores, color=colores)
    plt.axvline(mejor_acc, color="red", linestyle="--", label=f"Mejor = {mejor_acc:.1f}%")
    plt.xlabel("Precisión (%)")
    plt.title(f"Comparativa de modelos (con aumento) — {nombre}")
    plt.legend()
    for i, v in enumerate(valores):
        plt.text(v + 0.5, i, f"{v:.1f}%", va="center")
    plt.tight_layout()
    plt.show()

    # ==================== VIS: matriz de confusión ====================
    cm = confusion_matrix(y_test, mejores_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Matriz de Confusión (con aumento)\n{nombre}\n({mejor_modelo}: {mejor_acc:.2f}%)")
    plt.colorbar()
    ticks = np.arange(len(clases))
    plt.xticks(ticks, clases, rotation=25)
    plt.yticks(ticks, clases)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("Clase Real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.show()

    # ==================== VIS: recall por clase ====================
    total_por_clase = cm.sum(axis=1)
    recall = (np.diag(cm) / np.maximum(total_por_clase, 1)) * 100
    plt.figure(figsize=(9, 4))
    plt.bar(clases, recall, color="seagreen")
    plt.axhline(mejor_acc, color="red", linestyle="--", label=f"Media = {mejor_acc:.1f}%")
    plt.ylabel("Recall por clase (%)")
    plt.title(f"Aciertos por clase — {nombre}")
    plt.xticks(rotation=25)
    plt.ylim([0, 105])
    for i, v in enumerate(recall):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==================== VIS: ejemplos acierto/error ====================
    cantidad = min(10, len(imgs_vis))
    plt.figure(figsize=(14, 5))
    for i in range(cantidad):
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv.cvtColor(imgs_vis[i], cv.COLOR_BGR2RGB))
        real = clases[y_test[i]]
        pred = clases[mejores_preds[i]]
        color = "green" if real == pred else "red"
        plt.title(f"Real: {real}\nPred: {pred}", color=color, fontsize=9)
        plt.axis("off")
    plt.suptitle(f"{nombre}: verde = acierto, rojo = error")
    plt.tight_layout()
    plt.show()

    # ==================== VIS: galería solo de errores ====================
    if len(errores) > 0:
        n_err = min(10, len(errores))
        plt.figure(figsize=(14, 5))
        for i, idx in enumerate(errores[:n_err]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(cv.cvtColor(imgs_vis[idx], cv.COLOR_BGR2RGB))
            real = clases[y_test[idx]]
            pred = clases[mejores_preds[idx]]
            plt.title(f"Real: {real}\nPred: {pred}", color="red", fontsize=9)
            plt.axis("off")
        plt.suptitle(f"{nombre}: {len(errores)} errores (mostrando {n_err})")
        plt.tight_layout()
        plt.show()


def clasificador_1():
    ejecutar_experimento("CLASIFICADOR 1: Hojas muy diferentes",
                         ["aesculus", "amelanchier", "chamaecyparis"], False)


def clasificador_2():
    ejecutar_experimento("CLASIFICADOR 2: Hojas parecidas",
                         ["amelanchier", "betula", "salix"], False)


def clasificador_3():
    ejecutar_experimento("CLASIFICADOR 3: Subtipos de chamaecyparis",
                         ["chamaecyparis"], True)


def resumen_final():
    """Tabla-resumen global con los 3 experimentos."""
    if not RESUMEN_GLOBAL:
        print("[AVISO] No hay experimentos ejecutados.")
        return

    print("\n" + "=" * 90)
    print(" RESUMEN GLOBAL: MEJOR MODELO CON vs SIN AUMENTO")
    print("=" * 90)
    print(f"{'Experimento':<45} {'Sin aug.':>10} {'Con aug.':>10} {'Mejora':>10}")
    print("-" * 90)
    for r in RESUMEN_GLOBAL:
        m_s, a_s = r["mejor_s"]
        m_c, a_c = r["mejor_c"]
        print(f"{r['nombre'][:44]:<45} {a_s:>9.2f}% {a_c:>9.2f}% {a_c - a_s:>+9.2f}%")
        print(f"   -> Sin: {m_s} | Con: {m_c} | train sin={r['n_train_s']} con={r['n_train_c']} | test={r['n_test']}")
    print("=" * 90)

    # Figura-resumen
    nombres = [r["nombre"].split(":")[0].strip() for r in RESUMEN_GLOBAL]
    sin_vals = [r["mejor_s"][1] for r in RESUMEN_GLOBAL]
    con_vals = [r["mejor_c"][1] for r in RESUMEN_GLOBAL]
    x = np.arange(len(nombres))
    ancho = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - ancho / 2, sin_vals, ancho, label="Sin aumento", color="salmon")
    plt.bar(x + ancho / 2, con_vals, ancho, label="Con aumento", color="seagreen")
    for i, (s, c) in enumerate(zip(sin_vals, con_vals)):
        plt.text(i - ancho / 2, s + 0.5, f"{s:.1f}%", ha="center", fontsize=9)
        plt.text(i + ancho / 2, c + 0.5, f"{c:.1f}%", ha="center", fontsize=9)
    plt.xticks(x, nombres)
    plt.ylabel("Precisión (%)")
    plt.title("Resumen: mejor modelo sin vs con aumento")
    plt.ylim([0, 110])
    plt.legend()
    plt.tight_layout()
    plt.show()


def clasificador_todos():
    clasificador_1()
    clasificador_2()
    clasificador_3()
    resumen_final()