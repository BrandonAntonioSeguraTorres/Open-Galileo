#Open Galileo – Apilamiento avanzado de imágenes científicas y macro
#Open Galileo es un programa gratuito y de código abierto desarrollado
#por Brandon Antonio Segura Torres @micro.cosmonauta, diseñado para el
#apilamiento de imágenes (focus stacking) y la integración avanzada de
#señal, orientado a fotografía científica, microscopia, macrofotografía,
#astrofotografía y documentación técnica de alta precisión.
#El software permite combinar múltiples imágenes de una misma escena para
#obtener máxima nitidez, profundidad de campo extendida y reducción de ruido,
#incluso cuando cada imagen individual tiene zonas fuera de foco o muy baja señal.

#Open Galileo es y siempre será gratuito.
#Si este programa te resulta útil y deseas apoyar el tiempo, el esfuerzo y la
#investigación detrás de su desarrollo, puedes hacer una donación voluntaria.
#Tu aporte ayuda a mantener y mejorar este proyecto abierto.
#Gracias por apoyar la ciencia accesible.
#Donaciónes en Argentina (alias): opengalileo
#Resto del mundo por Paypal: antoniovangritte@gmail.com

#Open Galileo is and always will be free.
#If you find this program useful and wish to support the time, effort, and
#research behind its development, you can make a voluntary donation.
#Your contribution helps maintain and improve this open project.
#Thank you for supporting accessible science.
#Donations in Argentina (alias): opengalileo
#Rest of the world via PayPal: antoniovangritte@gmail.com
#Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
#Copyright (c) 2026 Brandon Antonio Segura Torres
#This software is licensed under the Creative Commons
#Attribution-NonCommercial 4.0 International License.
#You are free to:
#- Share — copy and redistribute the material in any medium or format
#- Adapt — remix, transform, and build upon the material
#Under the following terms:
#- Attribution — You must give appropriate credit to the original author
# (Brandon Antonio Segura Torres), provide a link to this license, and
# indicate if changes were made.
#- NonCommercial — You may not use the material for commercial purposes
# without explicit written permission from the author.
#For commercial use, licensing, or integration into paid products or services,
#you must contact the author beforehand.
#This software is provided "as is", without warranty of any kind.

import os
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageSequence, ImageEnhance

import cv2
import numpy as np


class ProcessCancelled(Exception):
    """Señal para cancelar tareas largas desde la interfaz."""
    pass

# RAW (NEF/CR2/ARW/DNG/...) via rawpy (opcional)
try:
    import rawpy
except Exception:
    rawpy = None

# Wavelets denoising via PyWavelets (opcional)
try:
    import pywt
except Exception:
    pywt = None

# BM3D denoising (opcional) -> pip install bm3d
try:
    import bm3d
except Exception:
    bm3d = None

# -------------------------
# Utiles
# -------------------------

RAW_EXTS = (".nef", ".raw", ".dng", ".arw", ".cr2", ".cr3", ".rw2", ".orf", ".raf", ".pef", ".srw")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp") + RAW_EXTS

def _resize_keep_aspect(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    s = max_side / float(max(h, w))
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out, s

def imread_unicode(path):
    """
    Lectura de imágenes que soporta rutas con tildes/ñ/etc.
    - Imágenes comunes: lee buffer y decodifica con cv2.imdecode (Unicode OK).
    - RAW (NEF/RAW/DNG/ARW/CR2/CR3/...): usa rawpy si está instalado.
    """
    ext = os.path.splitext(path)[1].lower().strip()

    # ---- RAW ----
    if ext in RAW_EXTS:
        if rawpy is None:
            raise RuntimeError(
                "Para abrir archivos RAW (.nef/.raw/.dng/.arw/.cr2/...) instala la dependencia:\n"
                "pip install rawpy"
            )
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as e:
            raise RuntimeError(f"Error leyendo RAW {path}: {e}")

    # ---- Imágenes comunes ----
    try:
        with open(path, "rb") as f:
            data = f.read()

        # Crear array de bytes y decodificar con OpenCV
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        print(f"Error leyendo imagen {path}: {e}")
        return None


def load_images_full_and_small(paths, align_max_side=1600):
    """
    Carga imágenes full-res (float32) y una versión downscale para alineación.
    Devuelve: full_imgs, small_imgs, scale_factor_small
    """
    full_imgs = []
    small_imgs = []
    scale_factors = []
    for p in paths:
        im = imread_unicode(p)
        if im is None:
            continue
        full_imgs.append(im.astype(np.float32))
        small, s = _resize_keep_aspect(im, align_max_side)
        small_imgs.append(small.astype(np.float32))
        scale_factors.append(s)
    if len(scale_factors) == 0:
        scale_factors = [1.0]
    return full_imgs, small_imgs, scale_factors[0]


def list_images_in_folder(folder):
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(IMG_EXTS):
            files.append(os.path.join(folder, fn))
    return sorted(files)

# -------------------------
# Alineación mejorada
# -------------------------

def align_ecc_multiscale(
    small_imgs,
    full_imgs,
    scale_small,
    motion=cv2.MOTION_AFFINE,
    number_of_iterations=220,
    termination_eps=1e-6,
    progress_cb=None,
):
    """
    Calcula transformaciones ECC en imágenes pequeñas y las aplica a full-res.
    No altera color/contraste.

    progress_cb(stage:str, current:int, total:int, src_preview_bgr:np.ndarray|None)
    """
    if len(small_imgs) < 2:
        return full_imgs

    ref_small = small_imgs[0].astype(np.uint8)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)

    aligned_full = [full_imgs[0]]
    sz_full = full_imgs[0].shape[:2]  # (h,w)

    warp_mode = motion
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    total = len(small_imgs)
    if progress_cb:
        progress_cb("align", 1, total, full_imgs[0].astype(np.uint8))

    for i in range(1, len(small_imgs)):
        im_small = small_imgs[i].astype(np.uint8)
        im_gray = cv2.cvtColor(im_small, cv2.COLOR_BGR2GRAY)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            wm = np.eye(3, 3, dtype=np.float32)
        else:
            wm = np.eye(2, 3, dtype=np.float32)

        try:
            cv2.findTransformECC(
                ref_gray,
                im_gray,
                wm,
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=7,
            )
        except cv2.error:
            aligned_full.append(full_imgs[i])
            if progress_cb:
                progress_cb("align", i + 1, total, full_imgs[i].astype(np.uint8))
            continue

        # Escala traslación a full-res
        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            wm_full = wm.copy()
            wm_full[0, 2] /= scale_small
            wm_full[1, 2] /= scale_small
        else:
            wm_full = wm.copy()
            wm_full[0, 2] /= scale_small
            wm_full[1, 2] /= scale_small

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            im_aligned = cv2.warpPerspective(
                full_imgs[i],
                wm_full,
                (sz_full[1], sz_full[0]),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
        else:
            im_aligned = cv2.warpAffine(
                full_imgs[i],
                wm_full,
                (sz_full[1], sz_full[0]),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )

        aligned_full.append(im_aligned)

        if progress_cb:
            progress_cb("align", i + 1, total, im_aligned.astype(np.uint8))

    return aligned_full

# -------------------------
# Parámetros globales ajustables por sliders
# -------------------------
# Cada método tendrá 2 parámetros: radio y suavizado
# -> PMax:         RADIUS_PMAX,      SMOOTH_PMAX
# -> Promedio:     RADIUS_WEIGHTED,  SMOOTH_WEIGHTED
# -> Profundidad:  RADIUS_DEPTH,     SMOOTH_DEPTH (para mapa de profundidad)

# PMax (Align + PMax)
RADIUS_PMAX = 5          # Radio de detalle PMax
SMOOTH_PMAX = 1          # Suavizado de mapa de foco PMax

# Promedio ponderado
RADIUS_WEIGHTED = 5      # Radio de detalle en promedio ponderado
SMOOTH_WEIGHTED = 1      # Suavizado de mapa de foco en promedio ponderado

# Mapa de profundidad
RADIUS_DEPTH = 5         # Radio de detalle para mapa de profundidad
SMOOTH_DEPTH = 4         # Suavizado del mapa de profundidad (y del mapa de foco)

# Integración (astronomía / suma de señal) - parámetros por método (independientes)
INT_PARAMS = {
    "mean": {
        "align_max_side": 1600,   # downscale máximo para alineación (px). Más alto = más preciso, más lento
        "ecc_iters": 120,         # iteraciones ECC (alineación)
        "ecc_eps_exp": 6,         # epsilon ECC = 1e-ecc_eps_exp
    },
    "median": {
        "align_max_side": 1600,
        "ecc_iters": 120,
        "ecc_eps_exp": 6,
    },
    "sum": {
        "align_max_side": 1600,
        "ecc_iters": 120,
        "ecc_eps_exp": 6,
    },
    "sigma": {
        "align_max_side": 1600,
        "ecc_iters": 120,
        "ecc_eps_exp": 6,
        "sigma_k": 3.0,           # sigma-clipping: k (umbral en sigmas)
    },
}

# Filtros de ruido (parámetros por sliders)
BM3D_SIGMA_PSD = 0.06            # 0.03–0.08 típico (escala 0..1)

# Bilateral
BILATERAL_D = 9                  # diámetro vecindario
BILATERAL_SIGMA_COLOR = 75       # fuerza por color
BILATERAL_SIGMA_SPACE = 75       # alcance espacial

# Wavelet Denoising
WAVELET_LEVEL = 2                # nivel de descomposición (1..5)

# Non-Local Means (OpenCV)
NLM_H = 10                       # fuerza (luma)
NLM_H_COLOR = 10                 # fuerza (color)
NLM_TEMPLATE_WINDOW = 7          # impar (3..7)
NLM_SEARCH_WINDOW = 21           # impar (7..31)

# Filtrado en Fourier
FOURIER_CUTOFF_RATIO = 0.12      # 0.05..0.30 (típico)
FOURIER_SOFTEN_RATIO = 0.05      # 0.01..0.10 (suavizado del borde)

# Noise2Noise / Noise2Void (denoising simple)
N2V_SIGMA = 1.2                  # sigma del GaussianBlur interno
N2V_THR_MULT = 1.6               # multiplicador de umbral (mediana + k*sigma)
N2V_ITERATIONS = 1               # iteraciones (1..5)

# Filtro Mediana
MEDIAN_KSIZE = 3                 # impar (3..25)

# Filtro Gaussiano (por bloques)
GAUSSIAN_SIGMA = 1.0             # sigma
GAUSSIAN_TILE_SIZE = 512         # tamaño de bloque (px)

def focus_measure_robust(gray_u8, radius):
    """Medida de foco robusta para decidir qué frame tiene más detalle en cada píxel.

    Se ha ajustado para que, con radius=5 (valor por defecto de los sliders),
    el comportamiento sea idéntico al del script anterior «apilador (23).py».
    Para otros valores de ``radius`` se escala el tamaño del kernel y los sigmas.
    """
    g = gray_u8.astype(np.float32)

    # Normalizar el radio a entero razonable
    try:
        r = int(round(float(radius)))
    except Exception:
        r = 5

    # ---- Caso especial: radius = 5  -> mismos parámetros que apilador (23) ----
    # ksize=3, sigmas 1.0 y 2.0, blur final 1.0
    if r == 5:
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))

        g1 = cv2.GaussianBlur(g, (0, 0), 1.0)
        g2 = cv2.GaussianBlur(g, (0, 0), 2.0)
        dog = np.abs(g1 - g2)

        mean = cv2.GaussianBlur(g, (0, 0), 2.0)
        mean2 = cv2.GaussianBlur(g * g, (0, 0), 2.0)
        var = np.maximum(mean2 - mean * mean, 0.0)

        fm = lap + 0.6 * dog + 0.4 * var
        fm = cv2.GaussianBlur(fm, (0, 0), 1.0)
        return fm

    # ---- Resto de valores: escalado en función de r ----
    r = max(1, r)

    # Kernel del Laplaciano según radio (más radio = kernel más grande)
    if r <= 2:
        k_lap = 3
    elif r <= 5:
        k_lap = 5
    else:
        k_lap = 7

    # Sigmas de los gaussianos en función del radio.
    # Elegidos para que alrededor de r=5 sean similares a (1.0, 2.0, 1.0).
    sigma1 = 0.2 * r      # ≈1.0 cuando r=5
    sigma2 = 0.4 * r      # ≈2.0 cuando r=5
    sigma_final = 0.2 * r # ≈1.0 cuando r=5

    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=k_lap))

    g1 = cv2.GaussianBlur(g, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma2)
    dog = np.abs(g1 - g2)

    mean = cv2.GaussianBlur(g, (0, 0), sigma2)
    mean2 = cv2.GaussianBlur(g * g, (0, 0), sigma2)
    var = np.maximum(mean2 - mean * mean, 0.0)

    fm = lap + 0.6 * dog + 0.4 * var
    fm = cv2.GaussianBlur(fm, (0, 0), sigma_final)
    return fm

# -------------------------
# PMax multi-banda real
# -------------------------

def build_gaussian_pyramid(img, levels):
    gp = [img]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(img, levels):
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        up = cv2.pyrUp(gp[i + 1], dstsize=size)
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def pmax_multiband(images_full, levels=5, weight_temp=0.65, progress_cb=None):
    """
    Fusión PMax multi-banda (sin tocar color/contraste).
    Usa RADIUS_PMAX para la medida de foco (slider "Radio PMax").
    """
    n = len(images_full)
    if n == 0:
        return None
    if n == 1:
        return images_full[0].astype(np.uint8)

    focus_maps = []
    for idx, im in enumerate(images_full):
        g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        fm = focus_measure_robust(g, RADIUS_PMAX)
        # Suavizado adicional controlado por slider de PMax
        if SMOOTH_PMAX > 1:
            fm = cv2.GaussianBlur(fm, (0, 0), float(SMOOTH_PMAX))
        focus_maps.append(fm)

        if progress_cb:
            progress_cb("focus", idx + 1, n, im.astype(np.uint8))

    focus_stack = np.stack(focus_maps, axis=0)  # (N,H,W)

    focus_stack = np.stack(focus_maps, axis=0)  # (N,H,W)

    mx = np.max(focus_stack, axis=0, keepdims=True)
    expv = np.exp((focus_stack - mx) * weight_temp)
    weights = expv / (np.sum(expv, axis=0, keepdims=True) + 1e-8)

    img_lps = [build_laplacian_pyramid(im, levels) for im in images_full]
    w_gps = [build_gaussian_pyramid(weights[i].astype(np.float32), levels) for i in range(n)]

    fused_lp = []
    for lvl in range(levels + 1):
        lvl_h, lvl_w = img_lps[0][lvl].shape[:2]
        acc = np.zeros_like(img_lps[0][lvl], dtype=np.float32)
        wsum = np.zeros((lvl_h, lvl_w), dtype=np.float32)

        for i in range(n):
            w_lvl = w_gps[i][lvl]
            if w_lvl.shape[:2] != (lvl_h, lvl_w):
                w_lvl = cv2.resize(w_lvl, (lvl_w, lvl_h), interpolation=cv2.INTER_LINEAR)
            acc += img_lps[i][lvl] * w_lvl[:, :, None]
            wsum += w_lvl

        acc /= (wsum[:, :, None] + 1e-8)
        fused_lp.append(acc)

        if progress_cb:
            preview = fused_lp[-1]
            preview_u8 = np.clip(preview, 0, 255).astype(np.uint8)
            progress_cb("pyramid", lvl + 1, levels + 1, preview_u8)

    out = fused_lp[-1]
    for lvl in reversed(range(levels)):
        size = (fused_lp[lvl].shape[1], fused_lp[lvl].shape[0])
        out = cv2.pyrUp(out, dstsize=size)
        out = out + fused_lp[lvl]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# -------------------------
# Post-procesado después de método Pirámide
# -------------------------

def pmax_multiband(images_full, levels=5, weight_temp=0.65, progress_cb=None):
    """
    Fusión PMax multi-banda (sin tocar color/contraste).
    weight_temp controla dureza de pesos (más alto = menos mezcla).

    progress_cb(stage:str, current:int, total:int, preview_bgr:np.ndarray|None)
    """
    n = len(images_full)
    if n == 0:
        return None
    if n == 1:
        return images_full[0].astype(np.uint8)

    focus_maps = []
    for idx, im in enumerate(images_full):
        g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        focus_maps.append(focus_measure_robust(g, RADIUS_PMAX))

        if progress_cb:
            progress_cb("focus", idx + 1, n, im.astype(np.uint8))

    focus_stack = np.stack(focus_maps, axis=0)  # (N,H,W)

    mx = np.max(focus_stack, axis=0, keepdims=True)
    expv = np.exp((focus_stack - mx) * weight_temp)
    weights = expv / (np.sum(expv, axis=0, keepdims=True) + 1e-8)

    img_lps = [build_laplacian_pyramid(im, levels) for im in images_full]
    w_gps = [build_gaussian_pyramid(weights[i].astype(np.float32), levels) for i in range(n)]

    fused_lp = []
    for lvl in range(levels + 1):
        lvl_h, lvl_w = img_lps[0][lvl].shape[:2]
        acc = np.zeros_like(img_lps[0][lvl], dtype=np.float32)
        wsum = np.zeros((lvl_h, lvl_w), dtype=np.float32)

        for i in range(n):
            w_lvl = w_gps[i][lvl]
            if w_lvl.shape[:2] != (lvl_h, lvl_w):
                w_lvl = cv2.resize(w_lvl, (lvl_w, lvl_h), interpolation=cv2.INTER_LINEAR)
            acc += img_lps[i][lvl] * w_lvl[:, :, None]
            wsum += w_lvl

        acc /= (wsum[:, :, None] + 1e-8)
        fused_lp.append(acc)

        if progress_cb:
            preview = fused_lp[-1]
            preview_u8 = np.clip(preview, 0, 255).astype(np.uint8)
            progress_cb("pyramid", lvl + 1, levels + 1, preview_u8)

    out = fused_lp[-1]
    for lvl in reversed(range(levels)):
        size = (fused_lp[lvl].shape[1], fused_lp[lvl].shape[0])
        out = cv2.pyrUp(out, dstsize=size)
        out = out + fused_lp[lvl]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def focus_weighted_average(images_full, weight_temp=2.0, progress_cb=None):
    """
    Fusión por promedio ponderado tipo (método de promedio).
    Usa la misma medida de foco que PMax, pero mezcla las imágenes
    directamente en espacio BGR usando pesos continuos.

    progress_cb(stage:str, current:int, total:int, preview_bgr:np.ndarray|None)
    """
    n = len(images_full)
    if n == 0:
        return None
    if n == 1:
        return images_full[0].astype(np.uint8)

    # 1) Medida de foco por imagen
    focus_maps = []
    for idx, im in enumerate(images_full):
        g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        fm = focus_measure_robust(g, RADIUS_WEIGHTED)
        # Suavizado adicional controlado por slider de promedio ponderado
        if SMOOTH_WEIGHTED > 1:
            fm = cv2.GaussianBlur(fm, (0, 0), float(SMOOTH_WEIGHTED))
        focus_maps.append(fm)
        if progress_cb:
            progress_cb("focusW", idx + 1, n, im.astype(np.uint8))

    focus_stack = np.stack(focus_maps, axis=0)  # (N,H,W)

    # 2) Softmax espacial para pesos
    mx = np.max(focus_stack, axis=0, keepdims=True)
    expv = np.exp((focus_stack - mx) * weight_temp)
    weights = expv / (np.sum(expv, axis=0, keepdims=True) + 1e-8)  # (N,H,W)

    # 3) Promedio ponderado directamente en BGR
    h, w = images_full[0].shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(n):
        w_i = weights[i]
        acc += images_full[i] * w_i[:, :, None]
        if progress_cb:
            preview = np.clip(acc / (i + 1), 0, 255).astype(np.uint8)
            progress_cb("fusionW", i + 1, n, preview)

    out = np.clip(acc, 0, 255).astype(np.uint8)
    return out


def focus_depth_map_fusion(images_full, smooth_ksize=9, smooth_sigma=2.0, progress_cb=None):
    """
    Fusión tipo 'mapa de profundidad' (similar al método depth map).
    Para cada píxel elige el frame con mayor foco (índice de profundidad),
    suaviza el mapa y reconstruye tomando los píxeles desde cada frame.

    progress_cb(stage:str, current:int, total:int, preview_bgr:np.ndarray|None)
    """
    n = len(images_full)
    if n == 0:
        return None
    if n == 1:
        return images_full[0].astype(np.uint8)

    # 1) Medida de foco por imagen
    focus_maps = []
    for idx, im in enumerate(images_full):
        g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # Usa radio específico del método de profundidad
        fm = focus_measure_robust(g, RADIUS_DEPTH)
        focus_maps.append(fm)
        if progress_cb:
            progress_cb("focusD", idx + 1, n, im.astype(np.uint8))

    focus_stack = np.stack(focus_maps, axis=0)  # (N,H,W)

    # 2) Índice del frame con máximo foco = mapa de profundidad bruto
    depth_idx = np.argmax(focus_stack, axis=0).astype(np.float32)  # (H,W)

    # 3) Suavizar el mapa de profundidad para reducir artefactos
    if smooth_ksize > 1:
        k = smooth_ksize | 1  # asegurar impar
        depth_blur = cv2.GaussianBlur(depth_idx, (k, k), smooth_sigma)
        depth_idx = np.rint(depth_blur)

    depth_idx = np.clip(depth_idx, 0, n - 1).astype(np.int32)

    # Preview del mapa de profundidad coloreado
    if progress_cb:
        depth_vis = cv2.normalize(depth_idx.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        progress_cb("refineD", 1, 1, depth_color)

    # 4) Reconstrucción según el mapa de profundidad
    h, w = images_full[0].shape[:2]
    fused = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(n):
        mask = (depth_idx == i)
        if not np.any(mask):
            continue
        fused[mask] = images_full[i][mask]
        if progress_cb:
            preview = np.clip(fused, 0, 255).astype(np.uint8)
            progress_cb("fusionD", i + 1, n, preview)

    out = np.clip(fused, 0, 255).astype(np.uint8)
    return out

def postprocess_clarity_like(img_bgr):
    """
    Aplica una cadena suave de mejoras locales:
    1) Normalización leve de contraste global
    2) CLAHE sobre luminancia (LAB)
    3) Unsharp mask (nitidez controlada)
    """
    out = img_bgr.copy()
    # Normalización:
    NORM_ALPHA = 0     # mínimo deseado (0–10 típico). Sube si quedan lavadas.
    NORM_BETA  = 255   # máximo deseado (240–255 típico). Baja si saturas blancos.

    # CLAHE (en canal L):
    CLAHE_CLIP_LIMIT = 0.8  # (1.2–4.0). Más alto = más contraste local.
    CLAHE_TILE_GRID  = (12, 12)  # (4x4 a 16x16). Más chico = efecto más fuerte/local.

    # Unsharp mask:
    UNSHARP_RADIUS = 1.2   # sigma del blur. (0.6–2.5). Más alto = halos más grandes.
    UNSHARP_AMOUNT = 0.8   # ganancia de nitidez. (0.5–2.0). Más alto = más nitidez.
    UNSHARP_THRESH = 2     # umbral para evitar ruido. (0–10). Sube si aparece grano.
    # ==========================================================

    out = cv2.normalize(out, None, alpha=NORM_ALPHA, beta=NORM_BETA, norm_type=cv2.NORM_MINMAX)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(out, (0, 0), UNSHARP_RADIUS)
    sharp = cv2.addWeighted(out, 1.0 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)

    if UNSHARP_THRESH > 0:
        diff = cv2.absdiff(out, blur)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, UNSHARP_THRESH, 255, cv2.THRESH_BINARY)
        mask3 = cv2.merge([mask, mask, mask])
        out = np.where(mask3 > 0, sharp, out).astype(np.uint8)
    else:
        out = np.clip(sharp, 0, 255).astype(np.uint8)

    return out

def apply_bilateral_filter(img_bgr, d=9, sigma_color=75, sigma_space=75):
    """Filtro bilateral (suaviza ruido preservando bordes)."""
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    # d = diámetro del vecindario; sigma_color/sigma_space controlan fuerza y alcance
    return cv2.bilateralFilter(
        img_bgr,
        d=int(d),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )


def apply_median_filter(img_bgr, ksize=3):
    """Filtro de Mediana (muy útil para ruido tipo 'sal y pimienta')."""
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        img = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        img = img_bgr

    k = int(ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    return cv2.medianBlur(img, k)


def apply_gaussian_filter(img_bgr, sigma=1.0, tile_size=512):
    """Filtro Gaussiano por bloques (NO bloquea la UI)."""
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        img = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        img = img_bgr

    s = float(sigma)
    s = max(0.3, min(s, 3.0))  # límite seguro

    h, w = img.shape[:2]
    out = np.zeros_like(img)

    for y in range(0, h, tile_size):
        y2 = min(y + tile_size, h)
        for x in range(0, w, tile_size):
            x2 = min(x + tile_size, w)

            tile = img[y:y2, x:x2]
            blurred = cv2.GaussianBlur(tile, (0, 0), s)
            out[y:y2, x:x2] = blurred

    return out

def apply_fourier_filter(img_bgr, cutoff_ratio=0.12, soften_ratio=0.05):
    """
    Filtrado en Fourier (pasa-bajos suave) para reducir ruido de alta frecuencia.
    - cutoff_ratio: radio del pasa-bajos relativo al semieje menor (0.05..0.30 típico)
    - soften_ratio: suavizado del borde del filtro (0.01..0.10 típico)
    """
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        img = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        img = img_bgr

    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2

    # Radio de corte
    r0 = float(cutoff_ratio) * float(min(h, w) / 2.0)
    r0 = max(2.0, r0)

    # Suavizado del borde
    soft = float(soften_ratio) * float(min(h, w))
    soft = max(1.0, soft)

    # Máscara circular suave (tipo “sigmoide” radial)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # transicion suave alrededor de r0
    mask = 1.0 / (1.0 + np.exp((dist - r0) / soft))
    mask = mask.astype(np.float32)

    out = np.zeros_like(img, dtype=np.float32)

    for c in range(3):
        ch = img[:, :, c].astype(np.float32)

        F = np.fft.fft2(ch)
        Fshift = np.fft.fftshift(F)

        Ff = Fshift * mask

        ishift = np.fft.ifftshift(Ff)
        rec = np.fft.ifft2(ishift)
        rec = np.real(rec).astype(np.float32)

        out[:, :, c] = rec

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_noise2void_denoising(img_bgr, sigma=1.2, thr_mult=1.6, iterations=1):
    """
    Noise2Noise / Noise2Void (aprox. "blind-spot" simple):
    - Predice cada píxel desde vecinos usando blur gaussiano.
    - Reemplaza SOLO píxeles "ruidosos" (outliers) según umbral robusto.
    """
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        out = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        out = img_bgr.copy()

    iters = max(1, int(iterations))

    for _ in range(iters):
        pred = cv2.GaussianBlur(out, (0, 0), float(sigma))

        diff = cv2.absdiff(out, pred)
        diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)

        med = float(np.median(diff_g))
        mad = float(np.median(np.abs(diff_g - med)))
        sigma_est = 1.4826 * mad
        thr = max(1.0, float(thr_mult) * sigma_est)

        mask = diff_g > thr
        if np.any(mask):
            out[mask] = pred[mask]

    return out


def apply_wavelet_denoising(img_bgr, wavelet="db2", level=2):
    """
    Denoising por Wavelets (requiere PyWavelets).
    - Estima sigma desde detalles de alta frecuencia (nivel 1)
    - Umbral universal + soft-threshold en coeficientes de detalle
    """

    if img_bgr is None:
        return None

    if pywt is None:
        raise RuntimeError("Para usar 'Denoising por Wavelets' instala: pip install PyWavelets")

    if img_bgr.dtype != np.uint8:
        img = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        img = img_bgr

    img_f = img.astype(np.float32)

    def _denoise_channel(ch2d):
        coeffs = pywt.wavedec2(ch2d, wavelet=wavelet, level=int(level))

        # coeffs = [cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)]
        details = coeffs[1:]
        cH1, cV1, cD1 = details[-1]

        # estimación robusta de sigma (MAD) en alta frecuencia
        abs_cd1 = np.abs(cD1)
        med = np.median(abs_cd1)
        sigma = med / 0.6745 if med > 0 else 0.0

        # umbral universal
        uthresh = sigma * np.sqrt(2.0 * np.log(max(2, ch2d.size)))

        new_coeffs = [coeffs[0]]
        for (cH, cV, cD) in details:
            new_coeffs.append((
                pywt.threshold(cH, uthresh, mode="soft"),
                pywt.threshold(cV, uthresh, mode="soft"),
                pywt.threshold(cD, uthresh, mode="soft"),
            ))

        rec = pywt.waverec2(new_coeffs, wavelet=wavelet)

        # rec puede quedar un poco más grande
        rec = rec[:ch2d.shape[0], :ch2d.shape[1]]
        return rec

    out = np.zeros_like(img_f, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = _denoise_channel(img_f[:, :, c])

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def apply_nlm_denoising(
    img_bgr,
    h=10,
    hColor=10,
    templateWindowSize=7,
    searchWindowSize=21,
):
    """
    Denoising Non-Local Means (NLM) usando OpenCV.
    - Color: fastNlMeansDenoisingColored
    - Gris:  fastNlMeansDenoising
    """
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    img = img_bgr
    # Gris
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        if len(img.shape) == 3:
            img = img[:, :, 0]
        return cv2.fastNlMeansDenoising(
            img,
            None,
            h=float(h),
            templateWindowSize=int(templateWindowSize),
            searchWindowSize=int(searchWindowSize),
        )

    # Color (BGR)
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=float(h),
        hColor=float(hColor),
        templateWindowSize=int(templateWindowSize),
        searchWindowSize=int(searchWindowSize),
    )


def apply_bm3d_denoising(
    img_bgr,
    sigma_psd=0.06,
    progress_cb=None,
    tile_size=768,
    overlap=48,
    tiling_auto=True,
):
    """
    Denoising BM3D (Block-Matching 3D) con progreso REAL (por canal y por bloques).

    Requiere el paquete 'bm3d':
        pip install bm3d

    sigma_psd: nivel de ruido estimado en escala 0..1 (típico 0.03–0.08).
    progress_cb(stage_text:str, current:int, total:int, preview_bgr:np.ndarray|None)
    """
    if img_bgr is None:
        return None

    if bm3d is None:
        raise RuntimeError("BM3D no está disponible. Instala el paquete: pip install bm3d")

    img = img_bgr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    def _bm3d_run(ch01):
        last = None
        stage_enum = getattr(bm3d, "BM3DStages", None)

        attempts = [
            lambda: bm3d.bm3d(ch01, sigma_psd=sigma_psd),
            lambda: bm3d.bm3d(ch01, sigma_psd=sigma_psd),
            lambda: bm3d.bm3d(ch01, sigma_psd),
        ]
        if stage_enum is not None and hasattr(stage_enum, "ALL_STAGES"):
            attempts.insert(1, lambda: bm3d.bm3d(ch01, sigma_psd=sigma_psd, stage_arg=stage_enum.ALL_STAGES))
            attempts.append(lambda: bm3d.bm3d(ch01, sigma_psd, stage_arg=stage_enum.ALL_STAGES))

        for fn in attempts:
            try:
                out = fn()
                if out is not None:
                    return out
            except Exception as e:
                last = e

        raise RuntimeError(f"Error ejecutando BM3D: {last}")

    def _tile_starts(L, ts, step):
        if L <= ts:
            return [0]
        starts = list(range(0, L - ts + 1, step))
        last = L - ts
        if starts[-1] != last:
            starts.append(last)
        # únicos y ordenados
        starts = sorted(set(starts))
        return starts

    def _feather_mask(th, tw, ov, fade_top, fade_bottom, fade_left, fade_right):
        ov = int(ov)
        if ov <= 0:
            return np.ones((th, tw), dtype=np.float32)

        ov_y = min(ov, th // 2)
        ov_x = min(ov, tw // 2)
        if ov_y <= 0 or ov_x <= 0:
            return np.ones((th, tw), dtype=np.float32)

        wy = np.ones(th, dtype=np.float32)
        wx = np.ones(tw, dtype=np.float32)

        # ramp 0..1 suave
        ry = (np.sin(np.linspace(0.0, np.pi / 2.0, ov_y, dtype=np.float32)) ** 2).astype(np.float32)
        rx = (np.sin(np.linspace(0.0, np.pi / 2.0, ov_x, dtype=np.float32)) ** 2).astype(np.float32)

        if fade_top:
            wy[:ov_y] = ry
        if fade_bottom:
            wy[-ov_y:] = ry[::-1]
        if fade_left:
            wx[:ov_x] = rx
        if fade_right:
            wx[-ov_x:] = rx[::-1]

        return (wy[:, None] * wx[None, :]).astype(np.float32)

    def _bm3d_tiled(ch01, channel_idx, channels_total, label):
        h, w = ch01.shape[:2]
        ts = max(128, int(tile_size))
        ov = max(0, int(overlap))

        step = ts - 2 * ov
        if step <= 0:
            step = ts

        xs = _tile_starts(w, ts, step)
        ys = _tile_starts(h, ts, step)

        total_tiles = max(1, len(xs) * len(ys))
        global_total = max(1, channels_total * total_tiles)

        # limitar updates (≈ 80 updates por canal)
        update_every = max(1, total_tiles // 80)

        acc = np.zeros((h, w), dtype=np.float32)
        wacc = np.zeros((h, w), dtype=np.float32)

        t = 0
        for y0 in ys:
            y1 = min(y0 + ts, h)
            for x0 in xs:
                x1 = min(x0 + ts, w)

                tile = ch01[y0:y1, x0:x1]
                den_tile = _bm3d_run(tile)

                fade_top = (y0 > 0)
                fade_bottom = (y1 < h)
                fade_left = (x0 > 0)
                fade_right = (x1 < w)

                mask = _feather_mask(
                    den_tile.shape[0], den_tile.shape[1],
                    ov,
                    fade_top, fade_bottom, fade_left, fade_right
                )

                acc[y0:y1, x0:x1] += (den_tile.astype(np.float32) * mask)
                wacc[y0:y1, x0:x1] += mask

                t += 1
                if progress_cb and (t == 1 or t == total_tiles or (t % update_every == 0)):
                    cur = channel_idx * total_tiles + t
                    progress_cb(f"BM3D {label} — bloque {t}/{total_tiles}", cur, global_total, None)

        out = acc / (wacc + 1e-8)
        return out

    # Decidir si usar bloques (para progreso más granular y evitar “parece tildado”)
    h, w = img.shape[:2]
    use_tiling = False
    if tiling_auto:
        use_tiling = (h * w >= 2_000_000) or (max(h, w) >= 2000)
    else:
        use_tiling = True

    # Gris
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        gray01 = gray.astype(np.float32) / 255.0

        if progress_cb:
            progress_cb("BM3D (gris) — iniciando…", 0, 1, None)

        if use_tiling:
            den01 = _bm3d_tiled(gray01, 0, 1, "(gris)")
        else:
            den01 = _bm3d_run(gray01)
            if progress_cb:
                progress_cb("BM3D (gris) — finalizando…", 1, 1, None)

        den = np.clip(den01 * 255.0, 0, 255).astype(np.uint8)
        return den

    # Color (BGR): canal por canal, con progreso real
    img01 = img.astype(np.float32) / 255.0
    out01 = np.empty_like(img01)

    channels_total = img01.shape[2]
    if progress_cb:
        progress_cb("BM3D (color) — iniciando…", 0, 1, None)

    for c in range(channels_total):
        label = f"(canal {c+1}/{channels_total})"

        if progress_cb:
            progress_cb(f"BM3D {label} — preparando…", c, channels_total, None)

        if use_tiling:
            out01[:, :, c] = _bm3d_tiled(img01[:, :, c], c, channels_total, label)
        else:
            out01[:, :, c] = _bm3d_run(img01[:, :, c])
            if progress_cb:
                progress_cb(f"BM3D {label} — listo", c + 1, channels_total, None)

    return np.clip(out01 * 255.0, 0, 255).astype(np.uint8)

# -------------------------
# Pipeline reusable (single o batch)
# -------------------------

def stack_paths(paths, progress_cb=None):
    """
    Ejecuta todo el pipeline sobre una lista de paths con PMax.
    progress_cb(stage, current, total, preview_bgr)
    """
    full_imgs, small_imgs, scale_small = load_images_full_and_small(paths)
    if len(full_imgs) < 2:
        raise RuntimeError("No se pudieron cargar suficientes imágenes.")

    if progress_cb:
        progress_cb("load", 1, 1, full_imgs[0].astype(np.uint8))

    aligned_full = align_ecc_multiscale(
        small_imgs,
        full_imgs,
        scale_small,
        motion=cv2.MOTION_AFFINE,
        progress_cb=progress_cb,
    )

    # --- PMax puro (RAW sin filtros) ---
    result_raw = pmax_multiband(
        aligned_full,
        levels=5,
        weight_temp=2,
        progress_cb=progress_cb,
    )
    if result_raw is None:
        raise RuntimeError("Falló el apilamiento.")

    # Preview para la interfaz
    if progress_cb:
        progress_cb("post", 1, 1, result_raw)

    # ========================================================
    #   GUARDAR RESULTADO RAW (antes de postprocesado)
    # ========================================================
    # Se guarda automáticamente SOLO si hay carpeta de autosave definida.
    # Se permite personalizar el nombre base vía FocusStackerApp.current_raw_name.
    if hasattr(FocusStackerApp, "current_auto_save_dir"):
        base_raw = getattr(FocusStackerApp, "current_raw_name", "resultado_pmax_RAW")
        fmt = getattr(FocusStackerApp, "current_output_format", "jpg")
        fmt = (fmt or "jpg").strip().lower()
        if fmt in ("jpeg", "jpg"):
            ext = ".jpg"
        elif fmt == "png":
            ext = ".png"
        elif fmt in ("tif", "tiff"):
            ext = ".tiff"
        else:
            ext = ".jpg"

        raw_out = os.path.join(FocusStackerApp.current_auto_save_dir, base_raw + ext)
        k = 1
        while os.path.exists(raw_out):
            raw_out = os.path.join(
                FocusStackerApp.current_auto_save_dir,
                f"{base_raw}_{k}" + ext
            )
            k += 1
        # Guardar RAW sin filtros
        rgb = cv2.cvtColor(result_raw, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if ext == ".jpg":
            pil.save(raw_out, format="JPEG", quality=100, subsampling=0)
        elif ext == ".png":
            pil.save(raw_out, format="PNG", compress_level=0)
        else:
            pil.save(raw_out, format="TIFF", compression="tiff_lzw")

    # --- Aplicar filtros clarity-like ---
    result = postprocess_clarity_like(result_raw)
    return result

def stack_paths_weighted(paths, progress_cb=None):
    """
    Pipeline alternativo: promedio ponderado tipo Focus.
    Alinea igual que PMax, pero fusiona con focus_weighted_average
    y NO aplica post-procesado clarity-like (para mantener color original).
    """
    full_imgs, small_imgs, scale_small = load_images_full_and_small(paths)
    if len(full_imgs) < 2:
        raise RuntimeError("No se pudieron cargar suficientes imágenes.")

    if progress_cb:
        progress_cb("load", 1, 1, full_imgs[0].astype(np.uint8))

    aligned_full = align_ecc_multiscale(
        small_imgs,
        full_imgs,
        scale_small,
        motion=cv2.MOTION_AFFINE,
        progress_cb=progress_cb,
    )

    result = focus_weighted_average(
        aligned_full,
        weight_temp=2.0,
        progress_cb=progress_cb,
    )
    if result is None:
        raise RuntimeError("Falló el apilamiento (promedio ponderado).")

    # === Aplicar el MISMO post-procesado clarity-like que usa PMax ===
    result = postprocess_clarity_like(result)

    return result

def stack_paths_depth(paths, progress_cb=None):
    """
    Pipeline alternativo: mapa de profundidad tipo.
    Alinea igual que PMax y fusiona con focus_depth_map_fusion.
    Usa SMOOTH_DEPTH (slider) para el suavizado del mapa de profundidad.
    """
    full_imgs, small_imgs, scale_small = load_images_full_and_small(paths)
    if len(full_imgs) < 2:
        raise RuntimeError("No se pudieron cargar suficientes imágenes.")

    if progress_cb:
        progress_cb("load", 1, 1, full_imgs[0].astype(np.uint8))

    aligned_full = align_ecc_multiscale(
        small_imgs,
        full_imgs,
        scale_small,
        motion=cv2.MOTION_AFFINE,
        progress_cb=progress_cb,
    )

    # Defaults equivalentes a Anterior.py cuando SMOOTH_DEPTH=4  -> ksize=9, sigma=2.0
    smooth_val = max(1, int(SMOOTH_DEPTH))
    smooth_ksize = (smooth_val * 2) + 1  # siempre impar
    smooth_sigma = smooth_val / 2.0

    result = focus_depth_map_fusion(
        aligned_full,
        smooth_ksize=smooth_ksize,
        smooth_sigma=smooth_sigma,
        progress_cb=progress_cb,
    )
    if result is None:
        raise RuntimeError("Falló el apilamiento (mapa de profundidad).")

    return result

def _integration_prepare_aligned(paths, progress_cb=None, align_max_side=1600, ecc_iters=120, ecc_eps_exp=6):
    full_imgs, small_imgs, scale_small = load_images_full_and_small(
        paths,
        align_max_side=int(align_max_side),
    )
    if len(full_imgs) < 2:
        raise RuntimeError("No se pudieron cargar suficientes imágenes.")

    if progress_cb:
        progress_cb("load", 1, 1, full_imgs[0].astype(np.uint8))

    try:
        eps = float(10.0 ** (-int(ecc_eps_exp)))
    except Exception:
        eps = 1e-6

    aligned_full = align_ecc_multiscale(
        small_imgs,
        full_imgs,
        scale_small,
        motion=cv2.MOTION_AFFINE,
        number_of_iterations=int(ecc_iters),
        termination_eps=eps,
        progress_cb=progress_cb,
    )
    if len(aligned_full) < 2:
        raise RuntimeError("Falló el alineado ECC.")

    return aligned_full


def stack_paths_integration_mean(paths, progress_cb=None):
    p = INT_PARAMS.get("mean", {})
    aligned_full = _integration_prepare_aligned(
        paths,
        progress_cb=progress_cb,
        align_max_side=int(p.get("align_max_side", 1600)),
        ecc_iters=int(p.get("ecc_iters", 120)),
        ecc_eps_exp=int(p.get("ecc_eps_exp", 6)),
    )

    stack = np.stack([im.astype(np.float32) for im in aligned_full], axis=0)
    if progress_cb:
        progress_cb("integrate", 1, 1, aligned_full[0].astype(np.uint8))

    out = np.mean(stack, axis=0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def stack_paths_integration_median(paths, progress_cb=None):
    p = INT_PARAMS.get("median", {})
    aligned_full = _integration_prepare_aligned(
        paths,
        progress_cb=progress_cb,
        align_max_side=int(p.get("align_max_side", 1600)),
        ecc_iters=int(p.get("ecc_iters", 120)),
        ecc_eps_exp=int(p.get("ecc_eps_exp", 6)),
    )

    stack = np.stack([im.astype(np.float32) for im in aligned_full], axis=0)
    if progress_cb:
        progress_cb("integrate", 1, 1, aligned_full[0].astype(np.uint8))

    out = np.median(stack, axis=0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def stack_paths_integration_sum(paths, progress_cb=None):
    p = INT_PARAMS.get("sum", {})
    aligned_full = _integration_prepare_aligned(
        paths,
        progress_cb=progress_cb,
        align_max_side=int(p.get("align_max_side", 1600)),
        ecc_iters=int(p.get("ecc_iters", 120)),
        ecc_eps_exp=int(p.get("ecc_eps_exp", 6)),
    )

    stack = np.stack([im.astype(np.float32) for im in aligned_full], axis=0)
    if progress_cb:
        progress_cb("integrate", 1, 1, aligned_full[0].astype(np.uint8))

    out = np.sum(stack, axis=0)
    # Normalizar a 8-bit para evitar saturación extrema
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def stack_paths_integration_sigma_clipping(paths, progress_cb=None, k=None):
    p = INT_PARAMS.get("sigma", {})
    aligned_full = _integration_prepare_aligned(
        paths,
        progress_cb=progress_cb,
        align_max_side=int(p.get("align_max_side", 1600)),
        ecc_iters=int(p.get("ecc_iters", 120)),
        ecc_eps_exp=int(p.get("ecc_eps_exp", 6)),
    )

    if k is None:
        try:
            k = float(p.get("sigma_k", 3.0))
        except Exception:
            k = 3.0

    stack = np.stack([im.astype(np.float32) for im in aligned_full], axis=0)
    if progress_cb:
        progress_cb("integrate", 1, 1, aligned_full[0].astype(np.uint8))

    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0) + 1e-6
    mask = np.abs(stack - mean) <= (float(k) * std)

    masked_sum = np.sum(stack * mask, axis=0)
    masked_n = np.sum(mask, axis=0)
    out = masked_sum / np.maximum(masked_n, 1.0)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# -------------------------
# GUI
# -------------------------

class Knob(tk.Canvas):
    """
    Perilla circular (from_..to) controlada por IntVar, similar a tk.Scale pero en forma de perilla.
    - Click + arrastre para ajustar.
    - Usa iconos:
        - icons/perilla_externa.png (estática)
        - icons/perilla_interna.png (rota según el valor)
    - Muestra el valor en el centro, por encima de la perilla interna.
    - Soporta habilitar/deshabilitar + tema de colores.
    """
    _BASE_EXT_PIL = None
    _BASE_INT_PIL = None

    @classmethod
    def _load_base_images(cls):
        if cls._BASE_EXT_PIL is not None and cls._BASE_INT_PIL is not None:
            return
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        ext_path = os.path.join(icons_dir, "perilla_externa.png")
        int_path = os.path.join(icons_dir, "perilla_interna.png")
        cls._BASE_EXT_PIL = Image.open(ext_path).convert("RGBA")
        cls._BASE_INT_PIL = Image.open(int_path).convert("RGBA")

    @staticmethod
    def _make_disabled(pil_img):
        # Oscurecer + bajar contraste para estado "disabled"
        try:
            img = ImageEnhance.Brightness(pil_img).enhance(0.55)
            img = ImageEnhance.Contrast(img).enhance(0.85)
            return img
        except Exception:
            return pil_img

    def __init__(
        self,
        master,
        from_=1,
        to=10,
        variable=None,
        command=None,
        size=72,
        bg="#535353",
        fg="#f0f0f0",
        dial_color="#3d3d3d",
        disabled_bg="#3a3a3a",
        disabled_fg="#888888",
        disabled_dial="#222222",
        **kwargs
    ):
        super().__init__(
            master,
            width=size,
            height=size,
            bg=bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
            **kwargs
        )

        self.from_ = int(from_)
        self.to = int(to)
        if self.to <= self.from_:
            self.to = self.from_ + 1

        self.var = variable if variable is not None else tk.IntVar(value=self.from_)
        self.command = command

        # Tema (se puede actualizar luego con set_theme)
        self._bg_enabled = bg
        self._fg_enabled = fg
        self._dial_enabled = dial_color

        self._bg_disabled = disabled_bg
        self._fg_disabled = disabled_fg
        self._dial_disabled = disabled_dial

        self._enabled = True
        self._size = int(size)

        # Rango angular (270°) dejando un gap de 90° en el lado izquierdo.
        self._ang_start = -135.0
        self._ang_end = 135.0

        # Iconos (cachés por valor)
        self._icons_ok = False
        self._photo_ext_enabled = None
        self._photo_ext_disabled = None
        self._int_cache_enabled = {}
        self._int_cache_disabled = {}

        try:
            self._load_base_images()
            self._setup_icons()
            self._icons_ok = True
        except Exception:
            self._icons_ok = False

        # Eventos mouse
        self.bind("<Button-1>", self._on_mouse)
        self.bind("<B1-Motion>", self._on_mouse)

        # Actualizar cuando cambia la variable
        try:
            self.var.trace_add("write", lambda *args: self._redraw())
        except Exception:
            pass

        self._clamp_var()
        self._redraw()

    def _setup_icons(self):
        # Escala: externa ocupa el canvas completo; interna un poco más chica para dejar ver el anillo externo.
        ext_s = self._size
        int_s = max(10, int(round(self._size * 0.84)))

        ext_en = self._BASE_EXT_PIL.resize((ext_s, ext_s), Image.LANCZOS)
        int_en = self._BASE_INT_PIL.resize((int_s, int_s), Image.LANCZOS)

        ext_dis = self._make_disabled(ext_en)
        int_dis = self._make_disabled(int_en)

        self._pil_ext_enabled = ext_en
        self._pil_int_enabled = int_en
        self._pil_ext_disabled = ext_dis
        self._pil_int_disabled = int_dis

        self._photo_ext_enabled = ImageTk.PhotoImage(ext_en)
        self._photo_ext_disabled = ImageTk.PhotoImage(ext_dis)

        # Limpiar caches de rotaciones
        self._int_cache_enabled.clear()
        self._int_cache_disabled.clear()

    def set_theme(self, bg_enabled, fg_enabled, dial_enabled, bg_disabled, fg_disabled, dial_disabled):
        self._bg_enabled = bg_enabled
        self._fg_enabled = fg_enabled
        self._dial_enabled = dial_enabled

        self._bg_disabled = bg_disabled
        self._fg_disabled = fg_disabled
        self._dial_disabled = dial_disabled
        self._redraw()

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)
        self._redraw()

    def _clamp_var(self):
        try:
            v = int(self.var.get())
        except Exception:
            v = self.from_
        v = max(self.from_, min(self.to, v))
        try:
            self.var.set(v)
        except Exception:
            pass

    def _value_to_angle(self, v: int) -> float:
        if self.to == self.from_:
            return self._ang_start
        t = (float(v) - float(self.from_)) / float(self.to - self.from_)
        return self._ang_start + t * (self._ang_end - self._ang_start)

    def _angle_to_value(self, ang: float) -> int:
        # Clamp al rango permitido
        ang = max(self._ang_start, min(self._ang_end, ang))
        t = (ang - self._ang_start) / (self._ang_end - self._ang_start)
        v = self.from_ + int(round(t * (self.to - self.from_)))
        return max(self.from_, min(self.to, v))

    def _on_mouse(self, event):
        if not self._enabled:
            return

        w = int(self.cget("width"))
        h = int(self.cget("height"))
        cx, cy = w / 2.0, h / 2.0

        dx = float(event.x) - cx
        dy = cy - float(event.y)  # invertir eje Y para trigonometría

        # Evitar saltos si clickean justo al centro
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return

        ang = float(np.degrees(np.arctan2(dy, dx)))
        # Clamp al rango de la perilla
        ang = max(self._ang_start, min(self._ang_end, ang))

        v = self._angle_to_value(ang)
        try:
            self.var.set(v)
        except Exception:
            pass

        if self.command is not None:
            try:
                self.command(str(v))
            except Exception:
                try:
                    self.command(v)
                except Exception:
                    pass

        self._redraw()

    def _get_internal_photo(self, v: int, enabled: bool):
        if not self._icons_ok:
            return None

        cache = self._int_cache_enabled if enabled else self._int_cache_disabled
        if v in cache:
            return cache[v]

        base = self._pil_int_enabled if enabled else self._pil_int_disabled

        # La imagen base tiene el "indicador" apuntando arriba (90°). Ajustar a nuestro ángulo.
        ang = self._value_to_angle(v)
        rot = ang - 90.0

        try:
            rotated = base.rotate(
                rot,
                resample=Image.BICUBIC,
                expand=False,
                center=(base.size[0] / 2.0, base.size[1] / 2.0),
                fillcolor=(0, 0, 0, 0),
            )
        except TypeError:
            # Compatibilidad con Pillow antiguo (sin fillcolor)
            rotated = base.rotate(
                rot,
                resample=Image.BICUBIC,
                expand=False,
                center=(base.size[0] / 2.0, base.size[1] / 2.0),
            )

        photo = ImageTk.PhotoImage(rotated)
        cache[v] = photo
        return photo

    def _redraw_vector_fallback(self):
        # Fallback (dibujo original) si no se pueden cargar los iconos.
        self.delete("all")

        w = int(self.cget("width"))
        h = int(self.cget("height"))
        cx, cy = w / 2.0, h / 2.0

        pad = 8
        r = min(w, h) / 2.0 - pad

        bg = self._bg_enabled if self._enabled else self._bg_disabled
        fg = self._fg_enabled if self._enabled else self._fg_disabled
        dial = self._dial_enabled if self._enabled else self._dial_disabled

        try:
            self.configure(bg=bg)
        except Exception:
            pass

        # Anillo
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=dial, width=4)

        # Ticks (from_..to)
        for vv in range(self.from_, self.to + 1):
            a = self._value_to_angle(vv)
            th = float(np.deg2rad(a))
            r1 = r - 3
            r2 = r - 10 if (vv in (self.from_, self.to, (self.from_ + self.to) // 2)) else r - 7
            x1 = cx + r1 * float(np.cos(th))
            y1 = cy - r1 * float(np.sin(th))
            x2 = cx + r2 * float(np.cos(th))
            y2 = cy - r2 * float(np.sin(th))
            self.create_line(x1, y1, x2, y2, fill=fg, width=2)

        # Aguja
        self._clamp_var()
        try:
            v = int(self.var.get())
        except Exception:
            v = self.from_

        ang = self._value_to_angle(v)
        th = float(np.deg2rad(ang))
        rp = r - 14
        x2 = cx + rp * float(np.cos(th))
        y2 = cy - rp * float(np.sin(th))
        self.create_line(cx, cy, x2, y2, fill=fg, width=3, capstyle=tk.ROUND)

        # Centro
        self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=fg, outline="")

        # Texto valor (centro)
        fs = max(10, int(round(self._size * 0.22)))
        self.create_text(cx + 1, cy + 1, text=str(v), fill="#000000", font=("", fs, "bold"))
        self.create_text(cx, cy, text=str(v), fill=fg, font=("", fs, "bold"))

    def _redraw(self):
        if not self._icons_ok:
            self._redraw_vector_fallback()
            return

        self.delete("all")

        w = int(self.cget("width"))
        h = int(self.cget("height"))
        cx, cy = w / 2.0, h / 2.0

        bg = self._bg_enabled if self._enabled else self._bg_disabled
        fg = self._fg_enabled if self._enabled else self._fg_disabled

        try:
            self.configure(bg=bg)
        except Exception:
            pass

        self._clamp_var()
        try:
            v = int(self.var.get())
        except Exception:
            v = self.from_

        # Interna (rota) + Externa (estática)
        internal_photo = self._get_internal_photo(v, enabled=self._enabled)
        ext_photo = self._photo_ext_enabled if self._enabled else self._photo_ext_disabled

        if internal_photo is not None:
            self.create_image(cx, cy, image=internal_photo)
        if ext_photo is not None:
            self.create_image(cx, cy, image=ext_photo)

        # Texto valor centrado (por encima de la interna)
        fs = max(10, int(round(self._size * 0.22)))
        self.create_text(cx + 1, cy + 1, text=str(v), fill="#000000", font=("", fs, "bold"))
        self.create_text(cx, cy, text=str(v), fill=fg, font=("", fs, "bold"))

class FocusStackerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Ocultar durante el armado para evitar parpadeos / “recuadros” al iniciar
        self.withdraw()

        # --- paths base ---
        try:
            self._base_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            self._base_dir = os.getcwd()

        # =============================
        # Splash Intro (3s) + barra falsa
        # =============================
        self._splash_duration_ms = 3000 #tiempos imagen de inicio
        self._splash_tick_ms = 30
        self._splash_progress_var = tk.DoubleVar(value=0.0)
        self._splash_win = None
        self._splash_imgtk = None
        self._splash_tick_job = None
        self._splash_close_job = None
        self._splash_t0 = None
        self._startup_done = False

        self._intro_path = os.path.join(self._base_dir, "icons", "Intro.jpg")

        def _startup():
            if self._startup_done:
                return
            self._startup_done = True

            # --- Icono (ventana + barra de tareas) ---
            icon_path = os.path.join(self._base_dir, "icons", "logo.ico")
            if os.path.exists(icon_path):
                try:
                    self.iconbitmap(icon_path)
                except Exception:
                    pass
                try:
                    ico_img = Image.open(icon_path)
                    self._app_icon_imgtk = ImageTk.PhotoImage(ico_img)
                    self.iconphoto(True, self._app_icon_imgtk)  # default para futuras ventanas también
                except Exception:
                    pass

            self.title("Open Galileo v1.0 / Programa desarrollado por Brandon Antonio Segura Torres / @micro.cosmonauta / antoniosegura3d@gmail.com")
            self.minsize(1100, 680)

            self.image_paths = []
            self.folder_paths = []  # carpetas para batch
            self.result = None
            # --- guardado automatico ---
            self.auto_save_dir = None  # carpeta de guardado automático
            self.auto_save_label_var = tk.StringVar(value="Guardado automático: (no seleccionada)")

            self.output_format_var = tk.StringVar(value="jpg")  # formato de salida: jpg/png/tiff

            # Flags de métodos de apilado
            # 1) Apilamiento con focus tracking
            self.use_pmax = tk.BooleanVar(value=False)
            self.use_weighted = tk.BooleanVar(value=False)
            self.use_depth = tk.BooleanVar(value=False)

            # 2) Apilado por integración (astronomía / suma de señal)
            self.use_int_mean = tk.BooleanVar(value=False)
            self.use_int_median = tk.BooleanVar(value=False)
            self.use_int_sum = tk.BooleanVar(value=False)
            self.use_int_sigma = tk.BooleanVar(value=False)

            self.use_calibration = tk.BooleanVar(value=False)
            self.use_fourier = tk.BooleanVar(value=False)
            self.use_n2v = tk.BooleanVar(value=False)
            self.use_median = tk.BooleanVar(value=False)
            self.use_gaussian = tk.BooleanVar(value=False)
            self.use_bm3d = tk.BooleanVar(value=False)
            self.use_bilateral = tk.BooleanVar(value=False)
            self.use_wavelet = tk.BooleanVar(value=False)
            self.use_nlm = tk.BooleanVar(value=False)

            # Modo "Suma seriada de filtros"
            self.use_serial_filters = tk.BooleanVar(value=False)
            self.serial_filter_order = []  # lista de tags en orden (p.ej. ["median","gaussian","nlm"])
            self._serial_filter_defs = []  # se inicializa en _build_ui

            # Calibración Darks / Flats / Biases
            self.master_dark = None
            self.master_flat = None
            self.master_bias = None

            self.progress_var = tk.DoubleVar(value=0.0)

            # Indicador de procesamiento (GIF + cronómetro)
            self._processing_indicator_on = False
            self._processing_t0 = None
            self._processing_timer_job = None

            # Animación GIF (luna)
            self._luna_frames = []
            self._luna_durations = []
            self._luna_frame_index = 0
            self._luna_animating = False
            self._luna_anim_job = None
            # referencias para previews
            self.src_imgtk = None
            self.res_imgtk = None

            # Cola thread-safe para actualizaciones de UI (evita cuelgues por Tk desde hilos)
            self._ui_queue = queue.Queue()
            self.after(50, self._drain_ui_queue)

            # Cancelación cooperativa de procesos (botón "Detener proceso")
            self.cancel_event = threading.Event()

            self._build_ui()

            # Mostrar ya con todo armado (evita “recuadros” al abrir)
            def _show_main():
                # Abrir maximizado (Windows). En otros sistemas hace best-effort.
                try:
                    self.state("zoomed")
                except Exception:
                    try:
                        self.attributes("-zoomed", True)
                    except Exception:
                        pass

                try:
                    self.update_idletasks()
                except Exception:
                    pass

                self.deiconify()
                try:
                    self.lift()
                except Exception:
                    pass
                try:
                    self.focus_force()
                except Exception:
                    pass

            self.after(0, _show_main)

        def _close_splash_and_start():
            try:
                if self._splash_tick_job is not None:
                    try:
                        self.after_cancel(self._splash_tick_job)
                    except Exception:
                        pass
            finally:
                self._splash_tick_job = None

            try:
                if self._splash_close_job is not None:
                    try:
                        self.after_cancel(self._splash_close_job)
                    except Exception:
                        pass
            finally:
                self._splash_close_job = None

            try:
                self._splash_progress_var.set(100.0)
            except Exception:
                pass

            try:
                if self._splash_win is not None:
                    self._splash_win.destroy()
            except Exception:
                pass

            self._splash_win = None
            self._splash_imgtk = None
            self._splash_t0 = None

            _startup()

        def _tick_splash():
            if self._splash_win is None:
                return
            if self._splash_t0 is None:
                self._splash_t0 = time.perf_counter()

            elapsed_ms = (time.perf_counter() - self._splash_t0) * 1000.0
            pct = (elapsed_ms / float(self._splash_duration_ms)) * 100.0
            if pct < 0.0:
                pct = 0.0
            if pct > 100.0:
                pct = 100.0

            try:
                self._splash_progress_var.set(pct)
            except Exception:
                pass

            if pct >= 100.0:
                return

            try:
                self._splash_tick_job = self.after(self._splash_tick_ms, _tick_splash)
            except Exception:
                self._splash_tick_job = None

        def _start_splash():
            if not os.path.exists(self._intro_path):
                self.after(0, _startup)
                return

            try:
                img = Image.open(self._intro_path)
            except Exception:
                self.after(0, _startup)
                return

            # Redimensionar si es demasiado grande para la pantalla (mantiene aspecto)
            try:
                sw = int(self.winfo_screenwidth())
                sh = int(self.winfo_screenheight())
            except Exception:
                sw, sh = 1280, 720

            try:
                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    try:
                        resample = Image.LANCZOS
                    except Exception:
                        resample = Image.ANTIALIAS
            except Exception:
                resample = None

            iw, ih = img.size
            max_w = int(sw * 0.85)
            max_h = int(sh * 0.80)
            if iw > max_w or ih > max_h:
                scale = min(max_w / float(iw), max_h / float(ih))
                nw = max(1, int(iw * scale))
                nh = max(1, int(ih * scale))
                try:
                    img = img.resize((nw, nh), resample=resample) if resample is not None else img.resize((nw, nh))
                except Exception:
                    try:
                        img = img.resize((nw, nh))
                    except Exception:
                        pass

            self._splash_win = tk.Toplevel(self)
            try:
                self._splash_win.overrideredirect(True)
            except Exception:
                pass
            try:
                self._splash_win.attributes("-topmost", True)
            except Exception:
                pass

            # Contenedor
            holder = tk.Frame(self._splash_win, bg="#000000")
            holder.pack(fill=tk.BOTH, expand=True)

            try:
                self._splash_imgtk = ImageTk.PhotoImage(img)
            except Exception:
                try:
                    self._splash_win.destroy()
                except Exception:
                    pass
                self._splash_win = None
                self.after(0, _startup)
                return

            lbl = tk.Label(holder, image=self._splash_imgtk, bd=0)
            lbl.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Barra de carga falsa abajo
            bar_frame = tk.Frame(holder, bg="#000000")
            bar_frame.pack(side=tk.BOTTOM, fill=tk.X)

            try:
                style = ttk.Style(self._splash_win)
                try:
                    if "clam" in style.theme_names():
                        style.theme_use("clam")
                except Exception:
                    pass
            except Exception:
                style = None

            pb = ttk.Progressbar(
                bar_frame,
                orient="horizontal",
                mode="determinate",
                maximum=100.0,
                variable=self._splash_progress_var,
            )
            pb.pack(side=tk.BOTTOM, fill=tk.X)

            # Centrar splash
            try:
                self._splash_win.update_idletasks()
                w = self._splash_win.winfo_width()
                h = self._splash_win.winfo_height()
                x = int((sw - w) / 2)
                y = int((sh - h) / 2)
                self._splash_win.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                pass

            # Arrancar progreso + cierre en 3s
            self._splash_t0 = time.perf_counter()
            self._splash_progress_var.set(0.0)
            _tick_splash()
            self._splash_close_job = self.after(self._splash_duration_ms, _close_splash_and_start)

        # Iniciar splash (o saltar directo si no existe Intro.jpg)
        self.after(0, _start_splash)

    def _build_ui(self):
        UI_BG = "#535353"
        BTN_BG = "#535353"
        BTN_FG = "#f0f0f0"
        TXT_FG = "#f0f0f0"
        CANVAS_BG = "#333333"

        # Colores guardados para poder cambiar aspecto al habilitar/deshabilitar
        self.UI_BG = UI_BG
        self.TXT_FG = TXT_FG

        # Botones
        self.BTN_BG_ENABLED = BTN_BG
        self.BTN_FG_ENABLED = BTN_FG
        self.BTN_BG_DISABLED = "#3a3a3a"
        self.BTN_FG_DISABLED = "#888888"

        # Sliders
        self.SLIDER_BG_ENABLED = UI_BG
        self.SLIDER_FG_ENABLED = TXT_FG
        self.SLIDER_TROUGH_ENABLED = "#3d3d3d"
        self.SLIDER_BG_DISABLED = "#3a3a3a"
        self.SLIDER_FG_DISABLED = "#888888"
        self.SLIDER_TROUGH_DISABLED = "#222222"

        self.configure(bg=UI_BG)

        top = tk.Frame(self, bg=UI_BG)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)


        # Botones
        # ---------- helper tooltip ----------
        class ToolTip:
            def __init__(self, widget, text):
                self.widget = widget
                self.text = text
                self.tip = None
                widget.bind("<Enter>", self.show)
                widget.bind("<Leave>", self.hide)

            def show(self, event=None):
                if self.tip:
                    return
                x = self.widget.winfo_rootx() + 20
                y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
                self.tip = tk.Toplevel(self.widget)
                self.tip.wm_overrideredirect(True)
                self.tip.wm_geometry(f"+{x}+{y}")
                tk.Label(
                    self.tip,
                    text=self.text,
                    bg="#222222",
                    fg="#f0f0f0",
                    relief="solid",
                    bd=1,
                    font=("", 9),
                    padx=6,
                    pady=3
                ).pack()

            def hide(self, event=None):
                if self.tip:
                    self.tip.destroy()
                    self.tip = None

        # Tooltip SOLO para widgets deshabilitados (sliders)
        class DisabledToolTip:
            def __init__(self, widget, text, state_widget=None):
                self.widget = widget
                self.text = text
                self.state_widget = state_widget if state_widget is not None else widget
                self.tip = None
                widget.bind("<Enter>", self.show)
                widget.bind("<Leave>", self.hide)

            def show(self, event=None):
                if self.tip:
                    return
                try:
                    if str(self.state_widget.cget("state")) != "disabled":
                        return
                except Exception:
                    return

                x = self.widget.winfo_rootx() + 20
                y = self.widget.winfo_rooty() - 30
                if y < 0:
                    y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

                self.tip = tk.Toplevel(self.widget)
                self.tip.wm_overrideredirect(True)
                self.tip.wm_geometry(f"+{x}+{y}")
                tk.Label(
                    self.tip,
                    text=self.text,
                    bg="#222222",
                    fg="#f0f0f0",
                    relief="solid",
                    bd=1,
                    font=("", 9),
                    padx=6,
                    pady=3
                ).pack()

            def hide(self, event=None):
                if self.tip:
                    self.tip.destroy()
                    self.tip = None

        # ---------- botones con iconos ----------
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")

        self.icon_add_images = ImageTk.PhotoImage(
            Image.open(os.path.join(icons_dir, "agregar_imagenes.png")).resize((55, 55), Image.LANCZOS) #tamaño de iconos
        )
        self.icon_add_folders = ImageTk.PhotoImage(
            Image.open(os.path.join(icons_dir, "agregar_carpetas.png")).resize((55, 55), Image.LANCZOS)
        )

        btns_frame = tk.Frame(top, bg=UI_BG)
        btns_frame.pack(side=tk.LEFT, padx=4)

        # --- Agregar imágenes ---
        frame_img = tk.Frame(btns_frame, bg=UI_BG)
        frame_img.pack(side=tk.LEFT, padx=6)

        self.btn_add = tk.Button(
            frame_img,
            image=self.icon_add_images,
            command=self.add_images,
            bg=BTN_BG,
            activebackground="#6b6b6b",
            bd=0
        )
        self.btn_add.pack()
        tk.Label(frame_img, text="Agregar imágenes", bg=UI_BG, fg=TXT_FG, font=("", 9)).pack()
        ToolTip(self.btn_add, "Agregar serie de imágenes al stack")

        # --- Agregar carpetas ---
        frame_fold = tk.Frame(btns_frame, bg=UI_BG)
        frame_fold.pack(side=tk.LEFT, padx=6)

        self.btn_add_folders = tk.Button(
            frame_fold,
            image=self.icon_add_folders,
            command=self.add_folders,
            bg=BTN_BG,
            activebackground="#6b6b6b",
            bd=0
        )
        self.btn_add_folders.pack()
        tk.Label(frame_fold, text="Agregar carpetas", bg=UI_BG, fg=TXT_FG, font=("", 9)).pack()
        ToolTip(self.btn_add_folders, "Agregar carpetas para procesamiento por lotes")

        # =============================
        # Grupos de métodos / filtros
        # =============================
        groups_frame = tk.Frame(top, bg=UI_BG)
        groups_frame.pack(side=tk.LEFT, padx=8)

        # --- 1) Apilamiento con focus tracking (métodos actuales) ---
        # --- Label personalizado (texto + ícono) ---
        label_frame = tk.Frame(groups_frame, bg=UI_BG)

        tk.Label(
            label_frame,
            text="Apilamiento con focus tracking",
            bg=UI_BG,
            fg=TXT_FG,
            font=("", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 6))

        bee_icon_path = os.path.join(icons_dir, "abeja.png")
        bee_img = Image.open(bee_icon_path).resize((40, 40), Image.LANCZOS)
        self.icon_bee = ImageTk.PhotoImage(bee_img)

        tk.Label(
            label_frame,
            image=self.icon_bee,
            bg=UI_BG,
            bd=0
        ).pack(side=tk.RIGHT)

        lf_focus = tk.LabelFrame(
            groups_frame,
            labelwidget=label_frame,
            bg=UI_BG,
            fg=TXT_FG,
            bd=2,
            relief="groove",
            padx=8,
            pady=4,
        )
        lf_focus.pack(side=tk.LEFT, padx=(0, 8), fill=tk.Y)

        # --- Métodos de apilamiento (focus tracking) ---
        self.chk_pmax = tk.Checkbutton(
            lf_focus,
            text="Piramide Max + Post",
            variable=self.use_pmax,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_focus_method_toggle,
        )
        self.chk_pmax.pack(fill=tk.X, anchor="w")

        self.chk_weighted = tk.Checkbutton(
            lf_focus,
            text="Promedio ponderado",
            variable=self.use_weighted,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_focus_method_toggle,
        )
        self.chk_weighted.pack(fill=tk.X, anchor="w")

        self.chk_depth = tk.Checkbutton(
            lf_focus,
            text="Mapa de profundidad",
            variable=self.use_depth,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_focus_method_toggle,
        )
        self.chk_depth.pack(fill=tk.X, anchor="w")

        # --- 2) Apilado por integración (placeholder para futuros métodos) ---

        label_integracion = tk.Frame(groups_frame, bg=UI_BG)

        tk.Label(
            label_integracion,
            text="Apilado por integración",
            bg=UI_BG,
            fg=TXT_FG,
            font=("", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 6))

        galaxy_icon_path = os.path.join(icons_dir, "galaxia.png")
        galaxy_img = Image.open(galaxy_icon_path).resize((40, 40), Image.LANCZOS)
        self.icon_galaxy = ImageTk.PhotoImage(galaxy_img)

        tk.Label(
            label_integracion,
            image=self.icon_galaxy,
            bg=UI_BG,
            bd=0
        ).pack(side=tk.RIGHT)

        lf_integracion = tk.LabelFrame(
            groups_frame,
            labelwidget=label_integracion,
            bg=UI_BG,
            fg=TXT_FG,
            bd=2,
            relief="groove",
            padx=8,
            pady=4,
        )
        lf_integracion.pack(side=tk.LEFT, padx=(0, 8), fill=tk.Y)

        # Métodos de integración (mutuamente excluyentes con Focus Tracking y Filtros de ruido)
        self.chk_int_mean = tk.Checkbutton(
            lf_integracion,
            text="Promedio (mean stacking)",
            variable=self.use_int_mean,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_integration_method_toggle,
        )
        self.chk_int_mean.pack(fill=tk.X, anchor="w")

        self.chk_int_median = tk.Checkbutton(
            lf_integracion,
            text="Mediana (median stacking)",
            variable=self.use_int_median,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_integration_method_toggle,
        )
        self.chk_int_median.pack(fill=tk.X, anchor="w")

        self.chk_int_sum = tk.Checkbutton(
            lf_integracion,
            text="Suma (sum stacking)",
            variable=self.use_int_sum,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_integration_method_toggle,
        )
        self.chk_int_sum.pack(fill=tk.X, anchor="w")

        self.chk_int_sigma = tk.Checkbutton(
            lf_integracion,
            text="Sigma-clipping",
            variable=self.use_int_sigma,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=self._on_integration_method_toggle,
        )
        self.chk_int_sigma.pack(fill=tk.X, anchor="w")

        # --- 3) Filtros de ruido (check de Darks/Flats/Biases) ---
        lf_noise = tk.LabelFrame(
            groups_frame,
            text="Filtros de ruido",
            bg=UI_BG,
            fg=TXT_FG,
            bd=2,
            relief="groove",
            labelanchor="nw",
            padx=8,
            pady=4,
        )
        lf_noise.pack(side=tk.LEFT, fill=tk.Y)

        # Layout compacto en 2 columnas (todo en el MISMO grid para evitar huecos):
        # col 0: Masters + (BM3D, Bilateral, Wavelet, NLM)
        # col 1: Fourier + Noise2Noise/Noise2Void
        noise_grid = tk.Frame(lf_noise, bg=UI_BG)
        noise_grid.pack(fill=tk.X, anchor="w")

        # Columnas parejas (alineación estable)
        noise_grid.columnconfigure(0, weight=1, uniform="noise_col")
        noise_grid.columnconfigure(1, weight=1, uniform="noise_col")

        # --- fila 0 ---
        self.chk_calibration = tk.Checkbutton(
            noise_grid,
            text="Masters de calibración",
            variable=self.use_calibration,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="calibrado": self._on_filter_checkbox_toggled(t),
        )
        self.chk_calibration.grid(row=0, column=0, sticky="w", pady=1)

        self.chk_fourier = tk.Checkbutton(
            noise_grid,
            text="Filtrado en Fourier",
            variable=self.use_fourier,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="fourier": self._on_filter_checkbox_toggled(t),
        )
        self.chk_fourier.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=1)

        # --- fila 1 (sube BM3D y alinea N2V a la derecha) ---
        self.chk_bm3d = tk.Checkbutton(
            noise_grid,
            text="BM3D (Block-Matching 3D)",
            variable=self.use_bm3d,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="bm3d": self._on_filter_checkbox_toggled(t),
        )
        self.chk_bm3d.grid(row=1, column=0, sticky="w", pady=1)

        self.chk_n2v = tk.Checkbutton(
            noise_grid,
            text="Noise2Noise / Noise2Void",
            variable=self.use_n2v,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="noise2void": self._on_filter_checkbox_toggled(t),
        )
        self.chk_n2v.grid(row=1, column=1, sticky="w", padx=(12, 0), pady=1)

        # --- derecha debajo de N2V ---
        self.chk_median = tk.Checkbutton(
            noise_grid,
            text="Filtro de Mediana",
            variable=self.use_median,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="median": self._on_filter_checkbox_toggled(t),
        )
        self.chk_median.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=1)

        self.chk_gaussian = tk.Checkbutton(
            noise_grid,
            text="Filtro Gaussiano",
            variable=self.use_gaussian,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="gaussian": self._on_filter_checkbox_toggled(t),
        )
        self.chk_gaussian.grid(row=3, column=1, sticky="w", padx=(12, 0), pady=1)

        # Modo seriado (debajo de Gaussiano)
        self.chk_serial_filters = tk.Checkbutton(
            noise_grid,
            text="Suma seriada de filtros",
            variable=self.use_serial_filters,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            font=("", 9, "bold"),
            command=self._on_serial_filters_toggle,
        )
        self.chk_serial_filters.grid(row=4, column=1, sticky="w", padx=(12, 0), pady=1)

        ToolTip(
            self.chk_serial_filters,
            "Aplica los filtros seleccionados uno tras otro (en el orden numerado),\nsumando sus efectos sobre la imagen. Puede mejorar el resultado,\npero aumenta el tiempo de procesamiento. "
            
            ""
        )

        # --- resto (columna izquierda, sin huecos) ---
        self.chk_bilateral = tk.Checkbutton(
            noise_grid,
            text="Filtro Bilateral",
            variable=self.use_bilateral,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="bilateral": self._on_filter_checkbox_toggled(t),
        )
        self.chk_bilateral.grid(row=2, column=0, sticky="w", pady=1)

        self.chk_wavelet = tk.Checkbutton(
            noise_grid,
            text="Wavelet Denoising (suave)",
            variable=self.use_wavelet,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="wavelet": self._on_filter_checkbox_toggled(t),
        )
        self.chk_wavelet.grid(row=3, column=0, sticky="w", pady=1)

        self.chk_nlm = tk.Checkbutton(
            noise_grid,
            text="Non-Local Means (OpenCV)",
            variable=self.use_nlm,
            bg=UI_BG,
            fg=TXT_FG,
            selectcolor=UI_BG,
            activebackground=UI_BG,
            activeforeground=TXT_FG,
            anchor="w",
            justify="left",
            command=lambda t="nlm": self._on_filter_checkbox_toggled(t),
        )
        self.chk_nlm.grid(row=4, column=0, sticky="w", pady=1)

        # Definiciones para numeración/orden del modo seriado (tag, var, chk, texto_base)
        self._serial_filter_defs = [
            ("calibrado", self.use_calibration, self.chk_calibration, "Masters de calibración"),
            ("fourier", self.use_fourier, self.chk_fourier, "Filtrado en Fourier"),
            ("bm3d", self.use_bm3d, self.chk_bm3d, "BM3D (Block-Matching 3D)"),
            ("noise2void", self.use_n2v, self.chk_n2v, "Noise2Noise / Noise2Void"),
            ("median", self.use_median, self.chk_median, "Filtro de Mediana"),
            ("gaussian", self.use_gaussian, self.chk_gaussian, "Filtro Gaussiano"),
            ("bilateral", self.use_bilateral, self.chk_bilateral, "Filtro Bilateral"),
            ("wavelet", self.use_wavelet, self.chk_wavelet, "Wavelet Denoising (suave)"),
            ("nlm", self.use_nlm, self.chk_nlm, "Non-Local Means (OpenCV)"),
        ]

        # Aplicar bloqueo inicial entre grupos
        self._update_focus_noise_exclusivity()
        self._refresh_serial_numbers()

        # Botón único para apilar (imágenes o carpetas)
        # --- Botón Procesar imágenes con ícono ---
        self.icon_process = ImageTk.PhotoImage(
            Image.open(os.path.join(icons_dir, "procesar.png"))
            .resize((70, 70), Image.LANCZOS)# estaba en 70x70
        )

        frame_process = tk.Frame(top, bg=UI_BG)
        frame_process.pack(side=tk.LEFT, padx=6)

        self.btn_process = tk.Button(
            frame_process,
            image=self.icon_process,
            command=self.apilar_imagenes,
            bg=BTN_BG,
            activebackground="#6b6b6b",
            bd=0
        )
        self.btn_process.pack()

        tk.Label(
            frame_process,
            text="Procesar imágenes",
            bg=UI_BG,
            fg=TXT_FG,
            font=("", 9)
        ).pack()

        ToolTip(
            self.btn_process,
            "Inicia el procesamiento y apilado de imágenes."

        )

        # Progressbar (ttk requiere estilo)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", background="#a0a0a0", troughcolor="#3d3d3d")

        # --- Barra de progreso con texto superpuesto ---

        progress_container = tk.Frame(top, bg=UI_BG)
        progress_container.pack(side=tk.LEFT, padx=(12, 6), fill=tk.X)

        # Texto de estado ARRIBA de la barra (alineado a la izquierda)
        self.status = tk.StringVar(value="Sin imágenes cargadas.")
        tk.Label(
            progress_container,
            textvariable=self.status,
            bg=UI_BG,
            fg=TXT_FG,
            font=("", 9, "bold"),
            anchor="w"
        ).pack(fill=tk.X, padx=2)

        # Barra de progreso
        self.progress = ttk.Progressbar(
            progress_container,
            variable=self.progress_var,
            orient="horizontal",
            length=260,
            mode="determinate",
            maximum=100.0
        )
        self.progress.pack(fill=tk.X)

        # Indicador: GIF (estático) + cronómetro (solo mientras procesa)
        self.processing_info_frame = tk.Frame(progress_container, bg=UI_BG)

        # Cargar frames del GIF (30x30). Queda estático por defecto; se anima solo al procesar.
        self._load_luna_gif(os.path.join(icons_dir, "luna.gif"), size=(30, 30))
        luna_img0 = self._luna_frames[0] if getattr(self, "_luna_frames", None) else None
        self.luna_label = tk.Label(self.processing_info_frame, image=luna_img0, bg=UI_BG, bd=0)
        self.luna_label.pack(side=tk.LEFT)

        self.timer_var = tk.StringVar(value="Tiempo de procesamiento: 00:00")
        self.timer_label = tk.Label(
            self.processing_info_frame,
            textvariable=self.timer_var,
            bg=UI_BG,
            fg=TXT_FG,
            font=("", 10, "bold")
        )
        self.timer_label.pack(side=tk.LEFT, padx=(6, 0))

        # Visible (se oculta cuando no está procesando)
        self.processing_info_frame.pack(anchor="w", pady=(4, 0))

        # Botones: detener proceso + about (en la misma fila)
        self.progress_buttons_row = tk.Frame(progress_container, bg=UI_BG)
        self.progress_buttons_row.pack(anchor="w", pady=(6, 0))

        # Botón: detener proceso (se habilita solo mientras se procesa)
        self.btn_stop = tk.Button(
            self.progress_buttons_row,
            text="Detener proceso",
            command=self.stop_all_processes,
            bg=self.BTN_BG_DISABLED,
            fg=self.BTN_FG_DISABLED,
            activebackground="#6b6b6b",
            activeforeground=self.BTN_FG_ENABLED,
            relief="raised",
            bd=2,
            padx=10,
            pady=2,
            state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT)

        # Botón: About Open Galileo
        self.btn_about = tk.Button(
            self.progress_buttons_row,
            text="About Open Galileo",
            command=self.show_about,
            bg=self.BTN_BG_ENABLED,
            fg=self.BTN_FG_ENABLED,
            activebackground="#6b6b6b",
            activeforeground=self.BTN_FG_ENABLED,
            relief="raised",
            bd=2,
            padx=10,
            pady=2,
            state=tk.NORMAL
        )
        self.btn_about.pack(side=tk.LEFT, padx=(8, 0))

        # Porcentaje (opcional, se mantiene)

        self.progress_label = tk.StringVar(value="0%")
        tk.Label(
            top,
            textvariable=self.progress_label,
            width=5,
            anchor="w",
            bg=UI_BG,
            fg=TXT_FG
        ).pack(side=tk.LEFT)

        # Panel principal

        main = tk.Frame(self, bg=UI_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Mantener el panel izquierdo angosto (NO se ensancha al agregar scroll)
        LEFT_PANEL_WIDTH = 260
        main.columnconfigure(0, weight=0, minsize=LEFT_PANEL_WIDTH)  # Panel izquierdo (fijo)
        main.columnconfigure(1, weight=1)  # Área de trabajo (se lleva el resto)

        # Fila 0: área de trabajo; Fila 1: espacio libre futuro
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        # =============================
        #  COLUMNA IZQUIERDA COMPLETA
        # =============================
        left = tk.Frame(main, bg=UI_BG, width=LEFT_PANEL_WIDTH)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        left.pack_propagate(False)

        tk.Label(left, text="Imágenes / Carpetas seleccionadas:",
                 bg=UI_BG, fg=TXT_FG).pack(anchor="w")

        self.listbox = tk.Listbox(left, bg="#4a4a4a", fg=TXT_FG,
                                  selectbackground="#6b6b6b", highlightbackground="#222222")
        # La listbox sigue con expand=True, pero dejará espacio para el texto de ruta debajo
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=(4, 4))

        # Texto de ruta (abajo de la lista)
        self.lbl_autosave_path = tk.Label(
            left,
            textvariable=self.auto_save_label_var,
            bg=UI_BG,
            fg=TXT_FG,
            anchor="w",
            justify="left",
            font=("", 9),
            wraplength=260,
        )
        self.lbl_autosave_path.pack(fill=tk.X, pady=(0, 6))
        self.lbl_autosave_path.bind(
            "<Configure>",
            lambda e: self.lbl_autosave_path.config(wraplength=max(120, e.width)),
        )

        # Fila: Ruta de guardado + Formato de salida
        save_row = tk.Frame(left, bg=UI_BG)
        save_row.pack(fill=tk.X, pady=(0, 4))

        self.btn_save = tk.Button(
            save_row,
            text="Ruta de guardado",
            command=self.save_result,
            width=14,
            bg=BTN_BG,
            fg=BTN_FG,
            state=tk.NORMAL,
            activebackground="#6b6b6b",
            activeforeground=BTN_FG
        )
        self.btn_save.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.btn_save, "Seleccione ruta para el guardado automático de archivos")

        # Botón desplegable (formato) - estilo botón
        def _update_format_button(*_):
            fmt = (self.output_format_var.get() or "").lower()
            if fmt not in ("jpg", "png", "tiff"):
                fmt = "jpg"
                self.output_format_var.set(fmt)
            self.btn_format.config(text=f"Formato: {fmt.upper()} ▼")

        def _set_format(fmt):
            self.output_format_var.set(fmt)
            _update_format_button()

        def _show_format_menu():
            try:
                x = self.btn_format.winfo_rootx()
                y = self.btn_format.winfo_rooty() + self.btn_format.winfo_height()
                self.format_menu.tk_popup(x, y)
            finally:
                try:
                    self.format_menu.grab_release()
                except Exception:
                    pass

        self.btn_format = tk.Button(
            save_row,
            text="Formato: JPG ▼",
            command=_show_format_menu,
            width=12,
            bg=BTN_BG,
            fg=BTN_FG,
            state=tk.NORMAL,
            activebackground="#6b6b6b",
            activeforeground=BTN_FG,
            relief="raised",
            bd=1,
            highlightthickness=0,
            cursor="hand2",
        )
        self.btn_format.pack(side=tk.LEFT)

        self.format_menu = tk.Menu(
            self.btn_format,
            tearoff=0,
            bg=BTN_BG,
            fg=BTN_FG,
            activebackground="#6b6b6b",
            activeforeground=BTN_FG,
        )
        for _fmt in ("jpg", "png", "tiff"):
            self.format_menu.add_command(label=_fmt.upper(), command=lambda f=_fmt: _set_format(f))

        try:
            self.output_format_var.trace_add("write", _update_format_button)
        except Exception:
            try:
                self.output_format_var.trace("w", lambda *_: _update_format_button())
            except Exception:
                pass
        _update_format_button()

        ToolTip(self.btn_format, "Elegir formato de salida")

        # ---------------- Sliders (agrupados por familia) ----------------
        # Pestañas internas: Focus tracking / Integración
        knobs_lf = tk.LabelFrame(
            left,
            text="Parámetros y métodos",
            bg=UI_BG,
            fg=TXT_FG
        )
        knobs_lf.pack(fill=tk.X, pady=(0, 4))

        # Notebook (pestañas)
        try:
            _knobs_style = ttk.Style()
            try:
                if "clam" in _knobs_style.theme_names():
                    _knobs_style.theme_use("clam")
            except Exception:
                pass

            _knobs_style.configure("Knobs.TNotebook", background=UI_BG, borderwidth=0)
            _knobs_style.configure(
                "Knobs.TNotebook.Tab",
                background="#3a3a3a",
                foreground=TXT_FG,
                padding=(10, 4),
            )
            _knobs_style.map(
                "Knobs.TNotebook.Tab",
                background=[("selected", UI_BG), ("active", "#6b6b6b")],
                foreground=[("selected", TXT_FG), ("active", TXT_FG)],
                padding=[("selected", (10, 4)), ("!selected", (6, 4))],
            )
        except Exception:
            pass

        knobs_nb = ttk.Notebook(knobs_lf, style="Knobs.TNotebook")
        self.knobs_nb = knobs_nb
        knobs_nb.pack(fill=tk.BOTH, expand=True)

        tab_focus = tk.Frame(knobs_nb, bg=UI_BG)
        tab_integ = tk.Frame(knobs_nb, bg=UI_BG)
        tab_noise = tk.Frame(knobs_nb, bg=UI_BG)

        self.tab_focus = tab_focus
        self.tab_integ = tab_integ
        self.tab_noise = tab_noise

        # ---- título completo SOLO para la pestaña seleccionada ----
        self._knobs_tab_titles_full = {
            tab_focus: "Focus tracking",
            tab_integ: "Integración",
            tab_noise: "Filtros de ruido",
        }
        self._knobs_tab_titles_short = {
            tab_focus: "Focus",
            tab_integ: "Integ",
            tab_noise: "Ruido",
        }

        knobs_nb.add(tab_focus, text=self._knobs_tab_titles_full[tab_focus])
        knobs_nb.add(tab_integ, text=self._knobs_tab_titles_full[tab_integ])
        knobs_nb.add(tab_noise, text=self._knobs_tab_titles_full[tab_noise])

        def _update_knobs_tab_titles(event=None):
            nb = getattr(self, "knobs_nb", None)
            if nb is None:
                return

            try:
                selected_id = nb.select()
            except Exception:
                selected_id = None

            selected_widget = None
            if selected_id:
                try:
                    selected_widget = nb.nametowidget(selected_id)
                except Exception:
                    selected_widget = None

            full = getattr(self, "_knobs_tab_titles_full", {}) or {}
            short = getattr(self, "_knobs_tab_titles_short", {}) or {}

            for t in (tab_focus, tab_integ, tab_noise):
                try:
                    if selected_widget is not None and selected_widget == t:
                        nb.tab(t, text=full.get(t, ""))
                    else:
                        nb.tab(t, text=short.get(t, full.get(t, "")))
                except Exception:
                    pass

        try:
            knobs_nb.bind("<<NotebookTabChanged>>", _update_knobs_tab_titles)
            knobs_nb.bind("<Configure>", _update_knobs_tab_titles)
        except Exception:
            pass

        try:
            self.after(10, _update_knobs_tab_titles)
        except Exception:
            try:
                _update_knobs_tab_titles()
            except Exception:
                pass

        def _make_scrollable_tab(parent):
            c = tk.Canvas(
                parent,
                bg=UI_BG,
                highlightthickness=0,
                bd=0,
                height=320,
            )
            vsb = tk.Scrollbar(
                parent,
                orient="vertical",
                command=c.yview,
                width=10,
            )
            c.configure(yscrollcommand=vsb.set)

            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            inner = tk.Frame(c, bg=UI_BG)
            win_id = c.create_window((0, 0), window=inner, anchor="nw")

            def _update_scrollregion(event=None, _c=c):
                try:
                    _c.configure(scrollregion=_c.bbox("all"))
                except Exception:
                    pass

            def _fit_width(event, _c=c, _win_id=win_id):
                try:
                    _c.itemconfigure(_win_id, width=event.width)
                except Exception:
                    pass

            inner.bind("<Configure>", _update_scrollregion)
            c.bind("<Configure>", _fit_width)

            def _on_mousewheel(event, _c=c):
                try:
                    delta = int(-1 * (event.delta / 120))
                except Exception:
                    delta = -1 if getattr(event, "delta", 0) > 0 else 1
                _c.yview_scroll(delta, "units")
                return "break"

            def _on_mousewheel_linux_up(event, _c=c):
                _c.yview_scroll(-1, "units")
                return "break"

            def _on_mousewheel_linux_down(event, _c=c):
                _c.yview_scroll(1, "units")
                return "break"

            def _bind_wheel(event, _c=c):
                _c.bind_all("<MouseWheel>", _on_mousewheel)
                _c.bind_all("<Button-4>", _on_mousewheel_linux_up)
                _c.bind_all("<Button-5>", _on_mousewheel_linux_down)

            def _unbind_wheel(event, _c=c):
                _c.unbind_all("<MouseWheel>")
                _c.unbind_all("<Button-4>")
                _c.unbind_all("<Button-5>")

            c.bind("<Enter>", _bind_wheel)
            c.bind("<Leave>", _unbind_wheel)

            return inner

        focus_inner = _make_scrollable_tab(tab_focus)
        integ_inner = _make_scrollable_tab(tab_integ)
        noise_inner = _make_scrollable_tab(tab_noise)

        focus_frame = tk.Frame(focus_inner, bg=UI_BG)
        focus_frame.pack(fill=tk.X)

        integ_frame = tk.Frame(integ_inner, bg=UI_BG)
        integ_frame.pack(fill=tk.X)

        noise_frame = tk.Frame(noise_inner, bg=UI_BG)
        noise_frame.pack(fill=tk.X)

        def _mk_slider_row(parent, title, var, cmd, from_=1, to=10, resolution=1, value_width=3, value_fmt=None):
            row = tk.Frame(parent, bg=UI_BG)
            row.pack(fill=tk.X, padx=6, pady=(4, 6))

            header = tk.Frame(row, bg=UI_BG)
            header.pack(fill=tk.X)

            tk.Label(
                header,
                text=title,
                bg=UI_BG,
                fg=TXT_FG,
                anchor="w"
            ).pack(side=tk.LEFT, anchor="w")

            def _fmt(v):
                if value_fmt is not None:
                    try:
                        return value_fmt(v)
                    except Exception:
                        pass
                try:
                    if isinstance(v, float):
                        return f"{v:.1f}"
                except Exception:
                    pass
                try:
                    return str(int(v))
                except Exception:
                    return str(v)

            value_lbl = tk.Label(
                header,
                text=_fmt(var.get()),
                bg=self.SLIDER_BG_ENABLED,
                fg=self.SLIDER_FG_ENABLED,
                width=value_width,
                anchor="e",
            )
            value_lbl.pack(side=tk.RIGHT)

            def _on_change(val):
                # val llega como string
                try:
                    vv = float(val)
                except Exception:
                    vv = None

                try:
                    value_lbl.config(text=_fmt(vv if vv is not None else var.get()))
                except Exception:
                    pass

                try:
                    cmd(val)
                except Exception:
                    try:
                        cmd()
                    except Exception:
                        pass

            s = tk.Scale(
                row,
                from_=from_,
                to=to,
                orient=tk.HORIZONTAL,
                showvalue=False,
                resolution=resolution,
                variable=var,
                command=_on_change,
                bg=self.SLIDER_BG_ENABLED,
                fg=self.SLIDER_FG_ENABLED,
                troughcolor=self.SLIDER_TROUGH_ENABLED,
                highlightthickness=0,
                bd=0,
                relief="flat",
                sliderlength=18,
                length=220,
            )
            s.pack(fill=tk.X, pady=(2, 0))

            def _sync_label(*_):
                try:
                    value_lbl.config(text=_fmt(var.get()))
                except Exception:
                    pass

            try:
                var.trace_add("write", _sync_label)
            except Exception:
                pass

            return s, value_lbl

        # ===== TAB: Focus tracking =====
        # ===== PMax =====
        pmax_lf = tk.LabelFrame(
            focus_frame,
            text="Pirámide Max (PMax)",
            bg=UI_BG,
            fg=TXT_FG
        )
        pmax_lf.pack(fill=tk.X, padx=4, pady=(4, 6))

        self.radius_pmax_var = tk.IntVar(value=RADIUS_PMAX)
        self.smooth_pmax_var = tk.IntVar(value=SMOOTH_PMAX)

        self.slider_radius_pmax, self.slider_radius_pmax_val_lbl = _mk_slider_row(
            pmax_lf,
            "Radio (detalle)",
            self.radius_pmax_var,
            self._on_radius_pmax_change,
            from_=1,
            to=10,
            resolution=1,
            value_width=3,
        )
        self.slider_smooth_pmax, self.slider_smooth_pmax_val_lbl = _mk_slider_row(
            pmax_lf,
            "Suavizado",
            self.smooth_pmax_var,
            self._on_smooth_pmax_change,
            from_=1,
            to=10,
            resolution=1,
            value_width=3,
        )

        # ===== Promedio ponderado =====
        weighted_lf = tk.LabelFrame(
            focus_frame,
            text="Promedio ponderado",
            bg=UI_BG,
            fg=TXT_FG
        )
        weighted_lf.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.radius_weighted_var = tk.IntVar(value=RADIUS_WEIGHTED)
        self.smooth_weighted_var = tk.IntVar(value=SMOOTH_WEIGHTED)

        self.slider_radius_weighted, self.slider_radius_weighted_val_lbl = _mk_slider_row(
            weighted_lf,
            "Radio",
            self.radius_weighted_var,
            self._on_radius_weighted_change,
            from_=1,
            to=10,
            resolution=1,
            value_width=3,
        )
        self.slider_smooth_weighted, self.slider_smooth_weighted_val_lbl = _mk_slider_row(
            weighted_lf,
            "Suavizado",
            self.smooth_weighted_var,
            self._on_smooth_weighted_change,
            from_=1,
            to=10,
            resolution=1,
            value_width=3,
        )

        # ===== Mapa de profundidad =====
        depth_lf = tk.LabelFrame(
            focus_frame,
            text="Mapa de profundidad",
            bg=UI_BG,
            fg=TXT_FG
        )
        depth_lf.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.radius_depth_var = tk.IntVar(value=RADIUS_DEPTH)
        self.smooth_depth_var = tk.IntVar(value=SMOOTH_DEPTH)

        self.slider_radius_depth, self.slider_radius_depth_val_lbl = _mk_slider_row(
            depth_lf,
            "Radio",
            self.radius_depth_var,
            self._on_radius_depth_change,
            from_=1,
            to=10,
            resolution=1,
            value_width=3,
        )
        self.slider_smooth_depth, self.slider_smooth_depth_val_lbl = _mk_slider_row(
            depth_lf,
            "Suavizado",
            self.smooth_depth_var,
            self._on_smooth_depth_change,
            from_=1,
            to=10,
            resolution=1,
            value_width=3,
        )

        # ===== TAB: Integración =====
        # ---- Promedio (mean stacking) ----
        mean_lf = tk.LabelFrame(
            integ_frame,
            text="Promedio (mean stacking)",
            bg=UI_BG,
            fg=TXT_FG
        )
        mean_lf.pack(fill=tk.X, padx=4, pady=(4, 6))

        self.int_mean_align_max_side_var = tk.IntVar(value=int(INT_PARAMS["mean"]["align_max_side"]))
        self.int_mean_ecc_iters_var = tk.IntVar(value=int(INT_PARAMS["mean"]["ecc_iters"]))
        self.int_mean_ecc_eps_exp_var = tk.IntVar(value=int(INT_PARAMS["mean"]["ecc_eps_exp"]))

        self.slider_int_mean_align_max_side, self.slider_int_mean_align_max_side_val_lbl = _mk_slider_row(
            mean_lf,
            "Máx lado (px)",
            self.int_mean_align_max_side_var,
            lambda v: self._on_int_param_change("mean", "align_max_side", v),
            from_=800,
            to=3000,
            resolution=50,
            value_width=6,
        )
        self.slider_int_mean_ecc_iters, self.slider_int_mean_ecc_iters_val_lbl = _mk_slider_row(
            mean_lf,
            "Iteraciones ECC",
            self.int_mean_ecc_iters_var,
            lambda v: self._on_int_param_change("mean", "ecc_iters", v),
            from_=60,
            to=400,
            resolution=10,
            value_width=4,
        )
        self.slider_int_mean_ecc_eps_exp, self.slider_int_mean_ecc_eps_exp_val_lbl = _mk_slider_row(
            mean_lf,
            "Epsilon (1e-N)",
            self.int_mean_ecc_eps_exp_var,
            lambda v: self._on_int_param_change("mean", "ecc_eps_exp", v),
            from_=4,
            to=8,
            resolution=1,
            value_width=6,
            value_fmt=lambda v: f"1e-{int(float(v))}",
        )

        # ---- Mediana (median stacking) ----
        median_lf = tk.LabelFrame(
            integ_frame,
            text="Mediana (median stacking)",
            bg=UI_BG,
            fg=TXT_FG
        )
        median_lf.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.int_median_align_max_side_var = tk.IntVar(value=int(INT_PARAMS["median"]["align_max_side"]))
        self.int_median_ecc_iters_var = tk.IntVar(value=int(INT_PARAMS["median"]["ecc_iters"]))
        self.int_median_ecc_eps_exp_var = tk.IntVar(value=int(INT_PARAMS["median"]["ecc_eps_exp"]))

        self.slider_int_median_align_max_side, self.slider_int_median_align_max_side_val_lbl = _mk_slider_row(
            median_lf,
            "Máx lado (px)",
            self.int_median_align_max_side_var,
            lambda v: self._on_int_param_change("median", "align_max_side", v),
            from_=800,
            to=3000,
            resolution=50,
            value_width=6,
        )
        self.slider_int_median_ecc_iters, self.slider_int_median_ecc_iters_val_lbl = _mk_slider_row(
            median_lf,
            "Iteraciones ECC",
            self.int_median_ecc_iters_var,
            lambda v: self._on_int_param_change("median", "ecc_iters", v),
            from_=60,
            to=400,
            resolution=10,
            value_width=4,
        )
        self.slider_int_median_ecc_eps_exp, self.slider_int_median_ecc_eps_exp_val_lbl = _mk_slider_row(
            median_lf,
            "Epsilon (1e-N)",
            self.int_median_ecc_eps_exp_var,
            lambda v: self._on_int_param_change("median", "ecc_eps_exp", v),
            from_=4,
            to=8,
            resolution=1,
            value_width=6,
            value_fmt=lambda v: f"1e-{int(float(v))}",
        )

        # ---- Suma (sum stacking) ----
        sum_lf = tk.LabelFrame(
            integ_frame,
            text="Suma (sum stacking)",
            bg=UI_BG,
            fg=TXT_FG
        )
        sum_lf.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.int_sum_align_max_side_var = tk.IntVar(value=int(INT_PARAMS["sum"]["align_max_side"]))
        self.int_sum_ecc_iters_var = tk.IntVar(value=int(INT_PARAMS["sum"]["ecc_iters"]))
        self.int_sum_ecc_eps_exp_var = tk.IntVar(value=int(INT_PARAMS["sum"]["ecc_eps_exp"]))

        self.slider_int_sum_align_max_side, self.slider_int_sum_align_max_side_val_lbl = _mk_slider_row(
            sum_lf,
            "Máx lado (px)",
            self.int_sum_align_max_side_var,
            lambda v: self._on_int_param_change("sum", "align_max_side", v),
            from_=800,
            to=3000,
            resolution=50,
            value_width=6,
        )
        self.slider_int_sum_ecc_iters, self.slider_int_sum_ecc_iters_val_lbl = _mk_slider_row(
            sum_lf,
            "Iteraciones ECC",
            self.int_sum_ecc_iters_var,
            lambda v: self._on_int_param_change("sum", "ecc_iters", v),
            from_=60,
            to=400,
            resolution=10,
            value_width=4,
        )
        self.slider_int_sum_ecc_eps_exp, self.slider_int_sum_ecc_eps_exp_val_lbl = _mk_slider_row(
            sum_lf,
            "Epsilon (1e-N)",
            self.int_sum_ecc_eps_exp_var,
            lambda v: self._on_int_param_change("sum", "ecc_eps_exp", v),
            from_=4,
            to=8,
            resolution=1,
            value_width=6,
            value_fmt=lambda v: f"1e-{int(float(v))}",
        )

        # ---- Sigma-clipping ----
        sigma_lf = tk.LabelFrame(
            integ_frame,
            text="Sigma-clipping",
            bg=UI_BG,
            fg=TXT_FG
        )
        sigma_lf.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.int_sigma_align_max_side_var = tk.IntVar(value=int(INT_PARAMS["sigma"]["align_max_side"]))
        self.int_sigma_ecc_iters_var = tk.IntVar(value=int(INT_PARAMS["sigma"]["ecc_iters"]))
        self.int_sigma_ecc_eps_exp_var = tk.IntVar(value=int(INT_PARAMS["sigma"]["ecc_eps_exp"]))

        self.slider_int_sigma_align_max_side, self.slider_int_sigma_align_max_side_val_lbl = _mk_slider_row(
            sigma_lf,
            "Máx lado (px)",
            self.int_sigma_align_max_side_var,
            lambda v: self._on_int_param_change("sigma", "align_max_side", v),
            from_=800,
            to=3000,
            resolution=50,
            value_width=6,
        )
        self.slider_int_sigma_ecc_iters, self.slider_int_sigma_ecc_iters_val_lbl = _mk_slider_row(
            sigma_lf,
            "Iteraciones ECC",
            self.int_sigma_ecc_iters_var,
            lambda v: self._on_int_param_change("sigma", "ecc_iters", v),
            from_=60,
            to=400,
            resolution=10,
            value_width=4,
        )
        self.slider_int_sigma_ecc_eps_exp, self.slider_int_sigma_ecc_eps_exp_val_lbl = _mk_slider_row(
            sigma_lf,
            "Epsilon (1e-N)",
            self.int_sigma_ecc_eps_exp_var,
            lambda v: self._on_int_param_change("sigma", "ecc_eps_exp", v),
            from_=4,
            to=8,
            resolution=1,
            value_width=6,
            value_fmt=lambda v: f"1e-{int(float(v))}",
        )

        self.int_sigma_k_var = tk.DoubleVar(value=float(INT_PARAMS["sigma"]["sigma_k"]))
        self.slider_int_sigma_k, self.slider_int_sigma_k_val_lbl = _mk_slider_row(
            sigma_lf,
            "K (sigmas)",
            self.int_sigma_k_var,
            lambda v: self._on_int_param_change("sigma", "sigma_k", v),
            from_=1.0,
            to=6.0,
            resolution=0.1,
            value_width=5,
            value_fmt=lambda v: f"{float(v):.1f}",
        )

        # ===== TAB: Filtros de ruido =====
        bm3d_lf = tk.LabelFrame(noise_frame, text="BM3D (Block-Matching 3D)", bg=UI_BG, fg=TXT_FG)
        bm3d_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.bm3d_sigma_var = tk.DoubleVar(value=float(BM3D_SIGMA_PSD))
        self.slider_bm3d_sigma, self.slider_bm3d_sigma_val_lbl = _mk_slider_row(
            bm3d_lf,
            "Sigma (0..1)",
            self.bm3d_sigma_var,
            self._on_bm3d_sigma_change,
            from_=0.01,
            to=0.12,
            resolution=0.01,
            value_width=5,
            value_fmt=lambda v: f"{float(v):.2f}",
        )

        bil_lf = tk.LabelFrame(noise_frame, text="Bilateral", bg=UI_BG, fg=TXT_FG)
        bil_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.bilateral_d_var = tk.IntVar(value=int(BILATERAL_D))
        self.bilateral_sigma_color_var = tk.IntVar(value=int(BILATERAL_SIGMA_COLOR))
        self.bilateral_sigma_space_var = tk.IntVar(value=int(BILATERAL_SIGMA_SPACE))

        self.slider_bilateral_d, self.slider_bilateral_d_val_lbl = _mk_slider_row(
            bil_lf,
            "Diámetro (d)",
            self.bilateral_d_var,
            self._on_bilateral_d_change,
            from_=1,
            to=25,
            resolution=1,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )
        self.slider_bilateral_sigma_color, self.slider_bilateral_sigma_color_val_lbl = _mk_slider_row(
            bil_lf,
            "Sigma color",
            self.bilateral_sigma_color_var,
            self._on_bilateral_sigma_color_change,
            from_=1,
            to=200,
            resolution=1,
            value_width=4,
            value_fmt=lambda v: f"{int(float(v))}",
        )
        self.slider_bilateral_sigma_space, self.slider_bilateral_sigma_space_val_lbl = _mk_slider_row(
            bil_lf,
            "Sigma espacio",
            self.bilateral_sigma_space_var,
            self._on_bilateral_sigma_space_change,
            from_=1,
            to=200,
            resolution=1,
            value_width=4,
            value_fmt=lambda v: f"{int(float(v))}",
        )

        wav_lf = tk.LabelFrame(noise_frame, text="Wavelet Denoising", bg=UI_BG, fg=TXT_FG)
        wav_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.wavelet_level_var = tk.IntVar(value=int(WAVELET_LEVEL))
        self.slider_wavelet_level, self.slider_wavelet_level_val_lbl = _mk_slider_row(
            wav_lf,
            "Nivel (1..5)",
            self.wavelet_level_var,
            self._on_wavelet_level_change,
            from_=1,
            to=5,
            resolution=1,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )

        nlm_lf = tk.LabelFrame(noise_frame, text="Non-Local Means (NLM)", bg=UI_BG, fg=TXT_FG)
        nlm_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.nlm_h_var = tk.IntVar(value=int(NLM_H))
        self.nlm_h_color_var = tk.IntVar(value=int(NLM_H_COLOR))
        self.nlm_template_window_var = tk.IntVar(value=int(NLM_TEMPLATE_WINDOW))
        self.nlm_search_window_var = tk.IntVar(value=int(NLM_SEARCH_WINDOW))

        self.slider_nlm_h, self.slider_nlm_h_val_lbl = _mk_slider_row(
            nlm_lf,
            "H (luma)",
            self.nlm_h_var,
            self._on_nlm_h_change,
            from_=1,
            to=30,
            resolution=1,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )
        self.slider_nlm_h_color, self.slider_nlm_h_color_val_lbl = _mk_slider_row(
            nlm_lf,
            "H (color)",
            self.nlm_h_color_var,
            self._on_nlm_h_color_change,
            from_=1,
            to=30,
            resolution=1,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )
        self.slider_nlm_template_window, self.slider_nlm_template_window_val_lbl = _mk_slider_row(
            nlm_lf,
            "Template window (impar)",
            self.nlm_template_window_var,
            self._on_nlm_template_window_change,
            from_=3,
            to=7,
            resolution=2,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )
        self.slider_nlm_search_window, self.slider_nlm_search_window_val_lbl = _mk_slider_row(
            nlm_lf,
            "Search window (impar)",
            self.nlm_search_window_var,
            self._on_nlm_search_window_change,
            from_=7,
            to=31,
            resolution=2,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )

        # ----- Filtrado en Fourier -----
        four_lf = tk.LabelFrame(noise_frame, text="Filtrado en Fourier", bg=UI_BG, fg=TXT_FG)
        four_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.fourier_cutoff_var = tk.DoubleVar(value=float(FOURIER_CUTOFF_RATIO))
        self.fourier_soften_var = tk.DoubleVar(value=float(FOURIER_SOFTEN_RATIO))

        self.slider_fourier_cutoff, self.slider_fourier_cutoff_val_lbl = _mk_slider_row(
            four_lf,
            "Corte (cutoff)",
            self.fourier_cutoff_var,
            self._on_fourier_cutoff_change,
            from_=0.05,
            to=0.30,
            resolution=0.01,
            value_width=5,
            value_fmt=lambda v: f"{float(v):.2f}",
        )
        self.slider_fourier_soften, self.slider_fourier_soften_val_lbl = _mk_slider_row(
            four_lf,
            "Suavizado borde",
            self.fourier_soften_var,
            self._on_fourier_soften_change,
            from_=0.01,
            to=0.10,
            resolution=0.01,
            value_width=5,
            value_fmt=lambda v: f"{float(v):.2f}",
        )

        # ----- Noise2Noise / Noise2Void -----
        n2v_lf = tk.LabelFrame(noise_frame, text="Noise2Noise / Noise2Void", bg=UI_BG, fg=TXT_FG)
        n2v_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.n2v_sigma_var = tk.DoubleVar(value=float(N2V_SIGMA))
        self.n2v_thr_mult_var = tk.DoubleVar(value=float(N2V_THR_MULT))
        self.n2v_iters_var = tk.IntVar(value=int(N2V_ITERATIONS))

        self.slider_n2v_sigma, self.slider_n2v_sigma_val_lbl = _mk_slider_row(
            n2v_lf,
            "Sigma (blur)",
            self.n2v_sigma_var,
            self._on_n2v_sigma_change,
            from_=0.3,
            to=3.0,
            resolution=0.1,
            value_width=4,
            value_fmt=lambda v: f"{float(v):.1f}",
        )
        self.slider_n2v_thr_mult, self.slider_n2v_thr_mult_val_lbl = _mk_slider_row(
            n2v_lf,
            "Umbral (k)",
            self.n2v_thr_mult_var,
            self._on_n2v_thr_mult_change,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            value_width=4,
            value_fmt=lambda v: f"{float(v):.1f}",
        )
        self.slider_n2v_iters, self.slider_n2v_iters_val_lbl = _mk_slider_row(
            n2v_lf,
            "Iteraciones",
            self.n2v_iters_var,
            self._on_n2v_iters_change,
            from_=1,
            to=5,
            resolution=1,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )

        # ----- Filtro Mediana -----
        med_lf = tk.LabelFrame(noise_frame, text="Filtro Mediana", bg=UI_BG, fg=TXT_FG)
        med_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.median_ksize_var = tk.IntVar(value=int(MEDIAN_KSIZE))
        self.slider_median_ksize, self.slider_median_ksize_val_lbl = _mk_slider_row(
            med_lf,
            "Kernel (impar)",
            self.median_ksize_var,
            self._on_median_ksize_change,
            from_=3,
            to=25,
            resolution=2,
            value_width=3,
            value_fmt=lambda v: f"{int(float(v))}",
        )

        # ----- Filtro Gaussiano -----
        gau_lf = tk.LabelFrame(noise_frame, text="Filtro Gaussiano", bg=UI_BG, fg=TXT_FG)
        gau_lf.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.gaussian_sigma_var = tk.DoubleVar(value=float(GAUSSIAN_SIGMA))
        self.gaussian_tile_var = tk.IntVar(value=int(GAUSSIAN_TILE_SIZE))

        self.slider_gaussian_sigma, self.slider_gaussian_sigma_val_lbl = _mk_slider_row(
            gau_lf,
            "Sigma",
            self.gaussian_sigma_var,
            self._on_gaussian_sigma_change,
            from_=0.3,
            to=3.0,
            resolution=0.1,
            value_width=4,
            value_fmt=lambda v: f"{float(v):.1f}",
        )
        self.slider_gaussian_tile, self.slider_gaussian_tile_val_lbl = _mk_slider_row(
            gau_lf,
            "Bloque (px)",
            self.gaussian_tile_var,
            self._on_gaussian_tile_change,
            from_=128,
            to=2048,
            resolution=128,
            value_width=4,
            value_fmt=lambda v: f"{int(float(v))}",
        )

        # Inicializar estado de los sliders según casillas
        self._update_slider_state()

        # Tooltips sobre sliders deshabilitados (Focus tracking)
        DisabledToolTip(
            self.slider_radius_pmax,
            'Para usar estos sliders, habilite el método "PMax" (marque la casilla PMax).'
        )
        DisabledToolTip(
            self.slider_radius_pmax_val_lbl,
            'Para usar estos sliders, habilite el método "PMax" (marque la casilla PMax).',
            state_widget=self.slider_radius_pmax
        )
        DisabledToolTip(
            self.slider_smooth_pmax,
            'Para usar estos sliders, habilite el método "PMax" (marque la casilla PMax).'
        )
        DisabledToolTip(
            self.slider_smooth_pmax_val_lbl,
            'Para usar estos sliders, habilite el método "PMax" (marque la casilla PMax).',
            state_widget=self.slider_smooth_pmax
        )

        DisabledToolTip(
            self.slider_radius_weighted,
            'Para usar estos sliders, habilite el método "Promedio ponderado" (marque la casilla Promedio ponderado).'
        )
        DisabledToolTip(
            self.slider_radius_weighted_val_lbl,
            'Para usar estos sliders, habilite el método "Promedio ponderado" (marque la casilla Promedio ponderado).',
            state_widget=self.slider_radius_weighted
        )
        DisabledToolTip(
            self.slider_smooth_weighted,
            'Para usar estos sliders, habilite el método "Promedio ponderado" (marque la casilla Promedio ponderado).'
        )
        DisabledToolTip(
            self.slider_smooth_weighted_val_lbl,
            'Para usar estos sliders, habilite el método "Promedio ponderado" (marque la casilla Promedio ponderado).',
            state_widget=self.slider_smooth_weighted
        )

        DisabledToolTip(
            self.slider_radius_depth,
            'Para usar estos sliders, habilite el método "Mapa de profundidad" (marque la casilla Mapa de profundidad).'
        )
        DisabledToolTip(
            self.slider_radius_depth_val_lbl,
            'Para usar estos sliders, habilite el método "Mapa de profundidad" (marque la casilla Mapa de profundidad).',
            state_widget=self.slider_radius_depth
        )
        DisabledToolTip(
            self.slider_smooth_depth,
            'Para usar estos sliders, habilite el método "Mapa de profundidad" (marque la casilla Mapa de profundidad).'
        )
        DisabledToolTip(
            self.slider_smooth_depth_val_lbl,
            'Para usar estos sliders, habilite el método "Mapa de profundidad" (marque la casilla Mapa de profundidad).',
            state_widget=self.slider_smooth_depth
        )

        # Tooltips sobre sliders deshabilitados (Integración)
        _tt_mean = 'Para usar estos sliders, habilite el método "Promedio (mean stacking)" (marque la casilla Promedio).'
        DisabledToolTip(self.slider_int_mean_align_max_side, _tt_mean)
        DisabledToolTip(self.slider_int_mean_align_max_side_val_lbl, _tt_mean,
                        state_widget=self.slider_int_mean_align_max_side)
        DisabledToolTip(self.slider_int_mean_ecc_iters, _tt_mean)
        DisabledToolTip(self.slider_int_mean_ecc_iters_val_lbl, _tt_mean, state_widget=self.slider_int_mean_ecc_iters)
        DisabledToolTip(self.slider_int_mean_ecc_eps_exp, _tt_mean)
        DisabledToolTip(self.slider_int_mean_ecc_eps_exp_val_lbl, _tt_mean,
                        state_widget=self.slider_int_mean_ecc_eps_exp)

        _tt_median = 'Para usar estos sliders, habilite el método "Mediana (median stacking)" (marque la casilla Mediana).'
        DisabledToolTip(self.slider_int_median_align_max_side, _tt_median)
        DisabledToolTip(self.slider_int_median_align_max_side_val_lbl, _tt_median,
                        state_widget=self.slider_int_median_align_max_side)
        DisabledToolTip(self.slider_int_median_ecc_iters, _tt_median)
        DisabledToolTip(self.slider_int_median_ecc_iters_val_lbl, _tt_median,
                        state_widget=self.slider_int_median_ecc_iters)
        DisabledToolTip(self.slider_int_median_ecc_eps_exp, _tt_median)
        DisabledToolTip(self.slider_int_median_ecc_eps_exp_val_lbl, _tt_median,
                        state_widget=self.slider_int_median_ecc_eps_exp)

        _tt_sum = 'Para usar estos sliders, habilite el método "Suma (sum stacking)" (marque la casilla Suma).'
        DisabledToolTip(self.slider_int_sum_align_max_side, _tt_sum)
        DisabledToolTip(self.slider_int_sum_align_max_side_val_lbl, _tt_sum,
                        state_widget=self.slider_int_sum_align_max_side)
        DisabledToolTip(self.slider_int_sum_ecc_iters, _tt_sum)
        DisabledToolTip(self.slider_int_sum_ecc_iters_val_lbl, _tt_sum, state_widget=self.slider_int_sum_ecc_iters)
        DisabledToolTip(self.slider_int_sum_ecc_eps_exp, _tt_sum)
        DisabledToolTip(self.slider_int_sum_ecc_eps_exp_val_lbl, _tt_sum, state_widget=self.slider_int_sum_ecc_eps_exp)

        _tt_sigma = 'Para usar estos sliders, habilite el método "Sigma-clipping" (marque la casilla Sigma-clipping).'
        DisabledToolTip(self.slider_int_sigma_align_max_side, _tt_sigma)
        DisabledToolTip(self.slider_int_sigma_align_max_side_val_lbl, _tt_sigma,
                        state_widget=self.slider_int_sigma_align_max_side)
        DisabledToolTip(self.slider_int_sigma_ecc_iters, _tt_sigma)
        DisabledToolTip(self.slider_int_sigma_ecc_iters_val_lbl, _tt_sigma,
                        state_widget=self.slider_int_sigma_ecc_iters)
        DisabledToolTip(self.slider_int_sigma_ecc_eps_exp, _tt_sigma)
        DisabledToolTip(self.slider_int_sigma_ecc_eps_exp_val_lbl, _tt_sigma,
                        state_widget=self.slider_int_sigma_ecc_eps_exp)
        DisabledToolTip(self.slider_int_sigma_k, _tt_sigma)
        DisabledToolTip(self.slider_int_sigma_k_val_lbl, _tt_sigma, state_widget=self.slider_int_sigma_k)

        # Tooltips sobre sliders deshabilitados (Filtros de ruido)
        _tt_bm3d = 'Para usar estos sliders, habilite el filtro "BM3D" (marque la casilla BM3D).'
        DisabledToolTip(self.slider_bm3d_sigma, _tt_bm3d)
        DisabledToolTip(self.slider_bm3d_sigma_val_lbl, _tt_bm3d, state_widget=self.slider_bm3d_sigma)

        _tt_bil = 'Para usar estos sliders, habilite el filtro "Bilateral" (marque la casilla Bilateral).'
        DisabledToolTip(self.slider_bilateral_d, _tt_bil)
        DisabledToolTip(self.slider_bilateral_d_val_lbl, _tt_bil, state_widget=self.slider_bilateral_d)
        DisabledToolTip(self.slider_bilateral_sigma_color, _tt_bil)
        DisabledToolTip(self.slider_bilateral_sigma_color_val_lbl, _tt_bil,
                        state_widget=self.slider_bilateral_sigma_color)
        DisabledToolTip(self.slider_bilateral_sigma_space, _tt_bil)
        DisabledToolTip(self.slider_bilateral_sigma_space_val_lbl, _tt_bil,
                        state_widget=self.slider_bilateral_sigma_space)

        _tt_wav = 'Para usar este slider, habilite el filtro "Wavelet Denoising" (marque la casilla Wavelet).'
        DisabledToolTip(self.slider_wavelet_level, _tt_wav)
        DisabledToolTip(self.slider_wavelet_level_val_lbl, _tt_wav, state_widget=self.slider_wavelet_level)

        _tt_nlm = 'Para usar estos sliders, habilite el filtro "Non-Local Means" (marque la casilla NLM).'
        DisabledToolTip(self.slider_nlm_h, _tt_nlm)
        DisabledToolTip(self.slider_nlm_h_val_lbl, _tt_nlm, state_widget=self.slider_nlm_h)
        DisabledToolTip(self.slider_nlm_h_color, _tt_nlm)
        DisabledToolTip(self.slider_nlm_h_color_val_lbl, _tt_nlm, state_widget=self.slider_nlm_h_color)
        DisabledToolTip(self.slider_nlm_template_window, _tt_nlm)
        DisabledToolTip(self.slider_nlm_template_window_val_lbl, _tt_nlm, state_widget=self.slider_nlm_template_window)
        DisabledToolTip(self.slider_nlm_search_window, _tt_nlm)
        DisabledToolTip(self.slider_nlm_search_window_val_lbl, _tt_nlm, state_widget=self.slider_nlm_search_window)

        _tt_four = 'Para usar estos sliders, habilite el filtro "Filtrado en Fourier" (marque la casilla Filtrado en Fourier).'
        DisabledToolTip(self.slider_fourier_cutoff, _tt_four)
        DisabledToolTip(self.slider_fourier_cutoff_val_lbl, _tt_four, state_widget=self.slider_fourier_cutoff)
        DisabledToolTip(self.slider_fourier_soften, _tt_four)
        DisabledToolTip(self.slider_fourier_soften_val_lbl, _tt_four, state_widget=self.slider_fourier_soften)

        _tt_n2v = 'Para usar estos sliders, habilite el filtro "Noise2Noise / Noise2Void" (marque la casilla Noise2Noise / Noise2Void).'
        DisabledToolTip(self.slider_n2v_sigma, _tt_n2v)
        DisabledToolTip(self.slider_n2v_sigma_val_lbl, _tt_n2v, state_widget=self.slider_n2v_sigma)
        DisabledToolTip(self.slider_n2v_thr_mult, _tt_n2v)
        DisabledToolTip(self.slider_n2v_thr_mult_val_lbl, _tt_n2v, state_widget=self.slider_n2v_thr_mult)
        DisabledToolTip(self.slider_n2v_iters, _tt_n2v)
        DisabledToolTip(self.slider_n2v_iters_val_lbl, _tt_n2v, state_widget=self.slider_n2v_iters)

        _tt_med = 'Para usar este slider, habilite el filtro "Filtro Mediana" (marque la casilla Filtro Mediana).'
        DisabledToolTip(self.slider_median_ksize, _tt_med)
        DisabledToolTip(self.slider_median_ksize_val_lbl, _tt_med, state_widget=self.slider_median_ksize)

        _tt_gau = 'Para usar estos sliders, habilite el filtro "Filtro Gaussiano" (marque la casilla Filtro Gaussiano).'
        DisabledToolTip(self.slider_gaussian_sigma, _tt_gau)
        DisabledToolTip(self.slider_gaussian_sigma_val_lbl, _tt_gau, state_widget=self.slider_gaussian_sigma)
        DisabledToolTip(self.slider_gaussian_tile, _tt_gau)
        DisabledToolTip(self.slider_gaussian_tile_val_lbl, _tt_gau, state_widget=self.slider_gaussian_tile)

        # =============================
        #  COLUMNA DERECHA (ÁREA DE TRABAJO)
        # =============================

        # Frame principal de la zona de trabajo
        work = tk.Frame(main, bg=UI_BG)
        work.grid(row=0, column=1, sticky="nsew")

        # Dos columnas: izquierda = Imagen fuente, derecha = Resultado
        work.columnconfigure(0, weight=1)
        work.columnconfigure(1, weight=1)
        work.rowconfigure(0, weight=0)  # fila de labels
        work.rowconfigure(1, weight=1)  # fila de canvases, misma altura

        # --- Columna 0: Imagen fuente ---
        tk.Label(work, text="Imagen fuente (en uso):",
                 bg=UI_BG, fg=TXT_FG).grid(row=0, column=0, sticky="w")

        self.src_canvas = tk.Canvas(work, bg=CANVAS_BG, highlightthickness=0)
        self.src_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(4, 0))

        # --- Columna 1: Resultado ---
        tk.Label(work, text="Resultado (en formación):",
                 bg=UI_BG, fg=TXT_FG).grid(row=0, column=1, sticky="w")

        self.res_canvas = tk.Canvas(work, bg=CANVAS_BG, highlightthickness=0)
        self.res_canvas.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))

        # =============================
        #  ESPACIO LIBRE INFERIOR (RESERVADO)
        # =============================
        bottom = tk.Frame(main, bg=UI_BG)
        bottom.grid(row=1, column=1, sticky="nsew")

    # ---------------- UI selección ----------------

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="Selecciona imágenes del stack",
            filetypes=[
                ("Imágenes",
                 "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.nef *.raw *.dng *.arw *.cr2 *.cr3 *.rw2 *.orf *.raf *.pef *.srw"),
                ("Todos", "*.*"),
            ],
        )

        if not paths:
            return

        self.image_paths = sorted(list(paths))
        self.folder_paths = []  # si cargas imágenes directas, vacía carpetas

        self.listbox.delete(0, tk.END)
        for p in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(p))

        self._set_progress(0, "Listo para procesar.", src_preview=None, res_preview=None)
        self.btn_save.config(state=tk.NORMAL)
        self.result = None
        self._clear_previews()

        first = imread_unicode(self.image_paths[0])
        if first is not None:
            self._show_src_preview(first)

    # ---------- control de sliders ----------
    def _set_slider_appearance(self, control, value_label, enabled: bool):
        """
        Cambia estado y colores de un control (Knob o Scale) y su etiqueta (si existe).
        """
        # Knob (Canvas custom)
        if hasattr(control, "set_enabled") and callable(getattr(control, "set_enabled")):
            control.set_theme(
                bg_enabled=self.SLIDER_BG_ENABLED,
                fg_enabled=self.SLIDER_FG_ENABLED,
                dial_enabled=self.SLIDER_TROUGH_ENABLED,
                bg_disabled=self.SLIDER_BG_DISABLED,
                fg_disabled=self.SLIDER_FG_DISABLED,
                dial_disabled=self.SLIDER_TROUGH_DISABLED,
            )
            control.set_enabled(enabled)

            if value_label is not None:
                value_label.config(
                    bg=self.SLIDER_BG_ENABLED if enabled else self.SLIDER_BG_DISABLED,
                    fg=self.SLIDER_FG_ENABLED if enabled else self.SLIDER_FG_DISABLED,
                )
            return

        # Fallback: tk.Scale
        if enabled:
            control.config(
                state=tk.NORMAL,
                bg=self.SLIDER_BG_ENABLED,
                fg=self.SLIDER_FG_ENABLED,
                troughcolor=self.SLIDER_TROUGH_ENABLED,
            )
            if value_label is not None:
                value_label.config(
                    bg=self.SLIDER_BG_ENABLED,
                    fg=self.SLIDER_FG_ENABLED,
                )
        else:
            control.config(
                state=tk.DISABLED,
                bg=self.SLIDER_BG_DISABLED,
                fg=self.SLIDER_FG_DISABLED,
                troughcolor=self.SLIDER_TROUGH_DISABLED,
            )
            if value_label is not None:
                value_label.config(
                    bg=self.SLIDER_BG_DISABLED,
                    fg=self.SLIDER_FG_DISABLED,
                )

    def _update_slider_state(self):
        """
        Habilita / deshabilita sliders según las casillas:
        - Focus tracking: radio y suavizado por método.
        - Integración: parámetros de alineación ECC y sigma-clipping.
        - Filtros de ruido: parámetros BM3D y Bilateral.
        """
        # Focus tracking
        self._set_slider_appearance(self.slider_radius_pmax, self.slider_radius_pmax_val_lbl, self.use_pmax.get())
        self._set_slider_appearance(self.slider_smooth_pmax, self.slider_smooth_pmax_val_lbl, self.use_pmax.get())

        self._set_slider_appearance(self.slider_radius_weighted, self.slider_radius_weighted_val_lbl,
                                    self.use_weighted.get())
        self._set_slider_appearance(self.slider_smooth_weighted, self.slider_smooth_weighted_val_lbl,
                                    self.use_weighted.get())

        self._set_slider_appearance(self.slider_radius_depth, self.slider_radius_depth_val_lbl, self.use_depth.get())
        self._set_slider_appearance(self.slider_smooth_depth, self.slider_smooth_depth_val_lbl, self.use_depth.get())

        # Integración (sliders independientes por método)
        mean_on = bool(self.use_int_mean.get())
        median_on = bool(self.use_int_median.get())
        sum_on = bool(self.use_int_sum.get())
        sigma_on = bool(self.use_int_sigma.get())

        # Promedio
        if hasattr(self, "slider_int_mean_align_max_side"):
            self._set_slider_appearance(self.slider_int_mean_align_max_side,
                                        self.slider_int_mean_align_max_side_val_lbl, mean_on)
        if hasattr(self, "slider_int_mean_ecc_iters"):
            self._set_slider_appearance(self.slider_int_mean_ecc_iters, self.slider_int_mean_ecc_iters_val_lbl, mean_on)
        if hasattr(self, "slider_int_mean_ecc_eps_exp"):
            self._set_slider_appearance(self.slider_int_mean_ecc_eps_exp, self.slider_int_mean_ecc_eps_exp_val_lbl,
                                        mean_on)

        # Mediana
        if hasattr(self, "slider_int_median_align_max_side"):
            self._set_slider_appearance(self.slider_int_median_align_max_side,
                                        self.slider_int_median_align_max_side_val_lbl, median_on)
        if hasattr(self, "slider_int_median_ecc_iters"):
            self._set_slider_appearance(self.slider_int_median_ecc_iters, self.slider_int_median_ecc_iters_val_lbl,
                                        median_on)
        if hasattr(self, "slider_int_median_ecc_eps_exp"):
            self._set_slider_appearance(self.slider_int_median_ecc_eps_exp, self.slider_int_median_ecc_eps_exp_val_lbl,
                                        median_on)

        # Suma
        if hasattr(self, "slider_int_sum_align_max_side"):
            self._set_slider_appearance(self.slider_int_sum_align_max_side, self.slider_int_sum_align_max_side_val_lbl,
                                        sum_on)
        if hasattr(self, "slider_int_sum_ecc_iters"):
            self._set_slider_appearance(self.slider_int_sum_ecc_iters, self.slider_int_sum_ecc_iters_val_lbl, sum_on)
        if hasattr(self, "slider_int_sum_ecc_eps_exp"):
            self._set_slider_appearance(self.slider_int_sum_ecc_eps_exp, self.slider_int_sum_ecc_eps_exp_val_lbl,
                                        sum_on)

        # Sigma-clipping
        if hasattr(self, "slider_int_sigma_align_max_side"):
            self._set_slider_appearance(self.slider_int_sigma_align_max_side,
                                        self.slider_int_sigma_align_max_side_val_lbl, sigma_on)
        if hasattr(self, "slider_int_sigma_ecc_iters"):
            self._set_slider_appearance(self.slider_int_sigma_ecc_iters, self.slider_int_sigma_ecc_iters_val_lbl,
                                        sigma_on)
        if hasattr(self, "slider_int_sigma_ecc_eps_exp"):
            self._set_slider_appearance(self.slider_int_sigma_ecc_eps_exp, self.slider_int_sigma_ecc_eps_exp_val_lbl,
                                        sigma_on)
        if hasattr(self, "slider_int_sigma_k"):
            self._set_slider_appearance(self.slider_int_sigma_k, self.slider_int_sigma_k_val_lbl, sigma_on)

        # Filtros de ruido
        four_on = bool(self.use_fourier.get())
        n2v_on = bool(self.use_n2v.get())
        med_on = bool(self.use_median.get())
        gau_on = bool(self.use_gaussian.get())
        bm3d_on = bool(self.use_bm3d.get())
        bil_on = bool(self.use_bilateral.get())
        wav_on = bool(self.use_wavelet.get())
        nlm_on = bool(self.use_nlm.get())

        # Fourier
        if hasattr(self, "slider_fourier_cutoff"):
            self._set_slider_appearance(self.slider_fourier_cutoff, self.slider_fourier_cutoff_val_lbl, four_on)
        if hasattr(self, "slider_fourier_soften"):
            self._set_slider_appearance(self.slider_fourier_soften, self.slider_fourier_soften_val_lbl, four_on)

        # Noise2Noise / Noise2Void
        if hasattr(self, "slider_n2v_sigma"):
            self._set_slider_appearance(self.slider_n2v_sigma, self.slider_n2v_sigma_val_lbl, n2v_on)
        if hasattr(self, "slider_n2v_thr_mult"):
            self._set_slider_appearance(self.slider_n2v_thr_mult, self.slider_n2v_thr_mult_val_lbl, n2v_on)
        if hasattr(self, "slider_n2v_iters"):
            self._set_slider_appearance(self.slider_n2v_iters, self.slider_n2v_iters_val_lbl, n2v_on)

        # Mediana
        if hasattr(self, "slider_median_ksize"):
            self._set_slider_appearance(self.slider_median_ksize, self.slider_median_ksize_val_lbl, med_on)

        # Gaussiano
        if hasattr(self, "slider_gaussian_sigma"):
            self._set_slider_appearance(self.slider_gaussian_sigma, self.slider_gaussian_sigma_val_lbl, gau_on)
        if hasattr(self, "slider_gaussian_tile"):
            self._set_slider_appearance(self.slider_gaussian_tile, self.slider_gaussian_tile_val_lbl, gau_on)

        # BM3D
        if hasattr(self, "slider_bm3d_sigma"):
            self._set_slider_appearance(self.slider_bm3d_sigma, self.slider_bm3d_sigma_val_lbl, bm3d_on)

        # Bilateral
        if hasattr(self, "slider_bilateral_d"):
            self._set_slider_appearance(self.slider_bilateral_d, self.slider_bilateral_d_val_lbl, bil_on)
        if hasattr(self, "slider_bilateral_sigma_color"):
            self._set_slider_appearance(self.slider_bilateral_sigma_color, self.slider_bilateral_sigma_color_val_lbl,
                                        bil_on)
        if hasattr(self, "slider_bilateral_sigma_space"):
            self._set_slider_appearance(self.slider_bilateral_sigma_space, self.slider_bilateral_sigma_space_val_lbl,
                                        bil_on)

        # Wavelet
        if hasattr(self, "slider_wavelet_level"):
            self._set_slider_appearance(self.slider_wavelet_level, self.slider_wavelet_level_val_lbl, wav_on)

        # NLM
        if hasattr(self, "slider_nlm_h"):
            self._set_slider_appearance(self.slider_nlm_h, self.slider_nlm_h_val_lbl, nlm_on)
        if hasattr(self, "slider_nlm_h_color"):
            self._set_slider_appearance(self.slider_nlm_h_color, self.slider_nlm_h_color_val_lbl, nlm_on)
        if hasattr(self, "slider_nlm_template_window"):
            self._set_slider_appearance(self.slider_nlm_template_window, self.slider_nlm_template_window_val_lbl,
                                        nlm_on)
        if hasattr(self, "slider_nlm_search_window"):
            self._set_slider_appearance(self.slider_nlm_search_window, self.slider_nlm_search_window_val_lbl, nlm_on)

    # -------------------------
    # Exclusión: Focus tracking <-> Filtros de ruido
    # -------------------------

    def _set_chk_enabled(self, chk, enabled: bool):
        if chk is None:
            return
        try:
            chk.config(
                state=(tk.NORMAL if enabled else tk.DISABLED),
                fg=(self.TXT_FG if enabled else self.BTN_FG_DISABLED),
            )
        except Exception:
            pass

    def _update_focus_noise_exclusivity(self):
        # Puede ser llamado durante _build_ui antes de crear todos los widgets
        chk_pmax = getattr(self, "chk_pmax", None)
        chk_weighted = getattr(self, "chk_weighted", None)
        chk_depth = getattr(self, "chk_depth", None)

        chk_int_mean = getattr(self, "chk_int_mean", None)
        chk_int_median = getattr(self, "chk_int_median", None)
        chk_int_sum = getattr(self, "chk_int_sum", None)
        chk_int_sigma = getattr(self, "chk_int_sigma", None)

        chk_cal = getattr(self, "chk_calibration", None)
        chk_four = getattr(self, "chk_fourier", None)
        chk_n2v = getattr(self, "chk_n2v", None)
        chk_med = getattr(self, "chk_median", None)
        chk_gau = getattr(self, "chk_gaussian", None)
        chk_bm3d = getattr(self, "chk_bm3d", None)
        chk_bil = getattr(self, "chk_bilateral", None)
        chk_wav = getattr(self, "chk_wavelet", None)
        chk_nlm = getattr(self, "chk_nlm", None)
        chk_serial = getattr(self, "chk_serial_filters", None)

        focus_selected = bool(self.use_pmax.get() or self.use_weighted.get() or self.use_depth.get())
        integ_selected = bool(
            self.use_int_mean.get()
            or self.use_int_median.get()
            or self.use_int_sum.get()
            or self.use_int_sigma.get()
        )
        noise_selected = bool(
            self.use_calibration.get()
            or self.use_fourier.get()
            or self.use_n2v.get()
            or self.use_median.get()
            or self.use_gaussian.get()
            or self.use_bm3d.get()
            or self.use_bilateral.get()
            or self.use_wavelet.get()
            or self.use_nlm.get()
            or self.use_serial_filters.get()
        )

        # Solo se puede trabajar con una sección a la vez:
        #   - Focus tracking
        #   - Integración
        #   - Filtros de ruido
        if focus_selected:
            # Bloquear integración
            self._set_chk_enabled(chk_int_mean, False)
            self._set_chk_enabled(chk_int_median, False)
            self._set_chk_enabled(chk_int_sum, False)
            self._set_chk_enabled(chk_int_sigma, False)

            # Bloquear filtros de ruido
            self._set_chk_enabled(chk_cal, False)
            self._set_chk_enabled(chk_four, False)
            self._set_chk_enabled(chk_n2v, False)
            self._set_chk_enabled(chk_med, False)
            self._set_chk_enabled(chk_gau, False)
            self._set_chk_enabled(chk_bm3d, False)
            self._set_chk_enabled(chk_bil, False)
            self._set_chk_enabled(chk_wav, False)
            self._set_chk_enabled(chk_nlm, False)
            self._set_chk_enabled(chk_serial, False)

            # Asegurar focus habilitado
            self._set_chk_enabled(chk_pmax, True)
            self._set_chk_enabled(chk_weighted, True)
            self._set_chk_enabled(chk_depth, True)

        elif integ_selected:
            # Bloquear focus tracking
            self._set_chk_enabled(chk_pmax, False)
            self._set_chk_enabled(chk_weighted, False)
            self._set_chk_enabled(chk_depth, False)

            # Bloquear filtros de ruido
            self._set_chk_enabled(chk_cal, False)
            self._set_chk_enabled(chk_four, False)
            self._set_chk_enabled(chk_n2v, False)
            self._set_chk_enabled(chk_med, False)
            self._set_chk_enabled(chk_gau, False)
            self._set_chk_enabled(chk_bm3d, False)
            self._set_chk_enabled(chk_bil, False)
            self._set_chk_enabled(chk_wav, False)
            self._set_chk_enabled(chk_nlm, False)
            self._set_chk_enabled(chk_serial, False)

            # Asegurar integración habilitada
            self._set_chk_enabled(chk_int_mean, True)
            self._set_chk_enabled(chk_int_median, True)
            self._set_chk_enabled(chk_int_sum, True)
            self._set_chk_enabled(chk_int_sigma, True)

        elif noise_selected:
            # Bloquear focus tracking
            self._set_chk_enabled(chk_pmax, False)
            self._set_chk_enabled(chk_weighted, False)
            self._set_chk_enabled(chk_depth, False)

            # Bloquear integración
            self._set_chk_enabled(chk_int_mean, False)
            self._set_chk_enabled(chk_int_median, False)
            self._set_chk_enabled(chk_int_sum, False)
            self._set_chk_enabled(chk_int_sigma, False)

            # Asegurar filtros habilitados
            self._set_chk_enabled(chk_cal, True)
            self._set_chk_enabled(chk_four, True)
            self._set_chk_enabled(chk_n2v, True)
            self._set_chk_enabled(chk_med, True)
            self._set_chk_enabled(chk_gau, True)
            self._set_chk_enabled(chk_bm3d, True)
            self._set_chk_enabled(chk_bil, True)
            self._set_chk_enabled(chk_wav, True)
            self._set_chk_enabled(chk_nlm, True)
            self._set_chk_enabled(chk_serial, True)

        else:
            # Nada seleccionado: permitir todas las secciones
            self._set_chk_enabled(chk_pmax, True)
            self._set_chk_enabled(chk_weighted, True)
            self._set_chk_enabled(chk_depth, True)

            self._set_chk_enabled(chk_int_mean, True)
            self._set_chk_enabled(chk_int_median, True)
            self._set_chk_enabled(chk_int_sum, True)
            self._set_chk_enabled(chk_int_sigma, True)

            self._set_chk_enabled(chk_cal, True)
            self._set_chk_enabled(chk_four, True)
            self._set_chk_enabled(chk_n2v, True)
            self._set_chk_enabled(chk_med, True)
            self._set_chk_enabled(chk_gau, True)
            self._set_chk_enabled(chk_bm3d, True)
            self._set_chk_enabled(chk_bil, True)
            self._set_chk_enabled(chk_wav, True)
            self._set_chk_enabled(chk_nlm, True)
            self._set_chk_enabled(chk_serial, True)

    def _on_focus_method_toggle(self):
        # Si se activa algún método de focus tracking, apagar Integración y filtros de ruido
        if self.use_pmax.get() or self.use_weighted.get() or self.use_depth.get():
            # Auto-seleccionar pestaña de parámetros correspondiente
            nb = getattr(self, "knobs_nb", None)
            tab = getattr(self, "tab_focus", None)
            if nb is not None and tab is not None:
                try:
                    nb.select(tab)
                except Exception:
                    pass

            # Apagar integración
            self.use_int_mean.set(False)
            self.use_int_median.set(False)
            self.use_int_sum.set(False)
            self.use_int_sigma.set(False)

            # Apagar filtros de ruido
            self.use_calibration.set(False)
            self.use_fourier.set(False)
            self.use_n2v.set(False)
            self.use_median.set(False)
            self.use_gaussian.set(False)
            self.use_bm3d.set(False)
            self.use_bilateral.set(False)
            self.use_wavelet.set(False)
            self.use_nlm.set(False)

            # Apagar modo seriado (no aplica durante apilado)
            self.use_serial_filters.set(False)
            self.serial_filter_order = []
            self._refresh_serial_numbers()

        self._update_slider_state()
        self._update_focus_noise_exclusivity()

    def _on_integration_method_toggle(self):
        # Si se activa algún método de integración, apagar Focus Tracking y Filtros de ruido
        if (
                self.use_int_mean.get()
                or self.use_int_median.get()
                or self.use_int_sum.get()
                or self.use_int_sigma.get()
        ):
            # Auto-seleccionar pestaña de parámetros correspondiente
            nb = getattr(self, "knobs_nb", None)
            tab = getattr(self, "tab_integ", None)
            if nb is not None and tab is not None:
                try:
                    nb.select(tab)
                except Exception:
                    pass

            # Apagar focus tracking
            self.use_pmax.set(False)
            self.use_weighted.set(False)
            self.use_depth.set(False)

            # Apagar filtros de ruido
            self.use_calibration.set(False)
            self.use_fourier.set(False)
            self.use_n2v.set(False)
            self.use_median.set(False)
            self.use_gaussian.set(False)
            self.use_bm3d.set(False)
            self.use_bilateral.set(False)
            self.use_wavelet.set(False)
            self.use_nlm.set(False)

            # Apagar modo seriado (no aplica durante apilado)
            self.use_serial_filters.set(False)
            self.serial_filter_order = []
            self._refresh_serial_numbers()

        self._update_slider_state()
        self._update_focus_noise_exclusivity()

    def _on_noise_filter_toggle(self):
        # Si se activa algún filtro de ruido, apagar métodos de focus tracking e integración
        if (
                self.use_calibration.get()
                or self.use_fourier.get()
                or self.use_n2v.get()
                or self.use_median.get()
                or self.use_gaussian.get()
                or self.use_bm3d.get()
                or self.use_bilateral.get()
                or self.use_wavelet.get()
                or self.use_nlm.get()
                or self.use_serial_filters.get()
        ):
            # Auto-seleccionar pestaña de parámetros correspondiente
            nb = getattr(self, "knobs_nb", None)
            tab = getattr(self, "tab_noise", None)
            if nb is not None and tab is not None:
                try:
                    nb.select(tab)
                except Exception:
                    pass

            # Apagar focus tracking
            self.use_pmax.set(False)
            self.use_weighted.set(False)
            self.use_depth.set(False)

            # Apagar integración
            self.use_int_mean.set(False)
            self.use_int_median.set(False)
            self.use_int_sum.set(False)
            self.use_int_sigma.set(False)

        self._update_slider_state()
        self._update_focus_noise_exclusivity()

    # ----------------"Suma seriada de filtros" ----------------

    def _serial_var_is_on(self, tag: str) -> bool:
        """Devuelve True si el filtro 'tag' está marcado."""
        for t, var, _, _ in getattr(self, "_serial_filter_defs", []) or []:
            if t == tag:
                try:
                    return bool(var.get())
                except Exception:
                    return False
        return False

    def _refresh_serial_numbers(self):
        """Actualiza los textos de los checkboxes para mostrar numeración (1..N) cuando el modo seriado está activo."""
        defs = getattr(self, "_serial_filter_defs", None) or []
        if not defs:
            return

        # Limpieza: eliminar del orden los tags que ya no estén marcados
        self.serial_filter_order = [t for t in self.serial_filter_order if self._serial_var_is_on(t)]

        if not self.use_serial_filters.get():
            # Modo apagado: restaurar texto base
            for tag, var, chk, base_text in defs:
                try:
                    chk.config(text=base_text)
                except Exception:
                    pass
            return

        order_index = {tag: i + 1 for i, tag in enumerate(self.serial_filter_order)}

        for tag, var, chk, base_text in defs:
            try:
                if tag in order_index:
                    chk.config(text=f"{base_text}  {order_index[tag]}")
                else:
                    chk.config(text=base_text)
            except Exception:
                pass

    def _on_serial_filters_toggle(self):
        """Al activar, inicializa numeración con los filtros ya marcados; al desactivar, limpia numeración."""
        if self.use_serial_filters.get():
            # Serial pertenece a "Filtros de ruido": apagar Focus Tracking e Integración
            self.use_pmax.set(False)
            self.use_weighted.set(False)
            self.use_depth.set(False)

            self.use_int_mean.set(False)
            self.use_int_median.set(False)
            self.use_int_sum.set(False)
            self.use_int_sigma.set(False)

            # Si no hay orden aún, tomar el orden visual/definido y numerar los ya marcados
            if not self.serial_filter_order:
                for tag, var, chk, base_text in getattr(self, "_serial_filter_defs", []) or []:
                    try:
                        if var.get():
                            self.serial_filter_order.append(tag)
                    except Exception:
                        pass
        else:
            self.serial_filter_order = []

        self._refresh_serial_numbers()

        # Mantener exclusividad de secciones
        self._on_noise_filter_toggle()

    def _on_filter_checkbox_toggled(self, tag: str):
        """Wrapper para TODOS los filtros (para poder numerarlos cuando el modo seriado está activo)."""
        # Caso especial: calibración (abre ventana / puede desmarcarse si cancela)
        if tag == "calibrado":
            self._on_calibration_toggle()

        # Advertencia BM3D (solo al ACTIVAR la casilla)
        if tag == "bm3d" and self.use_bm3d.get():
            self._show_bm3d_warning()

        if self.use_serial_filters.get():
            if self._serial_var_is_on(tag):
                if tag not in self.serial_filter_order:
                    self.serial_filter_order.append(tag)
            else:
                if tag in self.serial_filter_order:
                    self.serial_filter_order = [t for t in self.serial_filter_order if t != tag]

        self._refresh_serial_numbers()

        # Mantener exclusividad existente
        self._on_noise_filter_toggle()

    def _show_bm3d_warning(self):
        msg = (
            "Este filtro puede tardar hasta 15 minutos por imagen.\n\n"
            "BM3D (Block-Matching 3D) es un denoiser avanzado e intensivo en CPU/RAM, "
            "especialmente en imágenes de alta resolución.\n\n"
            "Durante el proceso la interfaz puede parecer congelada: es normal. "
            "Deja que termine."
        )

        win = tk.Toplevel(self)
        win.title("Advertencia: BM3D")
        bg = self.cget("bg")
        win.configure(bg=bg)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        tk.Label(
            win,
            text=msg,
            bg=bg,
            fg="#f0f0f0",
            justify="left",
            wraplength=440,
            font=("", 10),
        ).pack(padx=14, pady=(14, 10))

        tk.Button(
            win,
            text="Aceptar",
            command=win.destroy,
            bg="#6b6b6b",
            fg="#f0f0f0",
            activebackground="#7a7a7a",
            activeforeground="#f0f0f0",
            bd=0,
            padx=16,
            pady=6,
        ).pack(pady=(0, 14))

        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (win.winfo_width() // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (win.winfo_height() // 2)
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.focus_set()
        win.wait_window()

    def _on_radius_pmax_change(self, val):
        """Slider 'Radio PMax' -> RADIUS_PMAX."""
        global RADIUS_PMAX
        try:
            RADIUS_PMAX = max(1, int(float(val)))
        except Exception:
            pass

    def _on_smooth_pmax_change(self, val):
        """Slider 'Suavizado PMax' -> SMOOTH_PMAX."""
        global SMOOTH_PMAX
        try:
            SMOOTH_PMAX = max(1, int(float(val)))
        except Exception:
            pass

    def _on_radius_weighted_change(self, val):
        """Slider 'Radio Promedio ponderado' -> RADIUS_WEIGHTED."""
        global RADIUS_WEIGHTED
        try:
            RADIUS_WEIGHTED = max(1, int(float(val)))
        except Exception:
            pass

    def _on_smooth_weighted_change(self, val):
        """Slider 'Suavizado promedio ponderado' -> SMOOTH_WEIGHTED."""
        global SMOOTH_WEIGHTED
        try:
            SMOOTH_WEIGHTED = max(1, int(float(val)))
        except Exception:
            pass

    def _on_radius_depth_change(self, val):
        """Slider 'Radio mapa profundidad' -> RADIUS_DEPTH."""
        global RADIUS_DEPTH
        try:
            RADIUS_DEPTH = max(1, int(float(val)))
        except Exception:
            pass

    def _on_smooth_depth_change(self, val):
        """Slider 'Suavizado mapa profundidad' -> SMOOTH_DEPTH."""
        global SMOOTH_DEPTH
        try:
            SMOOTH_DEPTH = max(1, int(float(val)))
        except Exception:
            pass

    def _on_smooth_pmax_change(self, val):
        """Slider 'Suavizado PMax' -> SMOOTH_PMAX."""
        global SMOOTH_PMAX
        try:
            SMOOTH_PMAX = max(1, int(float(val)))
        except Exception:
            pass

    def _on_radius_weighted_change(self, val):
        """Slider 'Radio Promedio ponderado' -> RADIUS_WEIGHTED."""
        global RADIUS_WEIGHTED
        try:
            RADIUS_WEIGHTED = max(1, int(float(val)))
        except Exception:
            pass

    def _on_smooth_weighted_change(self, val):
        """Slider 'Suavizado promedio ponderado' -> SMOOTH_WEIGHTED."""
        global SMOOTH_WEIGHTED
        try:
            SMOOTH_WEIGHTED = max(1, int(float(val)))
        except Exception:
            pass

    def _on_radius_depth_change(self, val):
        """Slider 'Radio mapa profundidad' -> RADIUS_DEPTH."""
        global RADIUS_DEPTH
        try:
            RADIUS_DEPTH = max(1, int(float(val)))
        except Exception:
            pass

    def _on_smooth_depth_change(self, val):
        """Slider 'Suavizado mapa profundidad' -> SMOOTH_DEPTH."""
        global SMOOTH_DEPTH
        try:
            SMOOTH_DEPTH = max(1, int(float(val)))
        except Exception:
            pass

    # ---------------- Filtros de ruido: sliders ----------------

    def _on_bm3d_sigma_change(self, val):
        """Slider 'BM3D Sigma' -> BM3D_SIGMA_PSD."""
        global BM3D_SIGMA_PSD
        try:
            v = float(val)
            BM3D_SIGMA_PSD = max(0.0, min(1.0, v))
        except Exception:
            pass

    def _on_bilateral_d_change(self, val):
        """Slider 'Bilateral d' -> BILATERAL_D."""
        global BILATERAL_D
        try:
            BILATERAL_D = max(1, int(float(val)))
        except Exception:
            pass

    def _on_bilateral_sigma_color_change(self, val):
        """Slider 'Bilateral sigmaColor' -> BILATERAL_SIGMA_COLOR."""
        global BILATERAL_SIGMA_COLOR
        try:
            BILATERAL_SIGMA_COLOR = max(1, int(float(val)))
        except Exception:
            pass

    def _on_bilateral_sigma_space_change(self, val):
        """Slider 'Bilateral sigmaSpace' -> BILATERAL_SIGMA_SPACE."""
        global BILATERAL_SIGMA_SPACE
        try:
            BILATERAL_SIGMA_SPACE = max(1, int(float(val)))
        except Exception:
            pass

    def _on_wavelet_level_change(self, val):
        """Slider 'Wavelet nivel' -> WAVELET_LEVEL."""
        global WAVELET_LEVEL
        try:
            WAVELET_LEVEL = max(1, min(5, int(float(val))))
        except Exception:
            pass

    def _on_nlm_h_change(self, val):
        """Slider 'NLM h' -> NLM_H."""
        global NLM_H
        try:
            NLM_H = max(1, int(float(val)))
        except Exception:
            pass

    def _on_nlm_h_color_change(self, val):
        """Slider 'NLM hColor' -> NLM_H_COLOR."""
        global NLM_H_COLOR
        try:
            NLM_H_COLOR = max(1, int(float(val)))
        except Exception:
            pass

    def _on_nlm_template_window_change(self, val):
        """Slider 'NLM templateWindowSize' -> NLM_TEMPLATE_WINDOW (impar)."""
        global NLM_TEMPLATE_WINDOW
        try:
            v = max(3, min(7, int(float(val))))
            if v % 2 == 0:
                v += 1
            NLM_TEMPLATE_WINDOW = v
        except Exception:
            pass

    def _on_nlm_search_window_change(self, val):
        """Slider 'NLM searchWindowSize' -> NLM_SEARCH_WINDOW (impar)."""
        global NLM_SEARCH_WINDOW
        try:
            v = max(7, min(31, int(float(val))))
            if v % 2 == 0:
                v += 1
            NLM_SEARCH_WINDOW = v
        except Exception:
            pass

    def _on_fourier_cutoff_change(self, val):
        """Slider 'Fourier cutoff' -> FOURIER_CUTOFF_RATIO."""
        global FOURIER_CUTOFF_RATIO
        try:
            v = float(val)
            FOURIER_CUTOFF_RATIO = max(0.01, min(0.50, v))
        except Exception:
            pass

    def _on_fourier_soften_change(self, val):
        """Slider 'Fourier soften' -> FOURIER_SOFTEN_RATIO."""
        global FOURIER_SOFTEN_RATIO
        try:
            v = float(val)
            FOURIER_SOFTEN_RATIO = max(0.0, min(0.50, v))
        except Exception:
            pass

    def _on_n2v_sigma_change(self, val):
        """Slider 'Noise2Noise sigma' -> N2V_SIGMA."""
        global N2V_SIGMA
        try:
            v = float(val)
            N2V_SIGMA = max(0.0, min(10.0, v))
        except Exception:
            pass

    def _on_n2v_thr_mult_change(self, val):
        """Slider 'Noise2Noise thr mult' -> N2V_THR_MULT."""
        global N2V_THR_MULT
        try:
            v = float(val)
            N2V_THR_MULT = max(0.1, min(10.0, v))
        except Exception:
            pass

    def _on_n2v_iters_change(self, val):
        """Slider 'Noise2Noise iterations' -> N2V_ITERATIONS."""
        global N2V_ITERATIONS
        try:
            N2V_ITERATIONS = max(1, int(float(val)))
        except Exception:
            pass

    def _on_median_ksize_change(self, val):
        """Slider 'Median ksize' -> MEDIAN_KSIZE (impar)."""
        global MEDIAN_KSIZE
        try:
            v = max(3, int(float(val)))
            if v % 2 == 0:
                v += 1
            MEDIAN_KSIZE = v
        except Exception:
            pass

    def _on_gaussian_sigma_change(self, val):
        """Slider 'Gaussian sigma' -> GAUSSIAN_SIGMA."""
        global GAUSSIAN_SIGMA
        try:
            v = float(val)
            GAUSSIAN_SIGMA = max(0.0, min(50.0, v))
        except Exception:
            pass

    def _on_gaussian_tile_change(self, val):
        """Slider 'Gaussian tile' -> GAUSSIAN_TILE_SIZE."""
        global GAUSSIAN_TILE_SIZE
        try:
            GAUSSIAN_TILE_SIZE = max(32, int(float(val)))
        except Exception:
            pass

    # ---------------- Integración: sliders ----------------

    def _on_int_param_change(self, method_key, param_key, val):
        """Actualiza INT_PARAMS[method_key][param_key] desde un slider."""
        global INT_PARAMS
        try:
            if method_key not in INT_PARAMS:
                return

            if param_key == "align_max_side":
                INT_PARAMS[method_key][param_key] = max(400, int(float(val)))
            elif param_key == "ecc_iters":
                INT_PARAMS[method_key][param_key] = max(10, int(float(val)))
            elif param_key == "ecc_eps_exp":
                INT_PARAMS[method_key][param_key] = max(1, int(float(val)))
            elif param_key == "sigma_k":
                INT_PARAMS[method_key][param_key] = float(val)
        except Exception:
            pass

    def _choose_folders_dialog(self, base_dir):
        """
        Diálogo propio de multiselección de carpetas.
        Muestra las subcarpetas de base_dir y permite elegir varias de una.
        """
        win = tk.Toplevel(self)
        win.title("Selecciona carpetas")
        win.transient(self)
        win.grab_set()
        win.geometry("520x420")

        tk.Label(
            win,
            text=f"Selecciona una o varias carpetas dentro de:\n{base_dir}",
            anchor="w",
            justify="left"
        ).pack(fill=tk.X, padx=10, pady=(10, 6))

        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        lb = tk.Listbox(frame, selectmode=tk.EXTENDED)
        sb = tk.Scrollbar(frame, orient="vertical", command=lb.yview)
        lb.config(yscrollcommand=sb.set)

        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Cargar subcarpetas
        subdirs = []
        try:
            for name in sorted(os.listdir(base_dir)):
                p = os.path.join(base_dir, name)
                if os.path.isdir(p):
                    subdirs.append(p)
        except Exception:
            subdirs = []

        for p in subdirs:
            lb.insert(tk.END, os.path.basename(p))

        selected = {"folders": []}

        btns = tk.Frame(win)
        btns.pack(fill=tk.X, padx=10, pady=(6, 10))

        def on_ok():
            idxs = lb.curselection()
            selected["folders"] = [subdirs[i] for i in idxs] if idxs else []
            win.destroy()

        def on_cancel():
            selected["folders"] = []
            win.destroy()

        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.RIGHT, padx=4)
        tk.Button(btns, text="Cancelar", width=10, command=on_cancel).pack(side=tk.RIGHT, padx=4)

        self.wait_window(win)
        return selected["folders"]

    def add_folders(self):
        """
        Nueva versión simplificada:
        - El usuario elige una carpeta en Windows.
        - Se agregan automáticamente TODAS las subcarpetas que contengan imágenes.
        - No se abre ningún diálogo adicional.
        """
        base_dir = filedialog.askdirectory(title="Selecciona una carpeta que contiene carpetas con imágenes")
        if not base_dir:
            return

        # Obtener subcarpetas
        subfolders = []
        try:
            for name in sorted(os.listdir(base_dir)):
                p = os.path.join(base_dir, name)
                if os.path.isdir(p):
                    subfolders.append(p)
        except Exception:
            subfolders = []

        if not subfolders:
            messagebox.showwarning("Sin subcarpetas", "La carpeta seleccionada no contiene subcarpetas.")
            return

        # Filtrar solo las subcarpetas que tienen imágenes válidas
        valid_folders = []
        for folder in subfolders:
            imgs = list_images_in_folder(folder)
            if len(imgs) > 0:
                valid_folders.append(folder)

        if not valid_folders:
            messagebox.showwarning("Sin imágenes", "Ninguna subcarpeta contiene imágenes.")
            return

        # Reemplaza o suma
        self.folder_paths = valid_folders
        self.image_paths = []  # si cargamos carpetas, limpiamos imágenes sueltas

        # Actualizar UI
        self.listbox.delete(0, tk.END)
        for f in valid_folders:
            self.listbox.insert(tk.END, f"[CARPETA] " + os.path.basename(f))

        self._set_progress(0, f"{len(valid_folders)} carpetas listas para procesar.", None, None)
        self.btn_save.config(state=tk.NORMAL)
        self.result = None
        self._clear_previews()

        # Preview rápida
        first_imgs = list_images_in_folder(valid_folders[0])
        if first_imgs:
            first = imread_unicode(first_imgs[0])
            if first is not None:
                self._show_src_preview(first)

    def clear_list(self):
        self.image_paths = []
        self.folder_paths = []
        self.result = None
        self.listbox.delete(0, tk.END)
        self._set_progress(0, "Lista vacía.", None, None)
        self.btn_save.config(state=tk.NORMAL)
        self._clear_previews()

    def _update_autosave_label(self):
        if self.auto_save_dir:
            self.auto_save_label_var.set(f"Guardado automático:\n{self.auto_save_dir}")
        else:
            self.auto_save_label_var.set("Guardado automático: (no seleccionada)")

    # ---------- helpers UI thread-safe ----------

    def _set_progress(self, pct, text=None, src_preview=None, res_preview=None):
        # Cancelación: si el usuario pidió detener, abortar desde hilos de trabajo
        try:
            if getattr(self, "cancel_event", None) and self.cancel_event.is_set():
                if threading.current_thread() is not threading.main_thread():
                    raise ProcessCancelled()
        except ProcessCancelled:
            raise
        except Exception:
            pass

        # NO tocar Tk desde hilos: encolar y que el hilo UI lo aplique
        try:
            self._ui_queue.put_nowait((float(pct), text, src_preview, res_preview))
        except Exception:
            pass

    def _drain_ui_queue(self):
        last = None
        try:
            while True:
                last = self._ui_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            last = None

        if last is not None:
            pct, text, src_preview, res_preview = last
            self.progress_var.set(float(pct))
            self.progress_label.set(f"{int(pct)}%")
            if text is not None:
                self.status.set(text)
            if src_preview is not None:
                self._show_src_preview(src_preview)
            if res_preview is not None:
                self._show_res_preview(res_preview)

        self.after(50, self._drain_ui_queue)

    def _stage_progress_mapper(self, base, span, current, total):
        if total <= 0:
            return base
        return base + span * (current / total)

    # -------------------------------------------

    def stop_all_processes(self):
        # Solicita cancelación cooperativa (los workers abortan en el próximo update de progreso)
        try:
            self.cancel_event.set()
        except Exception:
            return

        try:
            self.status.set("Deteniendo proceso...")
        except Exception:
            pass

        try:
            cur = float(self.progress["value"])
        except Exception:
            cur = 0.0

        try:
            self._set_progress(cur, "Deteniendo proceso...", None, None)
        except Exception:
            pass

    # -------------------------------------------

    def show_about(self):
        about_text = (
            "Open Galileo – Apilamiento avanzado de imágenes científicas y macro.\n\n"
            "Open Galileo es un programa gratuito y de código abierto desarrollado por Brandon Antonio Segura Torres @micro.cosmonauta.\n"
            "Diseñado para el apilamiento de imágenes (focus stacking) y la integración avanzada de señal, orientado a fotografía científica,"
            "microscopia, macrofotografía, astrofotografía y documentación técnica de alta precisión.\n"
            "El software permite combinar múltiples imágenes de una misma escena para obtener\n"
            "máxima nitidez, profundidad de campo extendida y reducción de ruido, incluso cuando\n"
            "cada imagen individual tiene zonas fuera de foco o muy baja señal.\n"
            "\n"
            "Open Galileo nace como un proyecto abierto, educativo y colaborativo, inspirado en la\n"
            "ciencia libre y el acceso democrático a herramientas avanzadas.\n"
            "Es gratuito, sin restricciones, y pensado para crecer con la comunidad.\n"
            "Open Galileo es y siempre será gratuito.\n"
            "Si este programa te resulta útil y deseas apoyar el tiempo, el esfuerzo y la investigación\n"
            "detrás de su desarrollo, puedes hacer una donación voluntaria.\n"
            "Tu aporte ayuda a mantener y mejorar este proyecto abierto.\n"
            "\n"
            "Gracias por apoyar la ciencia accesible.\n"
            "Donaciónes en Argentina:\n"
            "(alias): opengalileo\n"
            "CVU Mercado Pago: 0000003100022518896098\n"
            "Resto del mundo por Paypal: antoniovangritte@gmail.com"
        )

        win = tk.Toplevel(self)
        win.title("About Open Galileo")
        win.configure(bg=self.UI_BG)
        win.resizable(False, False)

        try:
            win.transient(self)
        except Exception:
            pass

        try:
            if hasattr(self, "_app_icon_imgtk") and self._app_icon_imgtk is not None:
                win.iconphoto(True, self._app_icon_imgtk)
        except Exception:
            pass

        body = tk.Frame(win, bg=self.UI_BG, padx=18, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        lbl = tk.Label(
            body,
            text=about_text,
            bg=self.UI_BG,
            fg=self.TXT_FG,
            justify="left",
            anchor="w",
            wraplength=520,
            font=("", 10),
        )
        lbl.pack(fill=tk.BOTH, expand=True)

        btn_ok = tk.Button(
            body,
            text="Aceptar",
            command=win.destroy,
            bg=self.BTN_BG_ENABLED,
            fg=self.BTN_FG_ENABLED,
            activebackground="#6b6b6b",
            activeforeground=self.BTN_FG_ENABLED,
            relief="raised",
            bd=2,
            padx=12,
            pady=2,
        )
        btn_ok.pack(pady=(12, 0))
        try:
            btn_ok.focus_set()
        except Exception:
            pass

        # Centrar ventana
        try:
            win.update_idletasks()
            w = win.winfo_width()
            h = win.winfo_height()
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            x = int((sw - w) / 2)
            y = int((sh - h) / 2)
            win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

        try:
            win.grab_set()
        except Exception:
            pass

        win.protocol("WM_DELETE_WINDOW", win.destroy)

    # -------------------------------------------


    def _disable_all_buttons(self, disabled=True):
        state = tk.DISABLED if disabled else tk.NORMAL

        # Indicador (GIF + cronómetro)
        try:
            if disabled:
                self._start_processing_indicator()
            else:
                self._stop_processing_indicator()
        except Exception:
            pass

        # Al iniciar o terminar, limpiar cancelación pendiente
        try:
            if getattr(self, "cancel_event", None):
                if not disabled:
                    # al finalizar, dejar listo para el próximo run
                    self.cancel_event.clear()
                else:
                    # al comenzar, asegurar estado limpio
                    self.cancel_event.clear()
        except Exception:
            pass

        # Colores según estado
        if disabled:
            btn_bg = self.BTN_BG_DISABLED
            btn_fg = self.BTN_FG_DISABLED
            chk_fg = self.BTN_FG_DISABLED
        else:
            btn_bg = self.BTN_BG_ENABLED
            btn_fg = self.BTN_FG_ENABLED
            chk_fg = self.TXT_FG

        # Botones principales
        for b in (
                self.btn_add,
                self.btn_add_folders,
                self.btn_process,
                self.btn_save,
        ):
            b.config(state=state, bg=btn_bg, fg=btn_fg)

        # Botón detener (queda habilitado durante procesamiento)
        if hasattr(self, "btn_stop"):
            try:
                if disabled:
                    self.btn_stop.config(
                        state=tk.NORMAL,
                        bg=self.BTN_BG_ENABLED,
                        fg=self.BTN_FG_ENABLED,
                        activebackground="#6b6b6b",
                        activeforeground=self.BTN_FG_ENABLED
                    )
                else:
                    self.btn_stop.config(
                        state=tk.DISABLED,
                        bg=self.BTN_BG_DISABLED,
                        fg=self.BTN_FG_DISABLED,
                        activebackground="#6b6b6b",
                        activeforeground=self.BTN_FG_DISABLED
                    )
            except Exception:
                try:
                    self.btn_stop.config(state=tk.NORMAL if disabled else tk.DISABLED)
                except Exception:
                    pass

        # Formato (OptionMenu)
        if hasattr(self, "btn_format"):
            try:
                self.btn_format.config(
                    state=state,
                    bg=btn_bg,
                    fg=btn_fg,
                    activebackground="#6b6b6b",
                    activeforeground=btn_fg,
                )
            except Exception:
                try:
                    self.btn_format.config(state=state)
                except Exception:
                    pass

        # Checkboxes
        for c in (
                self.chk_pmax,
                self.chk_weighted,
                self.chk_depth,
                self.chk_int_mean,
                self.chk_int_median,
                self.chk_int_sum,
                self.chk_int_sigma,
                self.chk_calibration,
                self.chk_fourier,
                self.chk_n2v,
                self.chk_median,
                self.chk_gaussian,
                self.chk_bm3d,
                self.chk_bilateral,
                self.chk_wavelet,
                self.chk_nlm,
                self.chk_serial_filters,
        ):
            c.config(state=state, fg=chk_fg)

        # Sliders también se bloquean mientras se procesa
        if disabled:
            self._set_slider_appearance(self.slider_radius_pmax, self.slider_radius_pmax_val_lbl, False)
            self._set_slider_appearance(self.slider_smooth_pmax, self.slider_smooth_pmax_val_lbl, False)
            self._set_slider_appearance(self.slider_radius_weighted, self.slider_radius_weighted_val_lbl, False)
            self._set_slider_appearance(self.slider_smooth_weighted, self.slider_smooth_weighted_val_lbl, False)
            self._set_slider_appearance(self.slider_radius_depth, self.slider_radius_depth_val_lbl, False)
            self._set_slider_appearance(self.slider_smooth_depth, self.slider_smooth_depth_val_lbl, False)

            if hasattr(self, "slider_bm3d_sigma"):
                self._set_slider_appearance(self.slider_bm3d_sigma, self.slider_bm3d_sigma_val_lbl, False)
            if hasattr(self, "slider_bilateral_d"):
                self._set_slider_appearance(self.slider_bilateral_d, self.slider_bilateral_d_val_lbl, False)
            if hasattr(self, "slider_bilateral_sigma_color"):
                self._set_slider_appearance(self.slider_bilateral_sigma_color,
                                            self.slider_bilateral_sigma_color_val_lbl, False)
            if hasattr(self, "slider_bilateral_sigma_space"):
                self._set_slider_appearance(self.slider_bilateral_sigma_space,
                                            self.slider_bilateral_sigma_space_val_lbl, False)
        else:
            # Volver a estado según casillas
            self._update_slider_state()
            self._update_focus_noise_exclusivity()

    def _load_luna_gif(self, gif_path, size=(30, 30)):
        """Carga luna.gif como frames PhotoImage y guarda duraciones por frame."""
        self._luna_frames = []
        self._luna_durations = []
        try:
            im = Image.open(gif_path)
            base_duration = im.info.get("duration", 80)
            for frame in ImageSequence.Iterator(im):
                fr = frame.convert("RGBA").resize(size, Image.LANCZOS)
                self._luna_frames.append(ImageTk.PhotoImage(fr))

                dur = frame.info.get("duration", base_duration)
                try:
                    dur = int(dur) if dur is not None else int(base_duration)
                except Exception:
                    dur = int(base_duration) if base_duration else 80

                # Evita 0ms o duraciones enormes
                dur = max(20, min(500, dur))
                self._luna_durations.append(dur)

            if not self._luna_frames:
                fr = im.convert("RGBA").resize(size, Image.LANCZOS)
                self._luna_frames = [ImageTk.PhotoImage(fr)]
                self._luna_durations = [80]
        except Exception as e:
            print(f"[WARN] No se pudo cargar luna.gif: {e}")
            self._luna_frames = []
            self._luna_durations = []

    def _start_luna_animation(self):
        if getattr(self, "_luna_animating", False):
            return
        frames = getattr(self, "_luna_frames", None) or []
        if not frames:
            return
        self._luna_animating = True
        self._luna_frame_index = 0
        try:
            self.luna_label.config(image=frames[0])
        except Exception:
            pass
        self._tick_luna_animation()

    def _tick_luna_animation(self):
        if not getattr(self, "_luna_animating", False):
            return
        frames = getattr(self, "_luna_frames", None) or []
        if not frames:
            return

        self._luna_frame_index = (getattr(self, "_luna_frame_index", 0) + 1) % len(frames)
        try:
            self.luna_label.config(image=frames[self._luna_frame_index])
        except Exception:
            pass

        durations = getattr(self, "_luna_durations", None) or []
        dur = durations[self._luna_frame_index] if self._luna_frame_index < len(durations) else 80
        try:
            dur = int(dur)
        except Exception:
            dur = 80
        dur = max(20, min(500, dur))

        try:
            if getattr(self, "_luna_anim_job", None):
                try:
                    self.after_cancel(self._luna_anim_job)
                except Exception:
                    pass
            self._luna_anim_job = self.after(dur, self._tick_luna_animation)
        except Exception:
            self._luna_anim_job = None

    def _stop_luna_animation(self):
        self._luna_animating = False

        job = getattr(self, "_luna_anim_job", None)
        if job:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._luna_anim_job = None

        self._luna_frame_index = 0
        frames = getattr(self, "_luna_frames", None) or []
        if frames:
            try:
                self.luna_label.config(image=frames[0])
            except Exception:
                pass

    def _start_processing_indicator(self):
        if getattr(self, "_processing_indicator_on", False):
            return

        self._processing_indicator_on = True
        self._processing_t0 = time.perf_counter()

        try:
            self.timer_var.set("Tiempo de procesamiento: 00:00")
        except Exception:
            pass

        # Luna: animación solo durante el procesamiento
        try:
            self._start_luna_animation()
        except Exception:
            pass

        try:
            self.processing_info_frame.pack(anchor="w", pady=(4, 0))
        except Exception:
            pass

        self._tick_processing_timer()

    def _tick_processing_timer(self):
        if not getattr(self, "_processing_indicator_on", False):
            return

        t0 = getattr(self, "_processing_t0", None) or time.perf_counter()
        elapsed = int(time.perf_counter() - t0)
        mm = elapsed // 60
        ss = elapsed % 60

        try:
            self.timer_var.set(f"Tiempo de procesamiento: {mm:02d}:{ss:02d}")
        except Exception:
            pass

        try:
            if getattr(self, "_processing_timer_job", None):
                try:
                    self.after_cancel(self._processing_timer_job)
                except Exception:
                    pass
            self._processing_timer_job = self.after(200, self._tick_processing_timer)
        except Exception:
            self._processing_timer_job = None


    def _stop_processing_indicator(self):
        self._processing_indicator_on = False

        job = getattr(self, "_processing_timer_job", None)
        if job:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._processing_timer_job = None
        self._processing_t0 = None

        # Detener animación y volver a primer frame (la luna queda visible)
        try:
            self._stop_luna_animation()
        except Exception:
            pass

        try:
            self.processing_info_frame.pack_forget()
        except Exception:
            pass

    # ---------------- procesamiento single ----------------

    def _on_calibration_toggle(self):
        """
        Handler de la casilla 'Aplicar Darks / Flats / Biases'.
        Si se marca y todavía no hay masters cargados, abre la ventana
        para cargarlos. Si al final no se carga ninguno, se desmarca.
        """
        if self.use_calibration.get():
            ok = self.ensure_calibration_loaded()
            if not ok:
                # Volver a desmarcar si el usuario cancela o no carga nada
                self.use_calibration.set(False)

        # Reglas de bloqueo entre grupos
        self._on_noise_filter_toggle()

    def ensure_calibration_loaded(self):
        """
        Abre una ventana para cargar masters Dark / Flat / Bias si aún no se han cargado.
        - Master Dark: fondo negro (misma exposición/ISO que las lights, lente tapado).
        - Master Flat: campo uniforme (cielo plano / panel, misma óptica/enfoque).
        - Master Bias: exposición mínima, lente tapado.
        Devuelve True si hay al menos un master cargado, False si se cancela o no se carga nada.
        """
        # Si ya hay al menos uno cargado, no pedirlos otra vez
        if any(x is not None for x in (self.master_dark, self.master_flat, self.master_bias)):
            return True

        UI_BG = getattr(self, "UI_BG", "#535353")
        TXT_FG = getattr(self, "TXT_FG", "#f0f0f0")

        win = tk.Toplevel(self)
        win.title("Cargar Darks / Flats / Biases")
        win.transient(self)
        win.grab_set()
        win.configure(bg=UI_BG)
        win.resizable(False, False)

        info = (
            "Carga los masters de calibración (todos opcionales):\n"
            " - Dark: fondo negro, misma exposición/ISO que las imágenes de cielo.\n"
            " - Flat: cielo uniforme o panel plano, misma óptica y enfoque.\n"
            " - Bias: exposición mínima, lente tapado (offset de lectura).\n"
            "Puedes cargar uno, dos o los tres. Como mínimo debe haber uno para aplicar la calibración."
        )
        tk.Label(
            win,
            text=info,
            justify="left",
            anchor="w",
            bg=UI_BG,
            fg=TXT_FG,
            wraplength=520,
        ).pack(fill=tk.X, padx=10, pady=(10, 8))

        filetypes = [
            ("Imágenes",
             "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.nef *.raw *.dng *.arw *.cr2 *.cr3 *.rw2 *.orf *.raf *.pef *.srw"),
            ("Todos", "*.*"),
        ]

        selected = {
            "dark": None,
            "flat": None,
            "bias": None,
        }

        # --- fila helper ---
        def make_row(parent, key, title, desc):
            frame = tk.Frame(parent, bg=UI_BG)
            frame.pack(fill=tk.X, padx=10, pady=4)

            tk.Label(
                frame,
                text=title,
                bg=UI_BG,
                fg=TXT_FG,
                anchor="w",
                font=("", 10, "bold"),
            ).grid(row=0, column=0, sticky="w", columnspan=2)

            tk.Label(
                frame,
                text=desc,
                bg=UI_BG,
                fg=TXT_FG,
                anchor="w",
                wraplength=420,
                justify="left",
            ).grid(row=1, column=0, sticky="w")

            lbl_file = tk.Label(
                frame,
                text="(ningún archivo seleccionado)",
                bg=UI_BG,
                fg="#bbbbbb",
                anchor="w",
                width=40,
            )
            lbl_file.grid(row=2, column=0, sticky="w", pady=(2, 0))

            def choose():
                path = filedialog.askopenfilename(
                    title=f"Selecciona master {key.upper()}",
                    filetypes=filetypes,
                )
                if not path:
                    return
                selected[key] = path
                lbl_file.config(text=os.path.basename(path), fg=TXT_FG)

            btn = tk.Button(
                frame,
                text="Cargar...",
                width=12,
                command=choose,
                bg=self.BTN_BG_ENABLED,
                fg=self.BTN_FG_ENABLED,
                activebackground=self.BTN_BG_ENABLED,
                activeforeground=self.BTN_FG_ENABLED,
            )
            btn.grid(row=2, column=1, sticky="e", padx=(8, 0))

        body = tk.Frame(win, bg=UI_BG)
        body.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        make_row(
            body,
            "dark",
            "Master Dark (fondo negro)",
            "Ruido térmico del sensor. Misma exposición/ISO que las luces, lente/tapa cubriendo la entrada de luz.",
        )
        make_row(
            body,
            "flat",
            "Master Flat (luz plana)",
            "Corrige viñeteo y motas de polvo. Cielo uniforme / panel iluminado, misma posición y enfoque que las luces.",
        )
        make_row(
            body,
            "bias",
            "Master Bias (offset)",
            "Frames de exposición mínima, lente tapado. Corrige el offset de lectura del sensor.",
        )

        # Botones OK / Cancelar
        btn_frame = tk.Frame(win, bg=UI_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        result = {"accepted": False}

        def on_ok():
            result["accepted"] = True
            win.destroy()

        def on_cancel():
            result["accepted"] = False
            win.destroy()

        tk.Button(
            btn_frame,
            text="Cancelar",
            width=10,
            command=on_cancel,
            bg=self.BTN_BG_ENABLED,
            fg=self.BTN_FG_ENABLED,
        ).pack(side=tk.RIGHT, padx=4)
        tk.Button(
            btn_frame,
            text="Aceptar",
            width=10,
            command=on_ok,
            bg=self.BTN_BG_ENABLED,
            fg=self.BTN_FG_ENABLED,
        ).pack(side=tk.RIGHT, padx=4)

        self.wait_window(win)

        if not result["accepted"]:
            return False

        # Helper para cargar las imágenes
        def load_master(path, label):
            if not path:
                return None
            img = imread_unicode(path)
            if img is None:
                messagebox.showerror("Error", f"No se pudo cargar el master {label}:\n{path}")
                return None
            return img.astype(np.float32)

        self.master_dark = load_master(selected["dark"], "DARK")
        self.master_flat = load_master(selected["flat"], "FLAT")
        self.master_bias = load_master(selected["bias"], "BIAS")

        if not any(x is not None for x in (self.master_dark, self.master_flat, self.master_bias)):
            messagebox.showwarning(
                "Calibración",
                "No se cargó ningún master de calibración.\n"
                "La calibración D/F/B no se aplicará."
            )
            return False

        return True

    def apply_calibration(self, img_bgr):
        """
        Aplica la corrección básica con masters Darks / Flats / Biases:
        img_cal = (img - bias - dark) / flat_normalizado
        """
        img = img_bgr.astype(np.float32)
        h, w = img.shape[:2]

        def resize_if_needed(master):
            if master is None:
                return None
            if master.shape[:2] != (h, w):
                return cv2.resize(master, (w, h), interpolation=cv2.INTER_LINEAR)
            return master

        dark = resize_if_needed(self.master_dark)
        flat = resize_if_needed(self.master_flat)
        bias = resize_if_needed(self.master_bias)

        # Restar Bias y Dark si existen
        if bias is not None:
            img = img - bias
        if dark is not None:
            img = img - dark

        # Corregir por Flat normalizado
        if flat is not None:
            eps = 1e-6
            mean_flat = np.mean(flat)
            if mean_flat < eps:
                mean_flat = eps
            flat_norm = flat / mean_flat
            img = img / (flat_norm + eps)

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def process_stack(self):
        if len(self.image_paths) < 2:
            messagebox.showwarning("Faltan imágenes", "Agrega al menos 2 imágenes.")
            return

        self._disable_all_buttons(True)
        self._set_progress(0, "Procesando stack (PMax)...", None, None)

        def worker():
            try:
                def ui_cb(stage, current, total, preview):
                    # Etapas (%) para single PMax
                    if stage == "load":
                        self._set_progress(10, "Cargando imágenes...", preview, None)
                    elif stage == "align":
                        pct = self._stage_progress_mapper(10, 30, current, total)
                        self._set_progress(pct, f"Alineando {current}/{total}...", preview, None)
                    elif stage == "focus":
                        pct = self._stage_progress_mapper(40, 25, current, total)
                        self._set_progress(pct, f"Mapa de foco {current}/{total}...", preview, None)
                    elif stage == "pyramid":
                        pct = self._stage_progress_mapper(65, 20, current, total)
                        self._set_progress(pct, f"Fusionando pirámide {current}/{total}...", None, preview)
                    elif stage == "post":
                        self._set_progress(85, "Post-procesado clarity-like...", None, preview)

                result = stack_paths(self.image_paths, progress_cb=ui_cb)

                self.result = result
                self._set_progress(100, "Listo (PMax). Guardando automáticamente...", None, result)

                # Guardado automático si hay carpeta configurada
                if self.auto_save_dir:
                    base = "resultado"
                    out_path = os.path.join(self.auto_save_dir, base + ".jpg")
                    k = 1
                    while os.path.exists(out_path):
                        out_path = os.path.join(self.auto_save_dir, f"{base}_{k}.jpg")
                        k += 1
                    self._save_jpg(out_path, result)
                    self.status.set(f"Guardado automático en: {out_path}")
                else:
                    self.status.set("Procesado completo. Selecciona carpeta con 'Guardar JPG'.")

                self.btn_save.config(state=tk.NORMAL)

            except ProcessCancelled:
                try:
                    self.cancel_event.clear()
                except Exception:
                    pass
                self._set_progress(0, "Proceso detenido por el usuario.", None, None)
                try:
                    self.status.set("Proceso detenido por el usuario.")
                except Exception:
                    pass

            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_progress(0, "Error en el proceso.", None, None)
            finally:
                self.after(0, lambda: self._disable_all_buttons(False))
        threading.Thread(target=worker, daemon=True).start()

    def process_stack_weighted(self):
        if len(self.image_paths) < 2:
            messagebox.showwarning("Faltan imágenes", "Agrega al menos 2 imágenes.")
            return

        self._disable_all_buttons(True)
        self._set_progress(0, "Procesando stack (Promedio ponderado)...", None, None)

        def worker():
            try:
                def ui_cb(stage, current, total, preview):
                    # Etapas (%) para promedio ponderado
                    if stage == "load":
                        self._set_progress(10, "Cargando imágenes...", preview, None)
                    elif stage == "align":
                        pct = self._stage_progress_mapper(10, 40, current, total)
                        self._set_progress(pct, f"Alineando {current}/{total}...", preview, None)
                    elif stage == "focusW":
                        pct = self._stage_progress_mapper(50, 20, current, total)
                        self._set_progress(pct, f"Calculando pesos {current}/{total}...", preview, None)
                    elif stage == "fusionW":
                        pct = self._stage_progress_mapper(70, 25, current, total)
                        self._set_progress(pct, f"Promediando {current}/{total}...", None, preview)

                result = stack_paths_weighted(self.image_paths, progress_cb=ui_cb)

                self.result = result
                self._set_progress(100, "Listo (Promedio). Guardando automáticamente...", None, result)

                if self.auto_save_dir:
                    base = "resultado"
                    out_path = self._unique_out_path(self.auto_save_dir, base)
                    self._save_image(out_path, result)
                    self.status.set(f"Guardado automático (Promedio) en: {out_path}")

                else:
                    self.status.set("Procesado completo (Promedio). Selecciona carpeta con 'Guardar JPG'.")
                self.btn_save.config(state=tk.NORMAL)

            except ProcessCancelled:
                try:
                    self.cancel_event.clear()
                except Exception:
                    pass
                self._set_progress(0, "Proceso detenido por el usuario.", None, None)

                try:
                    self.status.set("Proceso detenido por el usuario.")
                except Exception:
                    pass

            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_progress(0, "Error en el proceso (Promedio).", None, None)
            finally:
                self.after(0, lambda: self._disable_all_buttons(False))
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- procesamiento batch por carpetas ----------------

    def apilar_imagenes(self):
        # Si hay carpetas cargadas, procesarlas en batch
        if self.folder_paths:
            self.process_folders()
            return

        # Debe haber al menos una imagen cargada
        if not self.image_paths:
            messagebox.showwarning("Sin elementos", "Agrega imágenes o carpetas.")
            return

        # --- ADVERTENCIA POR ARCHIVOS RAW ---
        has_raw = any(
            os.path.splitext(p)[1].lower() in RAW_EXTS
            for p in self.image_paths
        )

        if has_raw:
            messagebox.showwarning(
                "Procesamiento de archivos RAW",
                "Se detectaron archivos RAW (NEF/DNG/ARW/CR2/RAW, etc.).\n\n"
                "El procesamiento de imágenes RAW implica una decodificación completa "
                "del sensor y un uso intensivo de CPU y memoria.\n\n"
                "Durante esta etapa el programa puede parecer no responder, "
                "pero el procesamiento continúa en segundo plano.\n\n"
                "No cierre la aplicación y permita que el proceso finalice."
            )

        # Métodos de apilado seleccionados
        methods = []

        # Focus tracking
        if self.use_pmax.get():
            methods.append("pmax")
        if self.use_weighted.get():
            methods.append("weighted")
        if self.use_depth.get():
            methods.append("depth")

        # Integración
        if self.use_int_mean.get():
            methods.append("int_mean")
        if self.use_int_median.get():
            methods.append("int_median")
        if self.use_int_sum.get():
            methods.append("int_sum")
        if self.use_int_sigma.get():
            methods.append("int_sigma")

        # Filtros de ruido seleccionados (se aplican a TODAS las imágenes cargadas)

        noise_filters_selected = (
                self.use_calibration.get()
                or self.use_fourier.get()
                or self.use_n2v.get()
                or self.use_median.get()
                or self.use_gaussian.get()
                or self.use_bm3d.get()
                or self.use_bilateral.get()
                or self.use_wavelet.get()
                or self.use_nlm.get()
        )

        # Modo: SOLO filtros (sin apilamiento)
        filters_only_mode = (noise_filters_selected and not methods)

        # Si hay métodos de apilado, se requieren 2+ imágenes
        if methods and len(self.image_paths) < 2:
            messagebox.showwarning("Faltan imágenes", "Agrega al menos 2 imágenes para apilar.")
            return

        # Si no hay métodos y tampoco filtros, no hay nada que hacer
        if not methods and not filters_only_mode:
            messagebox.showwarning(
                "Sin método",
                "Selecciona al menos un método de apilado o activa al menos un filtro de ruido."
            )
            return

        # Si se va a usar calibración, asegurar que los masters estén cargados
        if self.use_calibration.get():
            if not self.ensure_calibration_loaded():
                return

        # Asegurar ruta de guardado: si no hay, pedirla aquí mismo
        if not self.auto_save_dir:
            folder = filedialog.askdirectory(title="Selecciona carpeta donde se guardarán los resultados")
            if not folder:
                return
            self.auto_save_dir = folder
            self._update_autosave_label()

        # ---------------- MODO: aplicar filtros a TODAS las imágenes (sin apilar) ----------------
        if filters_only_mode:
            self._disable_all_buttons(True)
            self._set_progress(0, "Aplicando filtros a todas las imágenes...", None, None)

            def worker_filters():
                try:
                    # Lista de filtros disponibles (tag -> callable / None en BM3D)
                    filter_map = {}

                    if self.use_calibration.get():
                        filter_map["calibrado"] = lambda im: self.apply_calibration(im)

                    if self.use_fourier.get():
                        filter_map["fourier"] = lambda im: apply_fourier_filter(
                            im,
                            cutoff_ratio=FOURIER_CUTOFF_RATIO,
                            soften_ratio=FOURIER_SOFTEN_RATIO,
                        )

                    if self.use_n2v.get():
                        filter_map["noise2void"] = lambda im: apply_noise2void_denoising(
                            im,
                            sigma=N2V_SIGMA,
                            thr_mult=N2V_THR_MULT,
                            iterations=N2V_ITERATIONS,
                        )

                    if self.use_median.get():
                        filter_map["median"] = lambda im: apply_median_filter(
                            im,
                            ksize=MEDIAN_KSIZE,
                        )

                    if self.use_gaussian.get():
                        filter_map["gaussian"] = lambda im: apply_gaussian_filter(
                            im,
                            sigma=GAUSSIAN_SIGMA,
                            tile_size=GAUSSIAN_TILE_SIZE,
                        )

                    if self.use_bm3d.get():
                        # Se maneja especial para tener progreso interno real
                        filter_map["bm3d"] = None

                    if self.use_bilateral.get():
                        filter_map["bilateral"] = lambda im: apply_bilateral_filter(
                            im,
                            d=BILATERAL_D,
                            sigma_color=BILATERAL_SIGMA_COLOR,
                            sigma_space=BILATERAL_SIGMA_SPACE,
                        )

                    if self.use_wavelet.get():
                        filter_map["wavelet"] = lambda im: apply_wavelet_denoising(
                            im,
                            level=WAVELET_LEVEL,
                        )

                    if self.use_nlm.get():
                        filter_map["nlm"] = lambda im: apply_nlm_denoising(
                            im,
                            h=NLM_H,
                            hColor=NLM_H_COLOR,
                            templateWindowSize=NLM_TEMPLATE_WINDOW,
                            searchWindowSize=NLM_SEARCH_WINDOW,
                        )

                    if not filter_map:
                        raise RuntimeError("No hay filtros seleccionados.")

                    # ---------------- MODO SERIADO: aplica filtros 1->2->3 sobre la MISMA imagen ----------------
                    if self.use_serial_filters.get():
                        ordered_tags = []

                        # 1) primero: orden elegido por el usuario
                        for t in self.serial_filter_order:
                            if t in filter_map and t not in ordered_tags:
                                ordered_tags.append(t)

                        # 2) luego: cualquier filtro marcado que no haya quedado en la lista (orden definido)
                        for t, var, chk, base_text in getattr(self, "_serial_filter_defs", []) or []:
                            if t in filter_map and t not in ordered_tags:
                                ordered_tags.append(t)

                        if not ordered_tags:
                            raise RuntimeError("No hay filtros seleccionados.")

                        total_jobs = max(1, len(self.image_paths) * len(ordered_tags))
                        done = 0
                        last_out = None

                        suffix = "_".join(ordered_tags)

                        for p in self.image_paths:
                            img = imread_unicode(p)
                            if img is None:
                                done += len(ordered_tags)
                                pct = 100.0 * done / total_jobs
                                self._set_progress(
                                    pct,
                                    f"Saltando (no se pudo leer): {os.path.basename(p)}",
                                    None,
                                    None
                                )
                                continue

                            base_in = os.path.splitext(os.path.basename(p))[0]
                            out = img.copy()

                            for tag in ordered_tags:
                                # Progreso base/span de ESTE paso (imagen+filtro)
                                job_base = 100.0 * done / total_jobs
                                job_span = 100.0 / total_jobs

                                if tag == "bm3d":
                                    def _bm3d_cb(stage_text, cur, tot, preview, _jb=job_base, _js=job_span):
                                        try:
                                            frac = 0.0 if tot <= 0 else float(cur) / float(tot)
                                        except Exception:
                                            frac = 0.0
                                        pct_local = _jb + _js * max(0.0, min(1.0, frac))
                                        self._set_progress(
                                            pct_local,
                                            stage_text,
                                            img,
                                            preview
                                        )

                                    out = apply_bm3d_denoising(
                                        out.copy(),
                                        sigma_psd=BM3D_SIGMA_PSD,
                                        progress_cb=_bm3d_cb,
                                    )

                                else:
                                    fn = filter_map.get(tag)
                                    out = fn(out.copy())

                                last_out = out

                                done += 1
                                pct = 100.0 * done / total_jobs
                                self._set_progress(
                                    pct,
                                    f"Aplicando {tag}: {os.path.basename(p)} ({done}/{total_jobs})",
                                    img,
                                    out
                                )

                            # Guardar SOLO el resultado final con todos los filtros en el nombre
                            base_name = f"{base_in}_{suffix}"
                            out_path = self._unique_out_path(self.auto_save_dir, base_name)
                            self._save_image(out_path, last_out)

                        self.result = last_out
                        self._set_progress(
                            100,
                            f"Listo. Archivos generados en: {self.auto_save_dir}",
                            None,
                            last_out
                        )
                        return

                    # ---------------- MODO NORMAL: cada filtro se aplica sobre la imagen ORIGINAL (1 salida por filtro) ----------------
                    selected_filters = [(tag, fn) for tag, fn in [(k, v) for k, v in filter_map.items()]]

                    total_jobs = max(1, len(self.image_paths) * len(selected_filters))
                    done = 0
                    last_out = None

                    for p in self.image_paths:
                        img = imread_unicode(p)
                        if img is None:
                            # avanzar igual para no “congelar” el progreso
                            done += len(selected_filters)
                            pct = 100.0 * done / total_jobs
                            self._set_progress(
                                pct,
                                f"Saltando (no se pudo leer): {os.path.basename(p)}",
                                None,
                                None
                            )
                            continue

                        base_in = os.path.splitext(os.path.basename(p))[0]

                        for tag, fn in selected_filters:
                            # Progreso base/span de ESTE job (imagen+filtro)
                            job_base = 100.0 * done / total_jobs
                            job_span = 100.0 / total_jobs

                            if tag == "bm3d":
                                def _bm3d_cb(stage_text, cur, tot, preview, _jb=job_base, _js=job_span):
                                    try:
                                        frac = 0.0 if tot <= 0 else float(cur) / float(tot)
                                    except Exception:
                                        frac = 0.0
                                    pct_local = _jb + _js * max(0.0, min(1.0, frac))
                                    self._set_progress(
                                        pct_local,
                                        stage_text,
                                        img,
                                        preview
                                    )

                                out = apply_bm3d_denoising(
                                    img.copy(),
                                    sigma_psd=BM3D_SIGMA_PSD,
                                    progress_cb=_bm3d_cb,
                                )

                            else:
                                out = fn(img.copy())

                            last_out = out

                            base_name = f"{base_in}_{tag}"
                            out_path = self._unique_out_path(self.auto_save_dir, base_name)
                            self._save_image(out_path, out)

                            done += 1
                            pct = 100.0 * done / total_jobs
                            self._set_progress(
                                pct,
                                f"Aplicando {tag}: {os.path.basename(p)} ({done}/{total_jobs})",
                                img,
                                out
                            )

                    self.result = last_out
                    self._set_progress(
                        100,
                        f"Listo. Archivos generados en: {self.auto_save_dir}",
                        None,
                        last_out
                    )

                except ProcessCancelled:
                    try:
                        self.cancel_event.clear()
                    except Exception:
                        pass
                    self._set_progress(0, "Proceso detenido por el usuario.", None, None)
                    try:
                        self.status.set("Proceso detenido por el usuario.")
                    except Exception:
                        pass

                except Exception as e:
                    self.after(0, lambda err=str(e): messagebox.showerror("Error", err))
                    self._set_progress(0, "Error al aplicar filtros a las imágenes.", None, None)
                finally:
                    self.after(0, lambda: self._disable_all_buttons(False))
            threading.Thread(target=worker_filters, daemon=True).start()
            return

        # ---------------- MODO: apilado con 2+ imágenes ----------------
        self._disable_all_buttons(True)
        self._set_progress(0, "Iniciando apilamiento...", None, None)

        def worker():
            try:
                total_methods = len(methods)
                last_result = None

                for idx, m in enumerate(methods, start=1):
                    prefix = f"[{idx}/{total_methods}]"

                    if m == "pmax":
                        method_tag = "pmax"

                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 30, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "focus":
                                pct = self._stage_progress_mapper(35, 25, current, total)
                                self._set_progress(pct, f"{prefix} Mapa de foco {current}/{total}...", preview, None)
                            elif stage == "pyramid":
                                pct = self._stage_progress_mapper(60, 20, current, total)
                                self._set_progress(
                                    pct,
                                    f"{prefix} Fusionando pirámide {current}/{total}...",
                                    None,
                                    preview
                                )
                            elif stage == "post":
                                self._set_progress(85, f"{prefix} Post-procesado clarity-like...", None, preview)

                        # Para que stack_paths guarde RAW en carpeta
                        FocusStackerApp.current_auto_save_dir = self.auto_save_dir
                        FocusStackerApp.current_output_format = (self.output_format_var.get() or "jpg").strip().lower()
                        FocusStackerApp.current_raw_name = "resultado_pmax_RAW"

                        result = stack_paths(self.image_paths, progress_cb=ui_cb)

                    elif m == "weighted":
                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 40, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "focusW":
                                pct = self._stage_progress_mapper(45, 20, current, total)
                                self._set_progress(pct, f"{prefix} Calculando pesos {current}/{total}...", preview,
                                                   None)
                            elif stage == "fusionW":
                                pct = self._stage_progress_mapper(65, 25, current, total)
                                self._set_progress(pct, f"{prefix} Promediando {current}/{total}...", None, preview)

                        result = stack_paths_weighted(self.image_paths, progress_cb=ui_cb)
                        method_tag = "promedio"

                    elif m == "depth":
                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 40, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "focusD":
                                pct = self._stage_progress_mapper(45, 20, current, total)
                                self._set_progress(pct, f"{prefix} Calculando mapa de foco {current}/{total}...",
                                                   preview, None)
                            elif stage == "fusionD":
                                pct = self._stage_progress_mapper(70, 25, current, total)
                                self._set_progress(pct, f"{prefix} Fusionando según profundidad {current}/{total}...",
                                                   None, preview)

                        result = stack_paths_depth(self.image_paths, progress_cb=ui_cb)
                        method_tag = "depth"

                    elif m == "int_mean":
                        method_tag = "mean"

                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 55, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "integrate":
                                self._set_progress(70, f"{prefix} Integrando (mean)...", None, preview)

                        result = stack_paths_integration_mean(self.image_paths, progress_cb=ui_cb)

                    elif m == "int_median":
                        method_tag = "median"

                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 55, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "integrate":
                                self._set_progress(70, f"{prefix} Integrando (median)...", None, preview)

                        result = stack_paths_integration_median(self.image_paths, progress_cb=ui_cb)

                    elif m == "int_sum":
                        method_tag = "sum"

                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 55, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "integrate":
                                self._set_progress(70, f"{prefix} Integrando (sum)...", None, preview)

                        result = stack_paths_integration_sum(self.image_paths, progress_cb=ui_cb)

                    elif m == "int_sigma":
                        method_tag = "sigma"

                        def ui_cb(stage, current, total, preview):
                            if stage == "load":
                                self._set_progress(5, f"{prefix} Cargando imágenes...", preview, None)
                            elif stage == "align":
                                pct = self._stage_progress_mapper(5, 55, current, total)
                                self._set_progress(pct, f"{prefix} Alineando {current}/{total}...", preview, None)
                            elif stage == "integrate":
                                self._set_progress(70, f"{prefix} Integrando (sigma-clipping)...", None, preview)

                        result = stack_paths_integration_sigma_clipping(self.image_paths, progress_cb=ui_cb)
                    else:
                        raise RuntimeError(f"Método desconocido: {m}")

                    # NO aplicar filtros de ruido al resultado apilado.
                    # Los filtros de ruido (calibración / bilateral / wavelet / nlm) se aplican
                    # sobre las imágenes cargadas y generan archivos nuevos por separado.

                    # Guardar resultado de este método
                    self.result = result
                    last_result = result

                    if self.auto_save_dir:
                        base = f"resultado_{method_tag}"
                        out_path = self._unique_out_path(self.auto_save_dir, base)
                        self._save_image(out_path, result)
                        self.status.set(f"{prefix} Guardado automático ({method_tag}) en: {out_path}")
                    else:
                        self.status.set(f"{prefix} Procesado completo ({method_tag}).")

                if last_result is not None:
                    self._set_progress(100, "Apilamientos completados.", None, last_result)
                else:
                    self._set_progress(0, "No se generó ningún resultado.", None, None)

                self.btn_save.config(state=tk.NORMAL)


            except ProcessCancelled:
                try:
                    self.cancel_event.clear()
                except Exception:
                    pass
                self._set_progress(0, "Proceso detenido por el usuario.", None, None)
                try:
                    self.status.set("Proceso detenido por el usuario.")
                except Exception:
                    pass

            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_progress(0, "Error en el proceso de apilado.", None, None)
            finally:
                self.after(0, lambda: self._disable_all_buttons(False))
        threading.Thread(target=worker, daemon=True).start()

    def process_stack_depth(self):
        if len(self.image_paths) < 2:
            messagebox.showwarning("Faltan imágenes", "Agrega al menos 2 imágenes.")
            return

        self._disable_all_buttons(True)
        self._set_progress(0, "Procesando stack (Mapa de profundidad)...", None, None)

        def worker():
            try:
                def ui_cb(stage, current, total, preview):
                    # Etapas (%) para mapa de profundidad
                    if stage == "load":
                        self._set_progress(10, "Cargando imágenes...", preview, None)
                    elif stage == "align":
                        pct = self._stage_progress_mapper(10, 40, current, total)
                        self._set_progress(pct, f"Alineando {current}/{total}...", preview, None)
                    elif stage == "focusD":
                        pct = self._stage_progress_mapper(40, 20, current, total)
                        self._set_progress(pct, f"Calculando mapa de foco {current}/{total}...", preview, None)
                    elif stage == "refineD":
                        self._set_progress(65, "Suavizando mapa de profundidad...", None, preview)
                    elif stage == "fusionD":
                        pct = self._stage_progress_mapper(65, 30, current, total)
                        self._set_progress(pct, f"Fusionando según profundidad {current}/{total}...", None, preview)

                result = stack_paths_depth(self.image_paths, progress_cb=ui_cb)

                self.result = result
                self._set_progress(100, "Listo (Mapa profundidad). Guardando automáticamente...", None, result)

                if self.auto_save_dir:
                    base = "resultado"
                    out_path = self._unique_out_path(self.auto_save_dir, base)
                    self._save_image(out_path, result)
                    self.status.set(f"Guardado automático (Mapa profundidad) en: {out_path}")

                else:
                    self.status.set("Procesado completo (Mapa profundidad). Selecciona carpeta con 'Guardar JPG'.")
                self.btn_save.config(state=tk.NORMAL)

            except ProcessCancelled:
                try:
                    self.cancel_event.clear()
                except Exception:
                    pass
                self._set_progress(0, "Proceso detenido por el usuario.", None, None)

                try:
                    self.status.set("Proceso detenido por el usuario.")
                except Exception:
                    pass

            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_progress(0, "Error en el proceso (Mapa profundidad).", None, None)
            finally:
                self.after(0, lambda: self._disable_all_buttons(False))
        threading.Thread(target=worker, daemon=True).start()

    def process_folders(self):
        if len(self.folder_paths) == 0:
            messagebox.showwarning("Faltan carpetas", "Agrega al menos una carpeta.")
            return

        methods = []

        # Focus tracking
        if self.use_pmax.get():
            methods.append("pmax")
        if self.use_weighted.get():
            methods.append("weighted")
        if self.use_depth.get():
            methods.append("depth")

        # Integración
        if self.use_int_mean.get():
            methods.append("int_mean")
        if self.use_int_median.get():
            methods.append("int_median")
        if self.use_int_sum.get():
            methods.append("int_sum")
        if self.use_int_sigma.get():
            methods.append("int_sigma")

        # Filtros de ruido seleccionados
        noise_filters_selected = (
                self.use_calibration.get()
                or self.use_fourier.get()
                or self.use_n2v.get()
                or self.use_median.get()
                or self.use_gaussian.get()
                or self.use_bm3d.get()
                or self.use_bilateral.get()
                or self.use_wavelet.get()
                or self.use_nlm.get()
        )

        # Modo: SOLO filtros (sin apilamiento)
        filters_only_mode = (noise_filters_selected and not methods)

        if not methods and not filters_only_mode:
            messagebox.showwarning(
                "Sin método",
                "Selecciona al menos un método de apilado o activa al menos un filtro de ruido."
            )
            return

        # Si se va a usar calibración, cargar masters una sola vez
        if self.use_calibration.get():
            if not self.ensure_calibration_loaded():
                return

        out_root = self.auto_save_dir
        if not out_root:
            out_root = filedialog.askdirectory(title="Selecciona carpeta destino para resultados")
            if not out_root:
                return
            self.auto_save_dir = out_root
            self._update_autosave_label()

        # ---------------- MODO: aplicar filtros a TODAS las imágenes de TODAS las carpetas ----------------
        if filters_only_mode:
            folders = list(self.folder_paths)

            # Orden fijo de tags (mismo orden que en modo imágenes sueltas)
            order_tags = [
                ("calibrado", self.use_calibration),
                ("fourier", self.use_fourier),
                ("noise2void", self.use_n2v),
                ("median", self.use_median),
                ("gaussian", self.use_gaussian),
                ("bm3d", self.use_bm3d),
                ("bilateral", self.use_bilateral),
                ("wavelet", self.use_wavelet),
                ("nlm", self.use_nlm),
            ]
            selected_tags = [tag for tag, var in order_tags if var.get()]
            if not selected_tags:
                messagebox.showwarning(
                    "Sin filtros",
                    "Activa al menos un filtro de ruido para procesar las carpetas."
                )
                return

            # Precontar imágenes para progreso (y evitar total_jobs=0)
            folder_to_imgs = []
            total_imgs = 0
            for folder in folders:
                imgs = list_images_in_folder(folder)
                folder_to_imgs.append((folder, imgs))
                total_imgs += len(imgs)

            if total_imgs == 0:
                messagebox.showwarning("Sin imágenes", "Ninguna carpeta contiene imágenes válidas.")
                return

            total_jobs = total_imgs * len(selected_tags)
            if total_jobs <= 0:
                messagebox.showwarning("Sin trabajo", "No hay imágenes/filtros para procesar.")
                return

            self._disable_all_buttons(True)
            self._set_progress(0, "Aplicando filtros a imágenes en carpetas.", None, None)

            def worker():
                try:
                    # Lista de filtros disponibles (tag -> callable / None en BM3D)
                    filter_map = {}

                    if self.use_calibration.get():
                        filter_map["calibrado"] = lambda im: self.apply_calibration(im)

                    if self.use_fourier.get():
                        filter_map["fourier"] = lambda im: apply_fourier_filter(
                            im,
                            cutoff_ratio=FOURIER_CUTOFF_RATIO,
                            soften_ratio=FOURIER_SOFTEN_RATIO,
                        )

                    if self.use_n2v.get():
                        filter_map["noise2void"] = lambda im: apply_noise2void_denoising(
                            im,
                            sigma=N2V_SIGMA,
                            thr_mult=N2V_THR_MULT,
                            iterations=N2V_ITERATIONS,
                        )

                    if self.use_median.get():
                        filter_map["median"] = lambda im: apply_median_filter(
                            im,
                            ksize=MEDIAN_KSIZE,
                        )

                    if self.use_gaussian.get():
                        filter_map["gaussian"] = lambda im: apply_gaussian_filter(
                            im,
                            sigma=GAUSSIAN_SIGMA,
                            tile_size=GAUSSIAN_TILE_SIZE,
                        )

                    if self.use_bm3d.get():
                        # Se maneja especial para tener progreso interno real
                        filter_map["bm3d"] = None

                    if self.use_bilateral.get():
                        filter_map["bilateral"] = lambda im: apply_bilateral_filter(
                            im,
                            d=BILATERAL_D,
                            sigma_color=BILATERAL_SIGMA_COLOR,
                            sigma_space=BILATERAL_SIGMA_SPACE,
                        )

                    if self.use_wavelet.get():
                        filter_map["wavelet"] = lambda im: apply_wavelet_denoising(
                            im,
                            level=WAVELET_LEVEL,
                        )

                    if self.use_nlm.get():
                        filter_map["nlm"] = lambda im: apply_nlm_denoising(
                            im,
                            h=NLM_H,
                            hColor=NLM_H_COLOR,
                            templateWindowSize=NLM_TEMPLATE_WINDOW,
                            searchWindowSize=NLM_SEARCH_WINDOW,
                        )

                    if not filter_map:
                        raise RuntimeError("No hay filtros seleccionados.")

                    # Re-ordenar según orden fijo
                    ordered_tags = [tag for tag, _ in order_tags if tag in filter_map]

                    done = 0
                    last_result = None
                    total_folders = len(folder_to_imgs)

                    for f_idx, (folder, img_paths) in enumerate(folder_to_imgs, start=1):
                        folder_name = os.path.basename(folder.rstrip("/\\"))
                        out_folder = os.path.join(out_root, f"{folder_name}_filtros")
                        try:
                            os.makedirs(out_folder, exist_ok=True)
                        except Exception:
                            out_folder = out_root  # fallback

                        if not img_paths:
                            self._set_progress(
                                100.0 * done / total_jobs,
                                f"[{f_idx}/{total_folders}] Carpeta sin imágenes: {folder_name}",
                                None,
                                None
                            )
                            continue

                        for p in img_paths:
                            img = imread_unicode(p)
                            base_in = os.path.splitext(os.path.basename(p))[0]

                            if img is None:
                                # Avanzar como si se hubieran hecho los jobs de esta imagen
                                done += len(ordered_tags)
                                self._set_progress(
                                    100.0 * done / total_jobs,
                                    f"[{f_idx}/{total_folders}] Error leyendo: {os.path.basename(p)}",
                                    None,
                                    None
                                )
                                continue

                            # Preview de origen
                            self._set_progress(
                                100.0 * done / total_jobs,
                                f"[{f_idx}/{total_folders}] Procesando: {os.path.basename(p)}",
                                img,
                                None
                            )

                            # ---------------- MODO SERIADO: aplica filtros 1->2->3 sobre la MISMA imagen ----------------
                            if self.use_serial_filters.get():
                                img_work = img.copy()

                                for step_idx, tag in enumerate(ordered_tags, start=1):
                                    job_base = 100.0 * done / total_jobs
                                    job_span = 100.0 / total_jobs

                                    if tag == "bm3d":
                                        def cb(current, total, preview):
                                            pct = job_base + job_span * (0.20 + 0.80 * (current / max(1, total)))
                                            self._set_progress(
                                                pct,
                                                f"[{f_idx}/{total_folders}] {base_in} ({step_idx}/{len(ordered_tags)}): BM3D {current}/{total}",
                                                None,
                                                preview
                                            )
                                        img_work = apply_bm3d_denoising(img_work, progress_cb=cb)
                                    else:
                                        img_work = filter_map[tag](img_work)

                                    done += 1
                                    last_result = img_work
                                    self.result = img_work

                                    self._set_progress(
                                        100.0 * done / total_jobs,
                                        f"[{f_idx}/{total_folders}] {base_in} ({step_idx}/{len(ordered_tags)}): {tag}",
                                        None,
                                        img_work
                                    )

                                suffix = "_".join(ordered_tags)
                                base_name = f"{base_in}_{suffix}"
                                out_path = self._unique_out_path(out_folder, base_name)
                                self._save_image(out_path, img_work)

                                self._set_progress(
                                    100.0 * done / total_jobs,
                                    f"[{f_idx}/{total_folders}] Guardado: {os.path.basename(out_path)}",
                                    None,
                                    img_work
                                )

                            # ---------------- MODO NO SERIADO: aplica cada filtro sobre la imagen ORIGINAL ----------------
                            else:
                                for tag_idx, tag in enumerate(ordered_tags, start=1):
                                    job_base = 100.0 * done / total_jobs
                                    job_span = 100.0 / total_jobs

                                    if tag == "bm3d":
                                        def cb(current, total, preview):
                                            pct = job_base + job_span * (0.20 + 0.80 * (current / max(1, total)))
                                            self._set_progress(
                                                pct,
                                                f"[{f_idx}/{total_folders}] {base_in} ({tag_idx}/{len(ordered_tags)}): BM3D {current}/{total}",
                                                None,
                                                preview
                                            )
                                        out = apply_bm3d_denoising(img, progress_cb=cb)
                                    else:
                                        out = filter_map[tag](img)

                                    done += 1
                                    last_result = out
                                    self.result = out

                                    base_name = f"{base_in}_{tag}"
                                    out_path = self._unique_out_path(out_folder, base_name)
                                    self._save_image(out_path, out)

                                    self._set_progress(
                                        100.0 * done / total_jobs,
                                        f"[{f_idx}/{total_folders}] Guardado: {os.path.basename(out_path)}",
                                        None,
                                        out
                                    )

                    if last_result is not None:
                        self._set_progress(100, "Procesamiento de filtros en carpetas terminado.", None, last_result)
                    else:
                        self._set_progress(100, "Procesamiento de filtros en carpetas terminado (sin resultados válidos).", None, None)

                    self.btn_save.config(state=tk.NORMAL)

                except ProcessCancelled:
                    try:
                        self.cancel_event.clear()
                    except Exception:
                        pass
                    self._set_progress(0, "Proceso detenido por el usuario.", None, None)
                    try:
                        self.status.set("Proceso detenido por el usuario.")
                    except Exception:
                        pass


                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", str(e)))
                    self._set_progress(0, "Error en el proceso batch (Filtros).", None, None)

                finally:
                    self.after(0, lambda: self._disable_all_buttons(False))

            threading.Thread(target=worker, daemon=True).start()
            return

        # ---------------- MODO: apilado por carpetas (batch) ----------------
        if not methods:
            messagebox.showwarning("Sin método", "Selecciona al menos un método de apilado.")
            return

        self._disable_all_buttons(True)
        self._set_progress(0, "Procesando carpetas.", None, None)

        folders = list(self.folder_paths)
        total_folders = len(folders)
        total_methods = len(methods)
        total_jobs = total_folders * total_methods

        def worker():
            try:
                job_index = 0
                last_result = None

                for f_idx, folder in enumerate(folders, start=1):
                    img_paths = list_images_in_folder(folder)
                    folder_name = os.path.basename(folder.rstrip("/\\"))

                    # Carpeta de salida por carpeta de entrada (más fácil de ubicar los resultados)
                    out_folder = os.path.join(out_root, folder_name)
                    try:
                        os.makedirs(out_folder, exist_ok=True)
                    except Exception:
                        out_folder = out_root

                    if len(img_paths) < 2:
                        # Avanzar el progreso equivalente a todos los métodos de esta carpeta
                        for _ in methods:
                            job_index += 1
                            pct = 100.0 * job_index / total_jobs
                            self._set_progress(
                                pct,
                                f"[{f_idx}/{total_folders}] Carpeta sin suficientes imágenes: {folder_name}",
                                None,
                                None
                            )
                        continue

                    for m in methods:
                        job_index += 1
                        base = 100.0 * (job_index - 1) / total_jobs
                        span = 100.0 / total_jobs

                        if m == "pmax":
                            method_tag = "pmax"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.30, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "focus":
                                    pct_local = self._stage_progress_mapper(0.40, 0.25, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Mapa de foco {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "pyramid":
                                    pct_local = self._stage_progress_mapper(0.65, 0.20, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Pirámide {current}/{total}.",
                                        None,
                                        preview
                                    )
                                elif stage == "post":
                                    pct = base + span * 0.90
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Post-procesado.",
                                        None,
                                        preview
                                    )

                            # Config para guardar RAW de PMax
                            FocusStackerApp.current_auto_save_dir = out_folder
                            FocusStackerApp.current_output_format = (
                                    self.output_format_var.get() or "jpg").strip().lower()
                            FocusStackerApp.current_raw_name = f"resultado_{folder_name}_pmax_RAW"

                            result = stack_paths(img_paths, progress_cb=ui_cb)

                        elif m == "weighted":
                            method_tag = "weighted"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.40, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "focusW":
                                    pct_local = self._stage_progress_mapper(0.50, 0.20, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Pesos {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "fusionW":
                                    pct = base + span * 0.85
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Integrando.",
                                        None,
                                        preview
                                    )

                            result = stack_paths_weighted(img_paths, progress_cb=ui_cb)

                        elif m == "depth":
                            method_tag = "depth"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.40, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "focusD":
                                    pct_local = self._stage_progress_mapper(0.50, 0.20, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Mapa profundidad {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "refineD":
                                    pct = base + span * 0.75
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Refinando mapa.",
                                        None,
                                        preview
                                    )
                                elif stage == "fusionD":
                                    pct = base + span * 0.90
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Fusionando.",
                                        None,
                                        preview
                                    )

                            result = stack_paths_depth(img_paths, progress_cb=ui_cb)

                        elif m == "int_mean":
                            method_tag = "mean"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.55, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "integrate":
                                    pct = base + span * 0.75
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Integrando.",
                                        None,
                                        preview
                                    )

                            result = stack_paths_integration_mean(img_paths, progress_cb=ui_cb)

                        elif m == "int_median":
                            method_tag = "median"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.55, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "integrate":
                                    pct = base + span * 0.75
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Integrando.",
                                        None,
                                        preview
                                    )

                            result = stack_paths_integration_median(img_paths, progress_cb=ui_cb)

                        elif m == "int_sum":
                            method_tag = "sum"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.55, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "integrate":
                                    pct = base + span * 0.75
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Integrando.",
                                        None,
                                        preview
                                    )

                            result = stack_paths_integration_sum(img_paths, progress_cb=ui_cb)

                        elif m == "int_sigma":
                            method_tag = "sigma"

                            def ui_cb(stage, current, total, preview):
                                if stage == "load":
                                    pct = base + span * 0.10
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Cargando {folder_name}.",
                                        preview,
                                        None
                                    )
                                elif stage == "align":
                                    pct_local = self._stage_progress_mapper(0.10, 0.55, current, total)
                                    pct = base + span * pct_local
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Alineando {current}/{total}.",
                                        preview,
                                        None
                                    )
                                elif stage == "integrate":
                                    pct = base + span * 0.75
                                    self._set_progress(
                                        pct,
                                        f"[{f_idx}/{total_folders}] ({method_tag}) Integrando.",
                                        None,
                                        preview
                                    )

                            result = stack_paths_integration_sigma_clipping(img_paths, progress_cb=ui_cb)

                        else:
                            raise RuntimeError(f"Método desconocido: {m}")

                        # NO aplicar filtros de ruido al resultado apilado.
                        # Los filtros de ruido se aplican sobre las imágenes cargadas y generan archivos nuevos.

                        # Guardar resultado de este método para esta carpeta
                        self.result = result
                        last_result = result

                        base_name = f"{folder_name}_{method_tag}"
                        out_path = self._unique_out_path(out_folder, base_name)
                        self._save_image(out_path, result)

                        # Actualizar progreso final de este job
                        pct_final = base + span
                        self._set_progress(
                            pct_final,
                            f"[{f_idx}/{total_folders}] Guardado: {os.path.basename(out_path)}",
                            None,
                            result
                        )

                if last_result is not None:
                    self._set_progress(100, "Procesamiento de carpetas terminado.", None, last_result)
                else:
                    self._set_progress(100, "Procesamiento de carpetas terminado (sin resultados válidos).", None, None)

            except ProcessCancelled:
                try:
                    self.cancel_event.clear()
                except Exception:
                    pass
                self._set_progress(0, "Proceso detenido por el usuario.", None, None)
                try:
                    self.status.set("Proceso detenido por el usuario.")
                except Exception:
                    pass


            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self._set_progress(0, "Error en el proceso batch.", None, None)

            finally:
                self.after(0, lambda: self._disable_all_buttons(False))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- guardado ----------------

    def _get_output_ext(self):
        fmt = (self.output_format_var.get() or "jpg").strip().lower()
        if fmt in ("jpg", "jpeg"):
            return ".jpg"
        if fmt == "png":
            return ".png"
        if fmt in ("tif", "tiff"):
            return ".tiff"
        return ".jpg"

    def _unique_out_path(self, folder, base_name):
        ext = self._get_output_ext()
        out_path = os.path.join(folder, base_name + ext)
        k = 1
        while os.path.exists(out_path):
            out_path = os.path.join(folder, f"{base_name}_{k}" + ext)
            k += 1
        return out_path

    def _save_image(self, out_path, img_bgr):
        ext = os.path.splitext(out_path)[1].lower().lstrip(".")
        if not ext:
            ext = "jpg"

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if ext in ("jpg", "jpeg"):
            pil.save(
                out_path,
                format="JPEG",
                quality=100,
                subsampling=0,
                optimize=True,
                progressive=False,
            )
        elif ext == "png":
            pil.save(out_path, format="PNG", compress_level=0)
        elif ext in ("tif", "tiff"):
            pil.save(out_path, format="TIFF", compression="tiff_lzw")
        else:
            pil.save(
                out_path,
                format="JPEG",
                quality=100,
                subsampling=0,
                optimize=True,
                progressive=False,
            )

    def save_result(self):
        """
        Botón para elegir la carpeta de guardado automático.
        Si ya hay un resultado en memoria, lo guarda inmediatamente en esa carpeta.
        """
        folder = filedialog.askdirectory(title="Selecciona carpeta donde se guardarán los resultados")
        if not folder:
            return

        # Guardar carpeta para usos futuros
        self.auto_save_dir = folder
        self._update_autosave_label()

        # También lo dejamos listo para el guardado RAW del stack_paths
        FocusStackerApp.current_auto_save_dir = self.auto_save_dir
        FocusStackerApp.current_output_format = (self.output_format_var.get() or "jpg").strip().lower()

        # Si ya hay una imagen procesada, guardarla de inmediato
        if self.result is not None:
            base = "resultado"
            out_path = self._unique_out_path(self.auto_save_dir, base)
            self._save_image(out_path, self.result)
            self.status.set(f"Guardado en: {out_path}")
        else:
            self.status.set("Carpeta seleccionada para guardado automático.")

    # ---------------- previews ----------------

    def _clear_previews(self):
        self.src_canvas.delete("all")
        self.res_canvas.delete("all")
        self.src_imgtk = None
        self.res_imgtk = None

    def _show_src_preview(self, img_bgr):
        self._show_on_canvas(self.src_canvas, img_bgr, is_src=True)

    def _show_res_preview(self, img_bgr):
        self._show_on_canvas(self.res_canvas, img_bgr, is_src=False)

    def _show_on_canvas(self, canvas, img_bgr, is_src):
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 600, 500

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        w, h = pil.size
        scale = min(canvas_w / w, canvas_h / h)
        new_size = (int(w * scale), int(h * scale))
        pil = pil.resize(new_size, Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=imgtk, anchor="center")

        if is_src:
            self.src_imgtk = imgtk
        else:
            self.res_imgtk = imgtk


if __name__ == "__main__":
    app = FocusStackerApp()
    app.mainloop()
