import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_specification(imgIn, imgRef_channel):
    old_shape = imgIn.shape

    src = imgIn.ravel()
    ref = imgRef_channel.ravel()

    src_values, bin_idx, src_counts = np.unique(src, return_inverse=True, return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64) / src.size

    ref_values, ref_counts = np.unique(ref, return_counts=True)
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / ref.size

    interp_values = np.interp(src_cdf, ref_cdf, ref_values)
    matched = interp_values[bin_idx].reshape(old_shape).astype(np.uint8)
    return matched

def histogram_equalization(imgIn):
    if len(imgIn.shape) == 2:  # grayscale
        return cv2.equalizeHist(imgIn)
    else:  # color image
        imgYCrCb = cv2.cvtColor(imgIn, cv2.COLOR_BGR2YCrCb)
        imgYCrCb[:, :, 0] = cv2.equalizeHist(imgYCrCb[:, :, 0])
        return cv2.cvtColor(imgYCrCb, cv2.COLOR_YCrCb2BGR)

# ===== Input Method Selection =====
method = input("Pilih metode (equalization / specification): ").strip().lower()

# ===== Load Images =====
imgIn = cv2.imread('masukan.jpg')   
imgRef = cv2.imread('referensi.jpg')   

if imgIn is None or (method == 'specification' and imgRef is None):
    raise IOError("Pastikan file citra tersedia dan path sudah benar.")

if method == 'equalization':
    imgOut = histogram_equalization(imgIn)
    process_title = 'Histogram Ekualisasi'

elif method == 'specification':
    if len(imgIn.shape) > 2 and imgIn.shape[2] == 3:
        imgIn_gray = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)
    else:
        imgIn_gray = imgIn

    if imgIn_gray.shape != imgRef[:, :, 0].shape:
        imgRef = cv2.resize(imgRef, (imgIn_gray.shape[1], imgIn_gray.shape[0]))

    I1 = histogram_specification(imgIn_gray, imgRef[:, :, 0])
    I2 = histogram_specification(imgIn_gray, imgRef[:, :, 1])
    I3 = histogram_specification(imgIn_gray, imgRef[:, :, 2])

    imgOut = np.zeros((imgIn_gray.shape[0], imgIn_gray.shape[1], 3), dtype=np.uint8)
    imgOut[:, :, 0] = I1
    imgOut[:, :, 1] = I2
    imgOut[:, :, 2] = I3

    process_title = 'Histogram Spesifikasi'
else:
    raise ValueError("Metode tidak dikenali. Pilih 'equalization' atau 'specification'.")

# ===== Display Results =====
colors = ('b', 'g', 'r')

plt.figure(figsize=(15, 6))

plt.subplot(2, 3, 1)
plt.title('Citra Masukan')
if len(imgIn.shape) == 2:
    plt.imshow(imgIn, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Citra Referensi')
if method == 'specification':
    plt.imshow(cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB))
else:
    plt.text(0.3, 0.5, 'Tidak Diperlukan', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title(f'Citra Hasil - {process_title}')
if len(imgOut.shape) == 2:
    plt.imshow(imgOut, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(imgOut, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Histogram Citra Masukan')
plt.hist(imgIn.ravel(), bins=256, color='gray', alpha=0.7)
plt.xlim(0, 255)

plt.subplot(2, 3, 5)
plt.title('Histogram Citra Referensi')
if method == 'specification':
    for i, col in enumerate(colors):
        plt.hist(imgRef[:, :, i].ravel(), bins=256, color=col, alpha=0.5)
else:
    plt.text(0.3, 0.5, 'Tidak Diperlukan', fontsize=12)
plt.xlim(0, 255)

plt.subplot(2, 3, 6)
plt.title(f'Histogram Hasil - {process_title}')
if len(imgOut.shape) == 2:
    plt.hist(imgOut.ravel(), bins=256, color='gray', alpha=0.7)
else:
    for i, col in enumerate(colors):
        plt.hist(imgOut[:, :, i].ravel(), bins=256, color=col, alpha=0.5)
plt.xlim(0, 255)

plt.tight_layout()
plt.show()
