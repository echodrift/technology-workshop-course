import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import img_as_float
from pywt import swt2, iswt2
from tkinter import filedialog
from tkinter import Tk

# Function to get file path using file dialog
def get_file_path():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

# Read panchromatic Image
file_path = get_file_path()
I1 = resize(cv2.imread(file_path), (512, 512))
if I1.ndim >= 3:
    I1 = rgb2gray(I1)

# Read Multi spectral Image
file_path = get_file_path()
I2 = resize(cv2.imread(file_path), (256, 256))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(I1, cmap='gray')
plt.title('Pan Image')
plt.subplot(1, 2, 2)
plt.imshow(I2)
plt.title('Spectral Image')
plt.show()

# Upsampling
I2 = resize(I2, (512, 512))

# UDWT
Ia1 = img_as_float(I1)
Ia2 = img_as_float(I2)

# Pan
coeffs1 = swt2(Ia1, 'sym4', level=1)
ca1, (chd1, cvd1, cdd1) = coeffs1[0]
dec1 = np.block([[ca1, chd1], [cvd1, cdd1]])
enc1 = iswt2([(ca1, (chd1, cvd1, cdd1))], 'sym4')

plt.figure()
plt.imshow(np.abs(dec1), cmap='gray')
plt.title('UDWT PAN')
plt.show()

plt.figure()
plt.imshow(np.abs(enc1), cmap='gray')
plt.title('Decode UDWT PAN')
plt.show()

# MS
coeffs2 = swt2(Ia2, 'sym4', level=1)
ca2, (chd2, cvd2, cdd2) = coeffs2[0]

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(dec2[:, :, 0], cmap='gray')
plt.title('UDWT MS Red')
plt.subplot(1, 3, 2)
plt.imshow(dec2[:, :, 1], cmap='gray')
plt.title('UDWT MS Green')
plt.subplot(1, 3, 3)
plt.imshow(dec2[:, :, 2], cmap='gray')
plt.title('UDWT MS Blue')
plt.show()

# Injection model
gk = []
for ik in range(3):
    s = ca2[:, :, ik]
    gk.append(np.cov(np.hstack((s.flatten(), ca1.flatten()))) / np.var(ca1.flatten()))

plt.figure()
plt.bar(range(1, 4), gk)
plt.xlabel('Bins')
plt.ylabel('Weight Gain')
plt.show()

# Fusion
# LL
y = 0.3 * ca2[:, :, 0] + 0.4 * ca2[:, :, 1] + 0.3 * ca2[:, :, 2]
G1 = 1 - np.array(gk)
Ims2LL = ca2.copy()
for i in range(3):
    Ims2LL[:, :, i] = ca2[:, :, i] + gk[i] * (ca1 - y)

plt.figure()
plt.imshow(Ims2LL, cmap='gray')
plt.title('Enh LL Image')
plt.show()

Ims2LH = chd2.copy()
for i in range(3):
    Ims2LH[:, :, i] = chd1 + chd2[:, :, i]

Ims2HL = cvd2.copy()
for i in range(3):
    Ims2HL[:, :, i] = cvd1 + cvd2[:, :, i]

Ims2HH = cdd2.copy()
for i in range(3):
    Ims2HH[:, :, i] = cdd1 + cdd2[:, :, i]

# Inverse conversion
X = np.zeros_like(I2)
for i in range(3):
    X[:, :, i] = iswt2([(Ims2LL[:, :, i], (Ims2LH[:, :, i], Ims2HL[:, :, i], Ims2HH[:, :, i]))], 'sym4')

plt.figure()
plt.imshow(X)
plt.title('Enhanced image')
plt.show()

# Performance
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(I1, cmap='gray')
plt.title('Pan Image')
plt.subplot(1, 3, 2)
plt.imshow(I2)
plt.title('Spectral Image')
plt.subplot(1, 3, 3)
plt.imshow(X)
plt.title('Enhanced Image')
plt.show()