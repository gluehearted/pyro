import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Ganti dengan nama file yang tadi kamu download
fp = '10N_120E.tif'

with rasterio.open(fp) as src:
    # Kita cuma baca 1/20 dari resolusi aslinya buat preview
    # out_shape = (jumlah_band, tinggi, lebar)
    data = src.read(1, out_shape=(1, int(src.height // 20), int(src.width // 20)))
    
    print(f"Resolusi asli: {src.width}x{src.height}")
    print(f"Resolusi preview: {data.shape[1]}x{data.shape[0]}")

    plt.figure(figsize=(10, 8))
    # Pakai interpolation='nearest' biar ngerendernya nggak berat
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Indikator Gambut')
    plt.title('Preview Cepat Data Gambut')
    plt.show()