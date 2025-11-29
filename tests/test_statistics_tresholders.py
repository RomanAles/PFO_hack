import numpy as np
import rasterio
from prompt_toolkit.utils import to_str
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from PIL import Image
import os

# Параметры палитры для маски
PALLETE = [
    [0, 0, 0],  # не вода - черный цвет
    [0, 0, 255]  # вода - синий цвет
]

# Функции для расчета водных индексов
def calculate_indices(blue, green, red, nir, mir, swir):
    indices = {
        'NDWI': (green - nir) / (green + nir),
        'NDMI': (nir - mir) / (nir + mir),
        'MNDWI': (green - mir) / (green + mir),
        'WRI': (green + red) / (nir + mir),
        'NDVI': (nir - red) / (nir + red),
        'AWEI': 4 * (green - mir) - (0.25 * nir + 2.75 * swir)
    }
    return indices

# Функция для создания маски воды на основе порогов индексов
def create_water_mask(indices, file_name):
    # Условия для каждого индекса
    ndwi_mask = indices['NDWI'] < 1   #74
    ndmi_mask = indices['NDMI'] < -6.48   #50
    mndwi_mask = indices['MNDWI'] < 1   #70
    wri_mask = indices['WRI'] > 1      #70
    ndvi_mask = indices['NDVI'] > 1    #70
    awei_mask = indices['AWEI'] < 4.35   #62

    # Объединяем условия для создания водной маски
    water_mask = (ndwi_mask | mndwi_mask | wri_mask | ndvi_mask).astype(np.uint8)

    # Статистика по пороговым значениям
    print(f"Statistics for thresholds of {file_name}:")
    print(f"NDWI: {np.sum(ndwi_mask)} pixels")
    print(f"NDMI: {np.sum(ndmi_mask)} pixels")
    print(f"MNDWI: {np.sum(mndwi_mask)} pixels")
    print(f"WRI: {np.sum(wri_mask)} pixels")
    print(f"NDVI: {np.sum(ndvi_mask)} pixels")
    print(f"AWEI: {np.sum(awei_mask)} pixels")

    return water_mask

# Функции для нормализации и увеличения яркости
def normalize(band):
    band_min, band_max = band.min(), band.max()
    return (band - band_min) / (band_max - band_min)

def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)

# Функция для преобразования изображения
def convert(im_path):
    with rasterio.open(im_path) as fin:
        red = fin.read(3)
        green = fin.read(2)
        blue = fin.read(1)
    red_b = brighten(red)
    green_b = brighten(green)
    blue_b = brighten(blue)
    red_bn = normalize(red_b)
    green_bn = normalize(green_b)
    blue_bn = normalize(blue_b)
    return np.dstack((blue_b, green_b, red_b)), np.dstack((red_bn, green_bn, blue_bn))

# Основная функция для обработки изображения Sentinel-2A и создания масок
def process_image(image_path, output_mask_path, reference_mask_path=None):
    with rasterio.open(image_path) as src:
        # Считывание необходимых каналов с понижением разрешения до 20 м
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        nir = src.read(7)
        mir = src.read(9)
        swir = src.read(10)
        # Расчет индексов
        indices = calculate_indices(blue, green, red, nir, mir, swir)
        file_name = os.path.basename(image_path)
        # Создание водной маски
        water_mask = create_water_mask(indices, file_name)
        # Сохранение маски в формате .tif
        with rasterio.open(
            output_mask_path,
            'w',
            driver='GTiff',
            height=water_mask.shape[0],
            width=water_mask.shape[1],
            count=1,
            dtype=water_mask.dtype,
            crs=src.crs,
            transform=src.transform
        ) as dst:
            dst.write(water_mask, 1)
    # Визуализация исходного изображения и маски воды
    #plot_data(image_path, output_mask_path)

# Визуализация исходного изображения и маски воды
def plot_data(image_path, mask_path):
    plt.figure(figsize=(12, 12))
    pal = [value for color in PALLETE for value in color]
    # Исходное изображение
    plt.subplot(1, 2, 1)
    _, img = convert(image_path)
    plt.imshow(img)
    plt.title('Исходное изображение')
    # Маска воды
    plt.subplot(1, 2, 2)
    with rasterio.open(mask_path) as fin:
        mask = fin.read(1)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(pal)
    plt.imshow(mask)
    plt.title('Маска воды')
    plt.show()

# Путь к изображению и пути для сохранения маски
image_path = 'gistograms/images'
output_mask_path = 'gistograms/mask_pred'

# Запуск обработки
files_name = ['1.tif', '2.tif', '4.tif', '5.tif', '6_1.tif', '6_2.tif', '9_1.tif', '9_2.tif']
for file_name in files_name:
    res_image_path = os.path.join(image_path, file_name)
    res_mask_path = os.path.join(output_mask_path, file_name)
    process_image(res_image_path, res_mask_path)
