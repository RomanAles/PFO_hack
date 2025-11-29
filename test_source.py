import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import os
import cv2
import joblib

# Функция для отображения водной маски
def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.title("Предсказанная водная маска")
    plt.axis('off')
    plt.show()

# Функция для размытия водной маски по Гауссу
def gaussian_smooth_mask(mask, kernel_size=(5, 5)):
    smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), kernel_size, 0)
    smoothed_mask = (smoothed_mask > 0.95).astype(np.uint8)
    return smoothed_mask

# Функция для создания водной маски на основе порогов индексов
def create_water_mask(indices):
    #
    ndwi_mask = indices['NDWI'] < 1
    ndmi_mask = indices['NDMI'] < 1
    mndwi_mask = indices['MNDWI'] < 1
    wri_mask = indices['WRI'] > 1
    ndvi_mask = indices['NDVI'] > 1
    awei_mask = indices['AWEI'] < 0
    water_mask = (ndwi_mask | mndwi_mask | wri_mask | ndvi_mask).astype(np.uint8)
    return water_mask

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

# Функции для нормализации и увеличения яркости
def normalize(band):
    band_min, band_max = band.min(), band.max()
    return (band - band_min) / (band_max - band_min)

# Функция для увеличения яркости спектральной полосы
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

# Функция для обработки изображения и создания водной маски
def process_image(image_path):
    with rasterio.open(image_path) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        nir = src.read(7)
        mir = src.read(9)
        swir = src.read(10)

        indices = calculate_indices(blue, green, red, nir, mir, swir)
        water_mask = create_water_mask(indices)
    
    return water_mask

# Функция для загрузки изображения и расчета индексов для модели CatBoost
def load_image_as_dataframe_cat(image_path):
    with rasterio.open(image_path) as img:
        image_data = img.read()
        image_data = image_data.reshape(image_data.shape[0], -1).T
    
    df = pd.DataFrame(image_data, columns=[f'{i}' for i in range(image_data.shape[1])])
    df['NDWI'] = (df['1'] - df['6']) / (df['1'] + df['6'])
    df['NDMI'] = (df['6'] - df['8']) / (df['6'] + df['8'])
    df['MNDWI'] = (df['1'] - df['8']) / (df['1'] + df['8'])
    df['WRI'] = (df['1'] + df['2']) / (df['6'] + df['8'])
    df['NDVI'] = (df['6'] - df['2']) / (df['6'] + df['2'])
    df['AWEI'] = 4 * (df['1'] - df['8']) - (0.25 * df['6'] + 2.75 * df['9'])
    return df

# Функция для предсказания водных пикселей с использованием модели CatBoost
def predict_water_pixels_cat(image_path, model):
    df = load_image_as_dataframe_cat(image_path)
    y_pred = model.predict(df)
    
    with rasterio.open(image_path) as img:
        height, width = img.height, img.width
    binary_mask = y_pred.reshape((height, width)).astype(np.uint8)

    return binary_mask

# Функция для загрузки изображения и расчета индексов для модели логистической регрессии
def load_image_as_dataframe_log(image_path):
    with rasterio.open(image_path) as img:
        image_data = img.read()
        image_data = image_data.reshape(image_data.shape[0], -1).T
    
    df = pd.DataFrame(image_data, columns=[f'{i}' for i in range(image_data.shape[1])])
    df['NDWI'] = (df['1'] - df['6']) / (df['1'] + df['6'])
    df['NDMI'] = (df['6'] - df['8']) / (df['6'] + df['8'])
    df['MNDWI'] = (df['1'] - df['8']) / (df['1'] + df['8'])
    df['WRI'] = (df['1'] + df['2']) / (df['6'] + df['8'])
    df['NDVI'] = (df['6'] - df['2']) / (df['6'] + df['2'])
    df['AWEI'] = 4 * (df['1'] - df['8']) - (0.25 * df['6'] + 2.75 * df['9'])
    return df

# Функция для предсказания водных пикселей с использованием модели логистической регрессии
def predict_water_pixels_log(image_path, model):
    """Загружает изображение, предсказывает водные пиксели и возвращает бинарную матрицу."""
    df = load_image_as_dataframe_log(image_path)
    df.fillna(df.mean(), inplace=True)
    
    # Предсказание класса "вода" или "не вода"
    features = df[['NDWI', 'NDMI', 'MNDWI', 'WRI', 'NDVI', 'AWEI']]  # Используем только необходимые признаки
    y_pred = model.predict(features)
    
    # Преобразование предсказаний в исходное разрешение изображения
    with rasterio.open(image_path) as img:
        height, width = img.height, img.width
    binary_mask = y_pred.reshape((height, width)).astype(np.uint8)  # Приводим к uint8 для бинарного изображения

    binary_mask = np.where(binary_mask == 1, 0, 1).astype(np.uint8)
    
    return binary_mask

# Функция для комбинирования полученных водных масок в одну
def combine_masks(mask1, mask2, mask3):
    combined_mask = np.logical_or(mask1, mask2)
    combined_mask = np.logical_or(combined_mask, mask3).astype(np.uint8)
    return combined_mask

# Функция для сохранения созданной водной маски
def save_binary_mask(mask, reference_image_path, output_path):
    with rasterio.open(reference_image_path) as src:
        meta = src.meta
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask, 1)

def main(input_folder, output_folder):
    # Запуск модели CatBoost
    model_cat_path = 'ваш путь' # Укажите путь к модели catboost_model.cbm
    model_cat = CatBoostClassifier()
    model_cat.load_model(model_cat_path)

    # Запуск модели logical_regression
    model_log_path = 'ваш путь' # Укажите путь к модели logistic_regression_model.joblib
    model_log = joblib.load(model_log_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Получение мультиспектральных изображений
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # Основная часть программы
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_mask_path = os.path.join(output_folder, f'mask_{image_file}')

        # Создание водных масок по физической, CatBoost, logical_regression моделям
        water_mask_index = process_image(image_path)
        water_mask_cat = gaussian_smooth_mask(predict_water_pixels_cat(image_path, model_cat))
        water_mask_log = gaussian_smooth_mask(predict_water_pixels_log(image_path, model_log))

        # Комбинирование полученных водных масок
        water_mask_result = combine_masks(water_mask_index, water_mask_cat, water_mask_log)

        # Сохранение полученной водной маски
        save_binary_mask(water_mask_result, image_path, output_mask_path)

        print(f"Бинарная маска для {image_file} успешно сохранена по пути: {output_mask_path}")

if __name__ == "__main__":
    input_folder = 'ваш путь'  # Укажите путь к папке с изображениями
    output_folder = 'ваш путь'             # Укажите путь к папке для сохранения масок
    main(input_folder, output_folder)
