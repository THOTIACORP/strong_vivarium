import os
import cv2
import numpy as np
import random
from glob import glob
import json
import tifffile
# === Configurações ===
image_dir = 'thermal_images'
mask_dir = 'masks_thermal_images_unet'  # máscaras multiclasses (cada parte um valor)
output_image_dir = 'output_images_panoptic'
output_mask_dir = 'output_masks_panoptic'  # máscaras panoptic (png 16-bit)
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Leitura das imagens e máscaras
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(glob(os.path.join(image_dir, ext)))

images_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
masks_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(mask_dir, '*.png'))}

common_keys = sorted(set(images_dict.keys()) & set(masks_dict.keys()))
assert len(common_keys) > 0, "Nenhum par imagem/máscara correspondente encontrado"

data = [(images_dict[k], masks_dict[k]) for k in common_keys]

standard_size = (512, 512)  # largura, altura
resized_data = []
for img_path, mask_path in data:
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    img_resized = cv2.resize(img, standard_size)
    mask_resized = cv2.resize(mask, standard_size, interpolation=cv2.INTER_NEAREST)

    resized_data.append((img_resized, mask_resized))

def random_transform(image, mask, size):
    h, w = size
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    transformed_image = cv2.warpAffine(image, M, (w, h), borderValue=0)
    transformed_mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)
    dx = random.randint(-w // 4, w // 4)
    dy = random.randint(-h // 4, h // 4)
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    transformed_image = cv2.warpAffine(transformed_image, T, (w, h), borderValue=0)
    transformed_mask = cv2.warpAffine(transformed_mask, T, (w, h), borderValue=0)
    return transformed_image, transformed_mask

def combine_multiple(data, idx, num_animals=4):
    selected = random.sample(data, num_animals)
    base_img, base_mask = selected[0]
    h, w = base_img.shape[:2]
    combined_img = base_img.copy()
    combined_mask = np.zeros((h, w), dtype=np.uint8)  # máscara multiclasses combinada

    for img, mask in selected:
        img, mask = random_transform(img, mask, (h, w))
        if combined_mask.shape != mask.shape:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Remove sobreposição (mantém somente a primeira instância)
        overlap = (combined_mask > 0) & (mask > 0)
        mask[overlap] = 0

        mask_region = mask > 0
        combined_img[mask_region] = img[mask_region]
        combined_mask[mask_region] = mask[mask_region]

    cv2.imwrite(os.path.join(output_image_dir, f'combined_{idx}.png'), combined_img)
    cv2.imwrite(os.path.join(output_mask_dir, f'combined_{idx}.png'), combined_mask)

# Gerar imagens compostas e máscaras multiclasses
for i in range(50):
    combine_multiple(resized_data, i, num_animals=random.randint(2,4))

print("✅ Imagens compostas e máscaras multiclasses salvas.")

# === Geração das máscaras panoptic e JSON ===

categories = [
    # categories para partes (stuff)
    {"id": 1, "name": "corpo", "supercategory": "parte", "isthing": 0},
    {"id": 2, "name": "cabeca", "supercategory": "parte", "isthing": 0},
    {"id": 3, "name": "cauda", "supercategory": "parte", "isthing": 0},
    # categoria para rato como instância (thing)
    {"id": 4, "name": "rato", "supercategory": "animal", "isthing": 1},
]

# Mapear id das partes para isthing=0, rato para isthing=1
stuff_ids = [c["id"] for c in categories if c["isthing"] == 0]
thing_id = 4  # id para rato

def create_panoptic_mask(mask_multiclass):
    """
    Recebe a máscara multiclasses (partes) e gera:
    - máscara panoptic 2D (uint32): pixel = category_id << 16 + instance_id
    - lista de segmentos (segments_info)
    """

    h, w = mask_multiclass.shape
    panoptic_mask = np.zeros((h, w), dtype=np.uint32)
    segments_info = []

    # Detectar ratos (instâncias) usando connectedComponents na máscara binarizada (mask > 0)
    binary_rato_mask = (mask_multiclass > 0).astype(np.uint8)
    num_ratos, labels = cv2.connectedComponents(binary_rato_mask)

    instance_id = 1  # ID começa em 1 para instâncias

    # Para cada rato
    for inst_idx in range(1, num_ratos):
        inst_mask = (labels == inst_idx)

        # Definir pixels da instância (rato) na máscara panoptic
        panoptic_mask[inst_mask] = (thing_id << 16) + instance_id

        # Calcular área
        area = int(np.sum(inst_mask))

        segments_info.append({
            "id": (thing_id << 16) + instance_id,
            "category_id": thing_id,
            "area": area,
            "iscrowd": 0,
        })
        instance_id += 1

    # Agora, para as partes (stuff) que não pertencem a uma instância (rato)
    # Como as partes são representadas por valores no mask_multiclass,
    # mas podem estar dentro dos ratos, vamos marcar pixels que não estão em nenhuma instância (rato)
    for cat in stuff_ids:
        part_mask = (mask_multiclass == cat)
        # remover pixels já atribuídos à instância rato (para evitar overlap)
        part_mask = part_mask & (panoptic_mask == 0)

        if np.sum(part_mask) == 0:
            continue

        # Definir categoria + instance_id=0 para stuff (sem instância)
        panoptic_mask[part_mask] = cat  # só category_id, instance_id=0

        area = int(np.sum(part_mask))

        segments_info.append({
            "id": cat,  # instance_id = 0 para stuff
            "category_id": cat,
            "area": area,
            "iscrowd": 0,
        })

    return panoptic_mask, segments_info

# Criar arquivo JSON panoptic
panoptic_json = {
    "info": {
        "description": "Dataset Panoptic Ratos",
        "version": "1.0",
        "year": 2025,
    },
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": [],
}

for idx in range(50):
    img_filename = f'combined_{idx}.png'
    panoptic_mask_path = os.path.join(output_mask_dir, f'combined_{idx}.png')

    mask_path = os.path.join(output_mask_dir, f'combined_{idx}.png')
    mask_multiclass = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_multiclass is None:
        print(f"Aviso: máscara não encontrada: {mask_path}")
        continue

    panoptic_mask, segments_info = create_panoptic_mask(mask_multiclass)

    # Salvar máscara panoptic como TIFF 32-bit uint
    panoptic_mask_path_tiff = panoptic_mask_path.replace('.png', '.tiff')
    tifffile.imwrite(panoptic_mask_path_tiff, panoptic_mask)

    panoptic_json["images"].append({
        "id": idx,
        "width": standard_size[0],
        "height": standard_size[1],
        "file_name": img_filename,
    })

    panoptic_json["annotations"].append({
        "image_id": idx,
        "file_name": img_filename,
        "file_name_panoptic": os.path.basename(panoptic_mask_path_tiff),
        "segments_info": segments_info,
    })

with open('panoptic_annotations.json', 'w', encoding='utf-8') as f:
    json.dump(panoptic_json, f, indent=2, ensure_ascii=False)

print("✅ Dataset panoptic gerado: panoptic_annotations.json + máscaras TIFF")