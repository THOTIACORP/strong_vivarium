import sys
import os
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QFileDialog, QLineEdit
)
from PyQt5.QtCore import Qt

def rotate_image_and_mask(image, mask, angle):
    """Rotaciona a imagem e a máscara mantendo o mesmo tamanho."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_img, rotated_mask

def process_images(image_dir, mask_dir, output_dir, num_fundos, num_ratos_por_fundo, log_widget):
    """
    Processa as imagens com as pastas especificadas pelo usuário.
    
    Args:
        image_dir: Pasta com as imagens originais
        mask_dir: Pasta com as máscaras correspondentes
        output_dir: Pasta de saída para os resultados
        num_fundos: Número de fundos a processar
        num_ratos_por_fundo: Número de variações por fundo
        log_widget: Widget para exibir logs
    """
    if not os.path.exists(image_dir):
        log_widget.append(f'❌ Pasta de imagens não encontrada: {image_dir}')
        return
    
    if not os.path.exists(mask_dir):
        log_widget.append(f'❌ Pasta de máscaras não encontrada: {mask_dir}')
        return
    
    # Criar subpastas de saída
    output_background_dir = os.path.join(output_dir, 'fundos_sem_objeto')
    output_objects_dir = os.path.join(output_dir, 'novos_objetos')
    output_masks_dir = os.path.join(output_dir, 'mascaras')
    
    os.makedirs(output_background_dir, exist_ok=True)
    os.makedirs(output_objects_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    target_size = (1428, 1068)

    def log(text):
        log_widget.append(text)
        log_widget.repaint()
        QApplication.processEvents()

    def get_matching_mask(img_filename, mask_dir):
        base_name = os.path.splitext(img_filename)[0]
        for f in os.listdir(mask_dir):
            if os.path.splitext(f)[0] == base_name:
                return f
        return None

    def find_background_patch(mask, w, h, exclude_rect):
        height, width = mask.shape
        for attempt in range(1000):
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            ex, ey, ew, eh = exclude_rect
            # Verifica se a região se sobrepõe ao objeto, se sim, pula
            if (x < ex + ew and x + w > ex and y < ey + eh and y + h > ey):
                continue
            patch = mask[y:y+h, x:x+w]
            if np.all(patch == 0):
                return x, y
        return None

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    log(f'🔎 Encontradas {len(image_files)} imagens para processar.')

    if len(image_files) == 0:
        log('❌ Nenhuma imagem encontrada na pasta especificada.')
        return

    fundos_selecionados = image_files[:num_fundos]
    for img_index, img_file in enumerate(fundos_selecionados):
        mask_file = get_matching_mask(img_file, mask_dir)
        if mask_file is None:
            log(f'⚠️ Nenhuma máscara encontrada para {img_file}, pulando...')
            continue

        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            log(f'⚠️ Erro ao abrir imagem ou máscara para {img_file}')
            continue

        img = cv2.resize(img, target_size)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Mascara binária para objeto (qualquer classe 1,2,3 vira 255)
        object_mask = np.isin(mask, [1, 2, 3]).astype(np.uint8) * 255
        object_mask_inv = cv2.bitwise_not(object_mask)

        coords = cv2.findNonZero(object_mask)
        if coords is None:
            log(f'⚠️ Nenhum objeto encontrado na máscara de {img_file}, pulando...')
            continue

        x, y, w, h = cv2.boundingRect(coords)
        log(f'Processando {img_file} ({img_index+1}/{len(fundos_selecionados)}), objeto bbox: x={x}, y={y}, w={w}, h={h}')

        object_cropped = img[y:y+h, x:x+w]
        object_mask_cropped = object_mask[y:y+h, x:x+w]
        object_mask_original_cropped = mask[y:y+h, x:x+w]  # máscara original com classes 1,2,3

        # Cria fundo sem objeto
        fundo_sem_objeto = cv2.bitwise_and(img, img, mask=object_mask_inv)

        # Encontra um patch do fundo para substituir a área do objeto no fundo_sem_objeto
        patch_pos = find_background_patch(mask, w, h, (x, y, w, h))
        if patch_pos is None:
            log(f'⚠️ Não encontrou região de fundo adequada para preencher o objeto em {img_file}, pulando...')
            continue

        patch_x, patch_y = patch_pos
        fundo_patch = img[patch_y:patch_y+h, patch_x:patch_x+w]
        fundo_sem_objeto[y:y+h, x:x+w] = fundo_patch

        # Salva a imagem do fundo sem objeto
        fundo_sem_objeto_path = os.path.join(output_background_dir, f'fundo_sem_objeto_{img_file}')
        cv2.imwrite(fundo_sem_objeto_path, fundo_sem_objeto)
        log(f'Fundo sem objeto salvo: {fundo_sem_objeto_path}')

        # SALVA MÁSCARA TODA ZERO PARA O FUNDO SEM OBJETO
        fundo_sem_objeto_mask = np.zeros_like(mask, dtype=np.uint8)
        fundo_sem_objeto_mask_path = os.path.join(output_masks_dir, f'fundo_sem_objeto_mask_{img_file}')
        cv2.imwrite(fundo_sem_objeto_mask_path, fundo_sem_objeto_mask)
        log(f'Máscara do fundo sem objeto salva: {fundo_sem_objeto_mask_path}')

        max_x = target_size[0] - w
        max_y = target_size[1] - h

        for i in range(num_ratos_por_fundo):
            img_variacao = fundo_sem_objeto.copy()
            mask_variacao = np.zeros_like(mask, dtype=np.uint8)

            # Rotação aleatória entre -180 e +180 graus (pode ajustar o range)
            angle = random.uniform(-180, 180)
            object_rotated, mask_rotated = rotate_image_and_mask(object_cropped, object_mask_original_cropped, angle)

            # Ajusta as dimensões após rotação (mesmo tamanho, mas pode ter áreas pretas)
            h_r, w_r = object_rotated.shape[:2]

            # Escolhe uma posição aleatória onde o objeto rotacionado caiba na imagem
            max_x_r = target_size[0] - w_r
            max_y_r = target_size[1] - h_r
            x_new = random.randint(0, max_x_r)
            y_new = random.randint(0, max_y_r)

            roi = img_variacao[y_new:y_new+h_r, x_new:x_new+w_r]
            # Máscara binária para usar no bitwise_and (apenas 255/0)
            mask_rotated_bin = (mask_rotated > 0).astype(np.uint8) * 255
            roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_rotated_bin))
            object_fg = cv2.bitwise_and(object_rotated, object_rotated, mask=mask_rotated_bin)
            dst = cv2.add(roi_bg, object_fg)
            img_variacao[y_new:y_new+h_r, x_new:x_new+w_r] = dst

            # Coloca a máscara rotacionada com as classes originais (1,2,3)
            # Usando máscara rotacionada com classes originais, que pode conter valores 1,2,3 em vez de 255
            mask_variacao[y_new:y_new+h_r, x_new:x_new+w_r] = mask_rotated

            output_object_path = os.path.join(output_objects_dir, f'{os.path.splitext(img_file)[0]}_var{i+1}.png')
            output_mask_path = os.path.join(output_masks_dir, f'{os.path.splitext(img_file)[0]}_var{i+1}_mask.png')

            cv2.imwrite(output_object_path, img_variacao)
            cv2.imwrite(output_mask_path, mask_variacao)

        log(f'✅ Processado {img_file} - {num_ratos_por_fundo} variações de objeto criadas com máscaras e rotação.')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("transforms_fake - Gerador de Dados Sintéticos")
        self.setGeometry(100, 100, 700, 500)
        
        # Variáveis para armazenar os caminhos
        self.image_dir = ""
        self.mask_dir = ""
        self.output_dir = ""

        layout = QVBoxLayout()

        # Seção de seleção de pastas
        layout.addWidget(QLabel("📁 Configuração de Pastas"))
        
        # Pasta de imagens
        hbox_img = QHBoxLayout()
        hbox_img.addWidget(QLabel("Pasta de Imagens:"))
        self.line_image_dir = QLineEdit()
        self.line_image_dir.setReadOnly(True)
        self.line_image_dir.setPlaceholderText("Selecione a pasta com as imagens...")
        hbox_img.addWidget(self.line_image_dir)
        btn_select_images = QPushButton("Selecionar")
        btn_select_images.clicked.connect(self.select_image_folder)
        hbox_img.addWidget(btn_select_images)
        layout.addLayout(hbox_img)

        # Pasta de máscaras
        hbox_mask = QHBoxLayout()
        hbox_mask.addWidget(QLabel("Pasta de Máscaras:"))
        self.line_mask_dir = QLineEdit()
        self.line_mask_dir.setReadOnly(True)
        self.line_mask_dir.setPlaceholderText("Selecione a pasta com as máscaras...")
        hbox_mask.addWidget(self.line_mask_dir)
        btn_select_masks = QPushButton("Selecionar")
        btn_select_masks.clicked.connect(self.select_mask_folder)
        hbox_mask.addWidget(btn_select_masks)
        layout.addLayout(hbox_mask)

        # Pasta de saída
        hbox_output = QHBoxLayout()
        hbox_output.addWidget(QLabel("Pasta de Saída:"))
        self.line_output_dir = QLineEdit()
        self.line_output_dir.setReadOnly(True)
        self.line_output_dir.setPlaceholderText("Selecione a pasta para salvar os resultados...")
        hbox_output.addWidget(self.line_output_dir)
        btn_select_output = QPushButton("Selecionar")
        btn_select_output.clicked.connect(self.select_output_folder)
        hbox_output.addWidget(btn_select_output)
        layout.addLayout(hbox_output)

        # Separador
        layout.addWidget(QLabel("⚙️ Parâmetros de Processamento"))

        # Número de fundos
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Número de fundos a usar:"))
        self.spin_fundos = QSpinBox()
        self.spin_fundos.setMinimum(1)
        self.spin_fundos.setMaximum(100)
        self.spin_fundos.setValue(5)
        hbox1.addWidget(self.spin_fundos)
        layout.addLayout(hbox1)

        # Número de objetos por fundo
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Número de objetos por fundo:"))
        self.spin_objetos = QSpinBox()
        self.spin_objetos.setMinimum(1)
        self.spin_objetos.setMaximum(100)
        self.spin_objetos.setValue(10)
        hbox2.addWidget(self.spin_objetos)
        layout.addLayout(hbox2)

        # Log de processamento
        layout.addWidget(QLabel("📋 Log de Processamento"))
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        layout.addWidget(self.log_widget)

        # Botão de processar
        btn_processar = QPushButton("🚀 Processar Imagens")
        btn_processar.clicked.connect(self.start_processing)
        layout.addWidget(btn_processar)

        self.setLayout(layout)

    def select_image_folder(self):
        """Seleciona a pasta de imagens."""
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Imagens")
        if folder:
            self.image_dir = folder
            self.line_image_dir.setText(folder)

    def select_mask_folder(self):
        """Seleciona a pasta de máscaras."""
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Máscaras")
        if folder:
            self.mask_dir = folder
            self.line_mask_dir.setText(folder)

    def select_output_folder(self):
        """Seleciona a pasta de saída."""
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída")
        if folder:
            self.output_dir = folder
            self.line_output_dir.setText(folder)

    def start_processing(self):
        """Inicia o processamento das imagens."""
        # Validar se todas as pastas foram selecionadas
        if not self.image_dir:
            self.log_widget.append("❌ Selecione a pasta de imagens primeiro!")
            return
        
        if not self.mask_dir:
            self.log_widget.append("❌ Selecione a pasta de máscaras primeiro!")
            return
        
        if not self.output_dir:
            self.log_widget.append("❌ Selecione a pasta de saída primeiro!")
            return

        num_fundos = self.spin_fundos.value()
        num_objetos = self.spin_objetos.value()
        
        self.log_widget.clear()
        self.log_widget.append(f"🎯 Iniciando processamento...")
        self.log_widget.append(f"📂 Pasta de imagens: {self.image_dir}")
        self.log_widget.append(f"📂 Pasta de máscaras: {self.mask_dir}")
        self.log_widget.append(f"📂 Pasta de saída: {self.output_dir}")
        self.log_widget.append(f"⚙️ Fundos: {num_fundos}, Objetos por fundo: {num_objetos}")
        self.log_widget.append("-" * 50)
        
        process_images(self.image_dir, self.mask_dir, self.output_dir, 
                      num_fundos, num_objetos, self.log_widget)
def get_widget():
    return MainWindow()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

