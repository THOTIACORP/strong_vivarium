import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from sklearn.decomposition import PCA
class ThermalImageProcessor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Imagens Térmicas")
        self.setGeometry(100, 100, 1000, 600)
        self.use_sam = False  # flag para fluxo SAM
    
        self.prev_endpoints = {}
        self.input_dir = ""
        self.output_dir = ""

        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()

     # Seleção de pastas
        self.btn_input = QtWidgets.QPushButton("Selecionar Pasta de Entrada")
        self.btn_input.clicked.connect(self.select_input_dir)
        self.lbl_input = QtWidgets.QLabel("Nenhuma pasta selecionada.")
        self.lbl_input.setStyleSheet("background-color: transparent; ")

        self.btn_output = QtWidgets.QPushButton("Selecionar Pasta de Saída")
        self.btn_output.clicked.connect(self.select_output_dir)
        self.lbl_output = QtWidgets.QLabel("Nenhuma pasta selecionada.")
        self.lbl_output.setStyleSheet("background-color: transparent; ")

        # Checkbox para ativar SAM
        self.checkbox_sam = QtWidgets.QCheckBox("Usar Segment Anything Model (SAM) para processamento adicional")
        self.checkbox_sam.setStyleSheet("""
            color: blue;
            background-color: transparent;
             text-decoration: underline;
        """)

        # Adicione os widgets na ordem desejada
        layout.addWidget(self.btn_input)
        layout.addWidget(self.lbl_input)
        layout.addWidget(self.btn_output)
        layout.addWidget(self.lbl_output)
        layout.addWidget(self.checkbox_sam)    # Checkbox depois dos botões e labels
        self.btn_process = QtWidgets.QPushButton("Iniciar Processamento")
        self.btn_process.clicked.connect(self.process_images)
        layout.addWidget(self.btn_process)

        # Visualização de imagens
        self.img_input = QtWidgets.QLabel("Imagem de Entrada")
        self.img_input.setFixedSize(400, 400)
        self.img_input.setStyleSheet("border: 1px solid black;")

        self.img_output = QtWidgets.QLabel("Imagem de Saída")
        self.img_output.setFixedSize(400, 400)
        self.img_output.setStyleSheet("border: 1px solid black;")

        img_layout = QtWidgets.QHBoxLayout()
        img_layout.addWidget(self.img_input)
        img_layout.addWidget(self.img_output)

        layout.addLayout(img_layout)
        self.setLayout(layout)


    def select_input_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecione a Pasta de Entrada")
        if folder:
            self.input_dir = folder
            self.lbl_input.setText(f"Entrada: {folder}")

    def select_output_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecione a Pasta de Saída")
        if folder:
            self.output_dir = folder
            self.lbl_output.setText(f"Saída: {folder}")

    def show_image(self, label, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def get_large_components(self, binary, min_area=1000):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        masks = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                masks.append((labels == i).astype(np.uint8))
        return masks

    def get_long_axis_endpoints(self, binary, prev_tail=None, prev_head=None, alpha=0.5):
        points = np.column_stack(np.nonzero(binary))
        if len(points) < 2:
            return np.array([0, 0]), np.array([0, 0])
        
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        threshold = np.percentile(distances, 95)
        filtered_points = points[distances <= threshold]

        if len(filtered_points) < 2:
            filtered_points = points

        pca = PCA(n_components=1)
        pca.fit(filtered_points)
        direction = pca.components_[0]
        proj = np.dot(filtered_points - center, direction)
        tail = center + direction * proj.min()
        head = center + direction * proj.max()

        tail = tail.astype(int)
        head = head.astype(int)

        # *** GARANTIR QUE CABEÇA E CAUDA NÃO SE INVERTAM ***
        # Exemplo: garantir que a cabeça tenha coordenada Y menor que a cauda (mais acima na imagem)
        if head[0] > tail[0]:  # supondo eixo y é o índice 0 (linha)
            # Inverter se estiver invertido
            head, tail = tail, head

        # Suavizar usando valores anteriores se existirem
        if prev_tail is not None and prev_head is not None:
            tail = (1 - alpha) * prev_tail + alpha * tail
            head = (1 - alpha) * prev_head + alpha * head
            tail = tail.astype(int)
            head = head.astype(int)

        return tail, head
    def generate_mask(self, binary, tail, head):
        h, w = binary.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # vetor e eixo principal
        vector = head - tail
        norm = np.linalg.norm(vector)
        if norm == 0:
            return mask
        direction = vector / norm

        # pontos da máscara e projeções
        points = np.column_stack(np.nonzero(binary))  # [(y,x),...]
        rel = points - tail
        proj = np.dot(rel, direction)

        proj_min, proj_max = proj.min(), proj.max()
        total_len = proj_max - proj_min

        # comprimentos: corpo ao redor do centróide, cabeça e cauda ficam com o resto
        # corpo ocupa 50% do comprimento total, centrado na projeção do centróide
        centroid_idx = int(np.median(np.where(binary > 0)[0])), int(np.median(np.where(binary > 0)[1]))
        # projeção do centróide
        cent_rel = np.array(centroid_idx) - tail
        cent_proj = np.dot(cent_rel, direction)

        corpo_len = total_len * 0.5
        semi_corpo = corpo_len / 2

        corpo_start = cent_proj - semi_corpo
        corpo_end   = cent_proj + semi_corpo

        # o que ficar abaixo de corpo_start é cauda, acima de corpo_end é cabeça
        for (y, x), p in zip(points, proj):
            if p < corpo_start:
                mask[y, x] = 3  # Cauda
            elif p > corpo_end:
                mask[y, x] = 1  # Cabeça
            else:
                mask[y, x] = 2  # Corpo

        return mask

  
    def process_images(self):
        if not self.input_dir or not self.output_dir:
            QtWidgets.QMessageBox.warning(self, "Erro", "Selecione as pastas de entrada e saída.")
            return
        self.use_sam = self.checkbox_sam.isChecked()

        prev_components = []  # Guarda máscaras dos componentes anteriores

        for file in sorted(os.listdir(self.input_dir)):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            input_path = os.path.join(self.input_dir, file)
            output_path = os.path.join(self.output_dir, os.path.splitext(file)[0] + ".png")

            print(f"Processando: {file}")
            img = cv2.imread(input_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_bg = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
            mask_animal = cv2.bitwise_not(mask_bg)

            components = self.get_large_components(mask_animal)
            if not components:
                continue

            # Função interna para obter centróide de uma máscara
            def centroid(mask):
                points = np.column_stack(np.nonzero(mask))
                return np.mean(points, axis=0) if len(points) > 0 else np.array([np.nan, np.nan])

            # Função para parear componentes atuais com anteriores
            def match_components(prev_comps, curr_comps, max_dist=50):
                matches = {}
                used_prev = set()
                for i, curr_comp in enumerate(curr_comps):
                    curr_c = centroid(curr_comp)
                    best_j = None
                    best_dist = float('inf')
                    for j, prev_comp in enumerate(prev_comps):
                        if j in used_prev:
                            continue
                        prev_c = centroid(prev_comp)
                        dist = np.linalg.norm(curr_c - prev_c)
                        if dist < best_dist and dist < max_dist:
                            best_dist = dist
                            best_j = j
                    if best_j is not None:
                        matches[i] = best_j
                        used_prev.add(best_j)
                return matches

            # Parear componentes
            matches = match_components(prev_components, components)

            h, w = mask_animal.shape
            final_mask = np.zeros((h, w), dtype=np.uint8)

            for i, comp in enumerate(components):
                try:
                    # Pega o índice do componente anterior correspondente (se existir)
                    prev_idx = matches.get(i, None)

                    # Pega pontos anteriores do dicionário se a correspondência existir
                    prev_tail, prev_head = (None, None)
                    if prev_idx is not None:
                        prev_tail, prev_head = self.prev_endpoints.get(prev_idx, (None, None))

                    # Ajusta alpha para suavizar conforme preferir
                    tail, head = self.get_long_axis_endpoints(comp, prev_tail, prev_head, alpha=0.4)

                    # Salva para próximo frame, mas usando o índice atual para não confundir
                    self.prev_endpoints[i] = (tail, head)

                    part_mask = self.generate_mask(comp, tail, head)
                    final_mask = np.maximum(final_mask, part_mask)
                except Exception as e:
                    print(f"Erro em componente: {e}")
                    continue

            # Atualiza componentes anteriores para próximo frame
            prev_components = components

            # Exibir imagens
            color_mask = np.zeros_like(img)
            color_mask[final_mask == 1] = (255, 0, 0)     # Cabeça
            color_mask[final_mask == 2] = (0, 255, 0)   # Corpo
            color_mask[final_mask == 3] = (0, 0, 255)     # Cauda

            self.show_image(self.img_input, img)
            self.show_image(self.img_output, color_mask)

            # Salvar máscara final
            cv2.imwrite(output_path, final_mask)
            QtWidgets.QApplication.processEvents()

        QtWidgets.QMessageBox.information(self, "Concluído", "Todas as imagens foram processadas.")

def process_with_sam(self, img):
    # Aqui você coloca seu código ou chamada para o Segment Anything Model
    # Exemplo dummy só para ilustrar:
    print("Processando com SAM ativado!")
    # processa a imagem com SAM e retorna imagem processada (color_mask, por ex)
    return img  # Retorna imagem ou máscara processada conforme necessário
def get_widget():
    return ThermalImageProcessor()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = ThermalImageProcessor()
    window.show()
    sys.exit(app.exec_())

