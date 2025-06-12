import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QComboBox, QFileDialog, QLineEdit, QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QPen, QKeySequence
from PyQt5.QtCore import Qt


NUM_CLASSES = 3
CLASS_COLORS = [
    (0,   0,   0),    # 0: fundo
    (255, 0,   0),    # 1: cabeça
    (0,   255, 0),    # 2: corpo
    (0,   0,   255),  # 3: cauda
]

def create_brush_cursor(size):
    diameter = size * 2
    pix = QPixmap(diameter+1, diameter+1)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    pen = QPen(Qt.black); pen.setWidth(2)
    p.setPen(pen)
    p.drawEllipse(0, 0, diameter, diameter)
    p.end()
    return QCursor(pix)

class MaskEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Editor PyQt5")
        self.setGeometry(100, 100, 700, 800)
        self.brush_size = 10
        self.current_class = 1
        self.drawing = False
        self.mask_history = []
        self.lock_background = False
        self.img_files = []
        self.index = 0

        # --- Configuração de pastas e imagem inicial ---
        cfg_layout = QHBoxLayout()

        self.input_folder = QLineEdit()
        bi = QPushButton("📁 Pasta Imagens")
        bi.clicked.connect(self.browse_input)
        cfg_layout.addWidget(self.input_folder)
        cfg_layout.addWidget(bi)

        self.mask_folder = QLineEdit()
        bm = QPushButton("📁 Pasta Máscaras")
        bm.clicked.connect(self.browse_mask)
        cfg_layout.addWidget(self.mask_folder)
        cfg_layout.addWidget(bm)

        self.save_folder = QLineEdit()
        bs = QPushButton("📁 Pasta Salvamento")
        bs.clicked.connect(self.browse_save)
        cfg_layout.addWidget(self.save_folder)
        cfg_layout.addWidget(bs)

        self.image_name = QLineEdit()
        self.image_name.setPlaceholderText("Nome da imagem (sem extensão)")
        ln = QPushButton("🔍 Carregar por nome")
        ln.clicked.connect(self.load_by_name)
        cfg_layout.addWidget(self.image_name)
        cfg_layout.addWidget(ln)

        # --- Layout principal ---
        main = QVBoxLayout()
        main.addLayout(cfg_layout)

        self.lbl = QLabel()
        self.lbl.setFixedSize(512, 512)
        main.addWidget(self.lbl)

        # --- Seleção de classe ---
        class_layout = QHBoxLayout()
        for i in range(NUM_CLASSES + 1):
            btn = QPushButton(f"Classe {i}")
            btn.clicked.connect(lambda _, c=i: self.set_class(c))
            class_layout.addWidget(btn)
        main.addLayout(class_layout)

        # --- Troca de classes ---
        repl = QHBoxLayout()
        self.cb_from = QComboBox(); self.cb_to = QComboBox()
        for i in range(NUM_CLASSES + 1):
            self.cb_from.addItem(f"{i}")
            self.cb_to.addItem(f"{i}")
        rr = QPushButton("🔄 Trocar")
        rr.clicked.connect(self.replace_class)
        repl.addWidget(self.cb_from)
        repl.addWidget(self.cb_to)
        repl.addWidget(rr)
        main.addLayout(repl)

        # --- Slider de pincel ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 50)
        self.slider.setValue(self.brush_size)
        self.slider.valueChanged.connect(self.change_brush)
        main.addWidget(self.slider)

        # --- Navegação e ações ---
        nav = QHBoxLayout()
        prev = QPushButton("← Anterior"); prev.clicked.connect(self.prev_image)
        nxt  = QPushButton("Próxima →"); nxt.clicked.connect(self.next_image)
        sv   = QPushButton("💾 Salvar"); sv.clicked.connect(self.save_mask)
        uz   = QPushButton("↶ Desfazer"); uz.clicked.connect(self.undo)
        lb   = QPushButton("🔒 Trava Fundo"); lb.setCheckable(True)
        lb.clicked.connect(self.toggle_lock)
        nav.addWidget(prev); nav.addWidget(nxt)
        nav.addWidget(sv); nav.addWidget(uz); nav.addWidget(lb)
        main.addLayout(nav)

        self.setLayout(main)
        self.setCursor(create_brush_cursor(self.brush_size))

             # --- Atalhos de teclado para navegação ---
        shortcut_prev = QShortcut(QKeySequence("A"), self)
        shortcut_prev.activated.connect(self.prev_image)

        shortcut_next = QShortcut(QKeySequence("D"), self)
        shortcut_next.activated.connect(self.next_image)

    def browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta Imagens")
        if d:
            self.input_folder.setText(d)
            self.img_files = sorted([f for f in os.listdir(d)
                                     if f.lower().endswith(('.png','.jpg','.jpeg'))])
            self.index = 0
            self.load_and_show()

    def browse_mask(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta Máscaras")
        if d:
            self.mask_folder.setText(d)

    def browse_save(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta Salvamento")
        if d:
            self.save_folder.setText(d)

    def load_by_name(self):
        name = self.image_name.text().strip()
        if not name: return
        for i,f in enumerate(self.img_files):
            if os.path.splitext(f)[0] == name:
                self.index = i
                self.load_and_show()
                return
        print("Não encontrada:", name)

    def set_class(self, c):
        self.current_class = c

    def change_brush(self, v):
        self.brush_size = v
        self.setCursor(create_brush_cursor(v))

    def load_and_show(self):
        if not self.img_files: return
        # carrega imagem
        ip = os.path.join(self.input_folder.text(), self.img_files[self.index])
        self.img = cv2.resize(cv2.imread(ip), (512,512))
        # carrega máscara
        bn = os.path.splitext(self.img_files[self.index])[0]
        mp = os.path.join(self.mask_folder.text(), bn+".png")
        mr = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        self.mask = (np.zeros((512,512),np.uint8) if mr is None
                     else cv2.resize(mr,(512,512),interpolation=cv2.INTER_NEAREST))
        self.mask_history.clear()
        self.update_display()

    def update_display(self):
        o = np.zeros_like(self.img, np.uint8)
        for c in range(NUM_CLASSES+1):
            o[self.mask==c] = CLASS_COLORS[c]
        blend = cv2.addWeighted(self.img,0.6,o,0.4,0)
        rgb = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape; bpl=ch*w
        qt = QImage(rgb.data,w,h,bpl,QImage.Format_RGB888)
        self.lbl.setPixmap(QPixmap.fromImage(qt))

    def mousePressEvent(self, e):
        if e.button()==Qt.LeftButton: self.drawing=True; self.paint(e.pos())

    def mouseMoveEvent(self, e):
        if self.drawing: self.paint(e.pos())

    def mouseReleaseEvent(self, e):
        if e.button()==Qt.LeftButton: self.drawing=False

    def paint(self, pos):
        x = pos.x()-self.lbl.pos().x(); y = pos.y()-self.lbl.pos().y()
        if 0<=x<512 and 0<=y<512:
            self.mask_history.append(self.mask.copy())
            if len(self.mask_history)>20: self.mask_history.pop(0)
            if self.lock_background:
                Y,X = np.ogrid[:512,:512]
                d = (X-x)**2+(Y-y)**2
                m = d<=self.brush_size**2
                self.mask[m & (self.mask!=0)] = self.current_class
            else:
                cv2.circle(self.mask,(x,y),self.brush_size,self.current_class,-1)
            self.update_display()

    def undo(self):
        if self.mask_history:
            self.mask = self.mask_history.pop()
            self.update_display()

    def toggle_lock(self):
        self.lock_background = not self.lock_background

    def prev_image(self):
        if self.index > 0:
            self.save_mask()  # salva a máscara atual
            self.index -= 1
            self.load_and_show()

    def next_image(self):
        if self.index < len(self.img_files) - 1:
            self.save_mask()  # salva a máscara atual
            self.index += 1
            self.load_and_show()

    def save_mask(self):
        sf = self.save_folder.text().strip()
        if not sf: return
        os.makedirs(sf, exist_ok=True)
        bn = os.path.splitext(self.img_files[self.index])[0]
        fp = os.path.join(sf, f"{bn}.png")
        cv2.imwrite(fp, self.mask)
        print("Salvo em", fp)

    def replace_class(self):
        fc = self.cb_from.currentIndex()
        tc = self.cb_to.currentIndex()
        if fc==tc: return
        self.mask_history.append(self.mask.copy())
        self.mask[self.mask==fc] = tc
        self.update_display()
def get_widget():
    return MaskEditor()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MaskEditor()
    window.show()
    sys.exit(app.exec_())

