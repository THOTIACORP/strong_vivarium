import sys
import platform
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QTextEdit, QHBoxLayout,
    QComboBox, QListWidget, QListWidgetItem,
    QStackedWidget
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

IS_WINDOWS = platform.system() == "Windows"

SCRIPTS = {
    "🎞️ Extract Frames": "opencv.py",
    "🧬 Pseudo-Classes": "pseud-class.py",
    "🕵️ View Mask Debug": "view_mask_debug.py",
    "🎨 Edit Masks (GUI)": "edit_mask_app.py",
    "🦊 Transforms Fake": "transforms_fake.py",
    "🦊 Transforms Fake Multiple": "transforms_fake_multiple.py",
    "🧠 Train Unet": "train_unet.py",
    "🧠 Train Detectron": "train_detectron.py",
    "🧠 Inferences": "loop_api_unet.py"
}


class ScriptTab(QWidget):
    def __init__(self, title, script_name):
        super().__init__()
        self.script_name = script_name
        self.process = None

        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignTop)

        self.title_label = QLabel(f"{title}")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setContentsMargins(
            0, 0, 0, 0)  # remove padding interno
        # remove margin/padding externos
        self.title_label.setStyleSheet("margin: 0px; padding: 0px;")

        self.run_button = QPushButton("▶ Run")
        self.run_button.clicked.connect(self.run_script)

        self.stop_button = QPushButton("■ Parar")
        self.stop_button.clicked.connect(self.stop_script)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        self.output_text.setFixedHeight(150)

        # Container para a interface embutida (GUI do script)
        self.embedded_widget_container = QVBoxLayout()

        # Adicione na ordem correta:
        self.layout.addWidget(self.title_label)              # Título no topo
        # Interface embutida logo abaixo do título
        self.layout.addLayout(self.embedded_widget_container)
        # Botões abaixo da interface embutida
        self.layout.addLayout(button_layout)
        # Console sempre abaixo dos botões
        self.layout.addWidget(self.output_text)

        self.setLayout(self.layout)

    def run_script(self):
        if self.process:
            self.output_text.append("⚠️ Já em execução...")
            return

        self.output_text.clear()

        try:
            # Importa o módulo do script e verifica se ele retorna um QWidget
            module = __import__(self.script_name.replace(".py", ""))
            if hasattr(module, "get_widget"):
                widget = module.get_widget()
                if isinstance(widget, QWidget):
                    # Limpa anterior
                    while self.embedded_widget_container.count():
                        old = self.embedded_widget_container.takeAt(0)
                        if old.widget():
                            old.widget().deleteLater()

                    self.embedded_widget_container.addWidget(widget)
                    self.output_text.append(
                        "✅ Interface carregada com sucesso.")
                else:
                    self.output_text.append(
                        "❌ O script não retornou um QWidget válido.")
            else:
                self.output_text.append(
                    "❌ O script não possui função 'get_widget()'.")
        except Exception as e:
            self.output_text.append(f"❌ Erro ao importar o script: {str(e)}")

    def stop_script(self):
        if self.process:
            self.process.kill()  # Termina o processo
            self.process = None
            self.output_text.append("🟥 Processo parado.")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        # Também limpar UI embutida, se quiser
        while self.embedded_widget_container.count():
            old = self.embedded_widget_container.takeAt(0)
            if old.widget():
                old.widget().deleteLater()
        self.output_text.append("🟥 Interface removida.")


class SettingsTab(QWidget):
    def __init__(self, apply_theme_callback):
        super().__init__()

        layout = QVBoxLayout()
        layout.addStretch()

        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["🌞 Tema Claro", "🌙 Tema Escuro"])
        self.theme_selector.currentIndexChanged.connect(apply_theme_callback)

        layout.addWidget(QLabel("Tema:"))
        layout.addWidget(self.theme_selector)

        layout.addStretch()

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biotério Forte - Painel de Ferramentas")
        self.setWindowIcon(QIcon("../../icon.ico"))
        self.setGeometry(100, 100, 900, 600)

        # Layout principal
        main_layout = QHBoxLayout()

        # Menu lateral
        self.menu = QListWidget()
        self.menu.setFixedWidth(200)
        self.menu.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 14px;
                border: none;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                font-weight: bold;
            }
        """)

        self.stack = QStackedWidget()

        # Preencher menu e stack com scripts
        for idx, (title, script) in enumerate(SCRIPTS.items()):
            item = QListWidgetItem(title)
            self.menu.addItem(item)
            self.stack.addWidget(ScriptTab(title, script))

        # Adiciona item "Configurações" no menu e sua aba
        self.settings_tab = SettingsTab(self.apply_theme)
        settings_item = QListWidgetItem("⚙ Configurações")
        self.menu.addItem(settings_item)
        self.stack.addWidget(self.settings_tab)

        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.menu.setCurrentRow(0)

        right_side = QVBoxLayout()
        right_side.addWidget(self.stack)

        main_layout.addWidget(self.menu)
        main_layout.addLayout(right_side)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.apply_theme(1)  # Dark Theme as default

    def apply_theme(self, index):
        if index == 1:
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; color: #f0f0f0; }
                QLabel, QPushButton, QComboBox {
                    color: #f0f0f0;
                    background-color: #3c3f41;
                }
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #ffffff; }
                QLabel, QPushButton, QComboBox {
                    background-color: #f5f5f5;
                    color: #000000;
                }
                QTextEdit {
                    background-color: #f5f5f5;
                    color: #000000;
                }
            """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
