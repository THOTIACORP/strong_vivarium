import subprocess
import platform
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
import locale
import webbrowser
import threading
INSTALL_FLAG = "installed.flag"
IS_WINDOWS = platform.system() == "Windows"

def get_language():
    lang = locale.getdefaultlocale()[0]
    if lang and "pt" in lang.lower():
        return "pt"
    return "en"

LANG = get_language()

TEXTS = {
    "title": {"pt": "Biotério Forte", "en": "Installer"},
    "choose": {"pt": "Escolha o tipo de instalação:", "en": "Choose installation type:"},
    "btn_docker": {"pt": "Instalar COM Docker", "en": "Install WITH Docker"},
    "btn_nodocker": {"pt": "Instalar SEM Docker", "en": "Install WITHOUT Docker"},
    "already_installed": {"pt": "O sistema já foi instalado.", "en": "System is already installed."},
    "success": {"pt": "Instalação concluída com sucesso!", "en": "Installation completed successfully!"},
    "error": {"pt": "Erro ao executar o instalador.", "en": "Error during installation."},
    "exit": {"pt": "Sair", "en": "Exit"},
    "website": {"pt": "Visite nosso site", "en": "Visit our website"},
    "accept_license": {"pt": "Li e aceito os termos da licença", "en": "I have read and accept the license terms"},
    "continue": {"pt": "Continuar", "en": "Continue"},
    "license_title": {"pt": "Licença de Uso de Dados", "en": "Strong Vivarium: Data Usage License"},
}

if os.path.exists(INSTALL_FLAG):
    messagebox.showinfo(TEXTS["title"][LANG], TEXTS["already_installed"][LANG])
    exit()
def finish_install():
    messagebox.showinfo("✅", TEXTS["success"][LANG])
    install_window.quit()    # Sai do mainloop
    install_window.destroy() # Fecha a janela

def run_script(script_name):
    full_cmd = ["bash", script_name] if not IS_WINDOWS else [script_name]
    try:
        process = subprocess.Popen(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=IS_WINDOWS
        )
        # Limpar log na segunda tela
        install_window.log_area.delete("1.0", tk.END)
        for line in process.stdout:
            install_window.log_area.insert(tk.END, line)
            install_window.log_area.see(tk.END)
        process.wait()

        if process.returncode == 0:
            with open(INSTALL_FLAG, "w") as f:
                f.write("installed")
                messagebox.showinfo("✅", TEXTS["success"][LANG])
                install_window.after(0, finish_install)
        else:
            messagebox.showerror("❌", TEXTS["error"][LANG])
    except Exception as e:
        messagebox.showerror("Erro", str(e))

def open_website(event=None):
    webbrowser.open_new("https://www.thotiacorp.com.br/")

def show_license_text(text_widget):
    license_text = (
        "📜 Biotério Forte: LICENÇA DE USO DE DADOS\n\n"
        "Este conjunto de dados está licenciado sob a Creative Commons Atribuição-NãoComercial 4.0 Internacional (CC BY-NC 4.0).\n\n"
        "✅ Gratuito para uso não comercial com atribuição ao autor original.\n"
        "❌ O uso comercial não é permitido sem autorização prévia.\n\n"
        "Autor: Ronnei Borges\n"
        "Fonte: Kaggle - https://www.kaggle.com/datasets/ronneiborges/thermal-images-of-rats-mice-for-segmentation\n\n"
        "Mais detalhes sobre a licença:\n"
        "https://creativecommons.org/licenses/by-nc/4.0/deed.pt\n"
        "-----------------------------------------------\n"
    )
    text_widget.insert(tk.END, license_text)
    text_widget.see(tk.END)

# --- TELA 1: Licença ---

license_window = tk.Tk()
license_window.title(TEXTS["title"][LANG])
license_window.geometry("520x440")
license_window.resizable(False, False)
if IS_WINDOWS:
    try:
        license_window.iconbitmap("icon.ico")  # Ícone na primeira tela
    except Exception as e:
        print(f"Erro ao carregar ícone na tela de licença: {e}")

frame_license = tk.Frame(license_window, padx=20, pady=15)
frame_license.pack(expand=True, fill=tk.BOTH)

label_license = tk.Label(frame_license, text=TEXTS["license_title"][LANG], font=("Segoe UI", 14, "bold"))
label_license.pack(pady=(0, 15))

license_text_area = scrolledtext.ScrolledText(frame_license, width=60, height=15, font=("Consolas", 10))
license_text_area.pack()

show_license_text(license_text_area)

license_accepted = tk.BooleanVar(value=False)

def toggle_continue():
    btn_continue.config(state=tk.NORMAL if license_accepted.get() else tk.DISABLED)

chk_license = tk.Checkbutton(
    frame_license, text=TEXTS["accept_license"][LANG], variable=license_accepted,
    command=toggle_continue, font=("Segoe UI", 10)
)
chk_license.pack(pady=(10, 10))

def open_install_window():
    license_window.destroy()
    open_install_screen()

btn_continue = tk.Button(
    frame_license, text=TEXTS["continue"][LANG], state=tk.DISABLED,
    command=open_install_window,
    bg="#2196F3", fg="white", activebackground="#1e88e5",
    font=("Segoe UI", 11), width=12, bd=0, relief="ridge", cursor="hand2"
)
btn_continue.pack(pady=10)

link = tk.Label(
    frame_license, text=TEXTS["website"][LANG] + " →",
    font=("Segoe UI", 10, "underline"), fg="#0645AD", cursor="hand2"
)
link.pack(side=tk.BOTTOM, pady=8)
link.bind("<Button-1>", open_website)

# --- TELA 2: Instalação ---


def animate_installing(label, dots=0):
    if getattr(label, "stop_animation", False):
        label.config(text="")
        return
    dots = (dots + 1) % 4
    label.config(text="Instalando" + "." * dots)
    label.after(500, animate_installing, label, dots)

def run_script_threaded(script_name, label):
    def target():
        try:
            full_cmd = ["bash", script_name] if not IS_WINDOWS else [script_name]
            process = subprocess.Popen(
                full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=IS_WINDOWS
            )
            install_window.log_area.delete("1.0", tk.END)
            captured_output = []

            for line in process.stdout:
                install_window.log_area.insert(tk.END, line)
                install_window.log_area.see(tk.END)
                captured_output.append(line)

            process.wait()

            if process.returncode == 0:
                full_output = "".join(captured_output).lower()
                if "instalação binarizada" in full_output:
                    with open(INSTALL_FLAG, "w") as f:
                        f.write("installed")

                        install_window.after(0, finish_install)



                else:
                    messagebox.showwarning("⚠️", "Instalação concluída, mas sem confirmação binarizada.")
            else:
                messagebox.showerror("❌", TEXTS["error"][LANG])

        except Exception as e:
            messagebox.showerror("Erro", str(e))
        finally:
            label.stop_animation = True
            label.config(text="")

    label.stop_animation = False
    animate_installing(label)
    thread = threading.Thread(target=target, daemon=True)
    thread.start()


def run_full_install():
    script_path = os.path.join("src", "install", "install_nodocker.bat")
    run_script_threaded(script_path, install_window.install_label)

def open_install_screen():
    global install_window
    install_window = tk.Tk()
    install_window.title(TEXTS["title"][LANG])
    install_window.geometry("520x500")  # mais alto para caber tudo
    install_window.resizable(False, False)

    if IS_WINDOWS:
        try:
            install_window.iconbitmap("icon.ico")
        except Exception as e:
            print(f"Erro ao carregar ícone: {e}")

    frame = tk.Frame(install_window, padx=20, pady=15)
    frame.pack(expand=True, fill=tk.BOTH)

    label = tk.Label(frame, text=TEXTS["choose"][LANG], font=("Segoe UI", 14, "bold"))
    label.pack(pady=(0, 15))

    install_type = tk.StringVar(value="full")  # padrão: completa

    radio_full = tk.Radiobutton(
        frame, text="Instalação Completa", variable=install_type, value="full",
        font=("Segoe UI", 11))
    radio_full.pack(anchor="w")

    radio_custom = tk.Radiobutton(
        frame, text="Instalação Customizada", variable=install_type, value="custom",
        font=("Segoe UI", 11))
    radio_custom.pack(anchor="w")

    # Frame para opções customizadas filhos
    custom_frame = tk.Frame(frame, padx=30, pady=10)
    custom_frame.pack(fill=tk.X, pady=(10, 0))

    # Variáveis para checkbuttons dos filhos (modelos)
    native_freedom_var = tk.BooleanVar()
    native_housing_var = tk.BooleanVar()
    sam_model1_var = tk.BooleanVar()
    sam_model2_var = tk.BooleanVar()
    sam_model3_var = tk.BooleanVar()

    # Modelos Nativos
    native_label = tk.Label(custom_frame, text="Modelos Nativos", font=("Segoe UI", 12, "bold"))
    native_cb1 = tk.Checkbutton(custom_frame, text="Freedom", variable=native_freedom_var, font=("Segoe UI", 11))
    native_cb2 = tk.Checkbutton(custom_frame, text="Housing", variable=native_housing_var, font=("Segoe UI", 11))

    # Modelos SAM
    sam_label = tk.Label(custom_frame, text="Outros Modelos", font=("Segoe UI", 12, "bold"))
    sam_cb1 = tk.Checkbutton(custom_frame, text="SAM vit_h", variable=sam_model1_var, font=("Segoe UI", 11))
    sam_cb2 = tk.Checkbutton(custom_frame, text="SAM vit_l", variable=sam_model2_var, font=("Segoe UI", 11))
    sam_cb3 = tk.Checkbutton(custom_frame, text="SAM vit_b", variable=sam_model3_var, font=("Segoe UI", 11))

    # Função para mostrar ou esconder as opções customizadas
    def toggle_custom_options():
        if install_type.get() == "custom":
            native_label.pack(anchor="w")
            native_cb1.pack(anchor="w", padx=20)
            native_cb2.pack(anchor="w", padx=20)

            sam_label.pack(anchor="w", pady=(10, 0))
            sam_cb1.pack(anchor="w", padx=20)
            sam_cb2.pack(anchor="w", padx=20)
            sam_cb3.pack(anchor="w", padx=20)
        else:
            native_label.pack_forget()
            native_cb1.pack_forget()
            native_cb2.pack_forget()

            sam_label.pack_forget()
            sam_cb1.pack_forget()
            sam_cb2.pack_forget()
            sam_cb3.pack_forget()

    # Sempre chama para ajustar inicial
    toggle_custom_options()

    # Chama toggle sempre que o radio mudar
    radio_full.config(command=toggle_custom_options)
    radio_custom.config(command=toggle_custom_options)

    btn_style = {
        "font": ("Segoe UI", 11),
        "width": 25,
        "bd": 0,
        "relief": "ridge",
        "cursor": "hand2"
    }
    
    def run_custom_install():
        # Aqui você pega as opções selecionadas para rodar seus scripts
        selecionados = []
        if native_freedom_var.get():
            selecionados.append("Freedom")
        if native_housing_var.get():
            selecionados.append("Housing")
        if sam_model1_var.get():
            selecionados.append("SAM 1")
        if sam_model2_var.get():
            selecionados.append("SAM 2")
        if sam_model3_var.get():
            selecionados.append("SAM 3")

        if not selecionados:
            messagebox.showwarning("Atenção", "Selecione pelo menos um modelo para instalação customizada.")
            return

        # Exemplo: só mostra selecionados por enquanto
        messagebox.showinfo("Modelos selecionados", "Você selecionou: " + ", ".join(selecionados))

        # Aqui você pode executar o script conforme selecionados:
        # run_script("seu_script_based_on_modelos.sh")  # ajustar conforme
    # Label do spin (instalando...)
    install_window.install_label = tk.Label(frame, text="", font=("Segoe UI", 12, "italic"), fg="green")
    install_window.install_label.pack(pady=(10, 10))


    # Botões normais para instalação completa e sem docker
    btn_nodocker = tk.Button(
        frame, text=TEXTS["btn_nodocker"][LANG], command=run_full_install, bg="#4CAF50", fg="white",
        activebackground="#45a049", **btn_style)
    btn_nodocker.pack(pady=6)

    btn_docker = tk.Button(
        frame, text=TEXTS["btn_docker"][LANG],
        command=lambda: run_script("install_docker.bat" if IS_WINDOWS else "install_docker.sh"),
        bg="#2196F3", fg="white", activebackground="#1e88e5", **btn_style)
    btn_docker.pack(pady=6)

    # habilita botões só se licença aceita (já aceita pois veio da tela anterior)
    btn_nodocker.config(state=tk.NORMAL)
    btn_docker.config(state=tk.NORMAL)
  
    # Área de log
    log_label = tk.Label(frame, text="Logs de instalação:", font=("Segoe UI", 12, "bold"))
    log_label.pack(pady=(20, 0), anchor="w")

    install_window.log_area = scrolledtext.ScrolledText(frame, width=60, height=10, font=("Consolas", 10))
    install_window.log_area.pack(pady=(5, 20))

    btn_exit = tk.Button(
        frame, text=TEXTS["exit"][LANG], command=install_window.destroy,
        bg="#f44336", fg="white", activebackground="#d32f2f",
        font=("Segoe UI", 11), width=10, bd=0, relief="ridge", cursor="hand2"
    )
    btn_exit.pack(pady=6)

    link = tk.Label(
        install_window, text=TEXTS["website"][LANG] + " →",
        font=("Segoe UI", 10, "underline"), fg="#0645AD", cursor="hand2"
    )
    link.pack(side=tk.BOTTOM, pady=8)
    link.bind("<Button-1>", open_website)

    install_window.mainloop()


license_window.mainloop()
