# Imagem base com PyTorch e CUDA pré-instalado
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Atualiza sistema e instala dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxkbcommon-x11-0 \
    qt5-default \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Copia e instala os requisitos
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Instala detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Exponha a porta para FastAPI
EXPOSE 8000

# Comando padrão (você pode mudar para rodar outro script se quiser)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
