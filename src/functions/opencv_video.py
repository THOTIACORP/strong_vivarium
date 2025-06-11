import cv2
import os
import glob

# Caminho da pasta com os vídeos
video_folder = "videos"  # Ex: "videos/"
output_video = "video_end.mp4"

# Pegando todos os vídeos suportados pela extensão
video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))

if not video_files:
    print("Nenhum vídeo encontrado na pasta.")
    exit()

# Ler propriedades do primeiro vídeo
first_cap = cv2.VideoCapture(video_files[0])
fps = first_cap.get(cv2.CAP_PROP_FPS)
width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
first_cap.release()

# Criar o objeto de gravação
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Loop para processar todos os vídeos
for video_path in video_files:
    print(f"Processando: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()

out.release()
print(f"Vídeo final salvo como: {output_video}")
