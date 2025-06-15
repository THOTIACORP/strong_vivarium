from PIL import ImageDraw, ImageFont, Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
import numpy as np
import io
import torch.nn as nn
import base64
import cv2
import tempfile
import cv2
import os
import math
app = FastAPI(title="Strong Vivarium: Thermal Mouse Analytics API")

# UNet modelo (mantido igual) treinamento


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def center_crop(self, enc_feat, target_size):
        _, _, h, w = target_size
        _, _, H, W = enc_feat.size()
        delta_h = (H - h) // 2
        delta_w = (W - w) // 2
        return enc_feat[:, :, delta_h:delta_h+h, delta_w:delta_w+w]

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.center_crop(enc4, dec4.size())
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.center_crop(enc3, dec3.size())
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.center_crop(enc2, dec2.size())
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.center_crop(enc1, dec1.size())
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.conv_last(dec1)


# Carregando modelo treinado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=4)
checkpoint = torch.load(
    "../models/model_unet_freedom/model_unet_freedon.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def add_displacement_text(map_img_np, deslocamentos_totais, class_names, class_colors):
    img = map_img_np.copy()

    img_height, img_width = img.shape[:2]

    # 🔤 Configurações de fonte
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = (img_width / 1200 * 1.2) / 3  # 🔸 3x menor
    base_thickness = max(int(img_width / 500 / 3), 1)  # 🔸 3x menor

    # 🔖 Título
    title_text = "Displacement Map"
    (tw, th), _ = cv2.getTextSize(title_text, font, base_font_scale * 1.5, base_thickness + 1)
    title_x = (img_width - tw) // 2
    title_y = int(0.02 * img_width)  # 🔸 também ajustado para ficar mais próximo do topo

    # 🗂️ Caixa branca atrás do título
    padding = int(10 / 3)
    cv2.rectangle(
        img,
        (title_x - padding, title_y - th - padding),
        (title_x + tw + padding, title_y + padding),
        (255, 255, 255),
        -1
    )

    # ✍️ Texto do título
    cv2.putText(
        img,
        title_text,
        (title_x, title_y),
        font,
        base_font_scale * 1.5,
        (0, 0, 0),
        base_thickness + 1,
        cv2.LINE_AA
    )

    # 📦 Textos das classes
    textos = []
    for cls, desloc in deslocamentos_totais.items():
        nome = class_names.get(cls, str(cls))
        texto = f"{nome}: total displacement = {desloc:.2f} px"
        textos.append((cls, texto))

    # 📐 Cálculo de caixa de fundo dos textos
    x_text = int(0.02 * img_width)
    y_text = title_y + padding * 4

    spacing = int(0.01 * img_width / 3)
    quad_size = int(0.02 * img_width / 3)
    pad_x = int(10 / 3)
    pad_y = int(10 / 3)

    largura_max = 0
    altura_linha = 0

    for _, texto in textos:
        (w, h), _ = cv2.getTextSize(texto, font, base_font_scale, base_thickness)
        largura_max = max(largura_max, w)
        altura_linha = max(altura_linha, h)

    altura_total = len(textos) * (altura_linha + spacing) - spacing

    caixa_x0 = x_text - pad_x
    caixa_y0 = y_text - pad_y
    caixa_x1 = x_text + quad_size + 10 + largura_max + pad_x
    caixa_y1 = y_text + altura_total + pad_y

    # 🗂️ Caixa de fundo
    cv2.rectangle(
        img,
        (caixa_x0, caixa_y0),
        (caixa_x1, caixa_y1),
        (255, 255, 255),
        -1
    )

    # 🔳 Quadrados coloridos + textos
    yy = y_text
    for cls, texto in textos:
        color = class_colors.get(cls, (0, 0, 0))

        # 🎨 Quadrado colorido
        quad_x0 = x_text
        quad_y0 = yy
        quad_x1 = quad_x0 + quad_size
        quad_y1 = quad_y0 + quad_size

        cv2.rectangle(img, (quad_x0, quad_y0), (quad_x1, quad_y1), color, -1)

        # ✍️ Texto
        text_x = quad_x1 + 5  # 🔸 diminui o espaçamento lateral
        text_y = yy + quad_size

        cv2.putText(
            img,
            texto,
            (text_x, text_y - 2),  # 🔸 ajuste fino vertical
            font,
            base_font_scale,
            (0, 0, 0),
            base_thickness,
            cv2.LINE_AA
        )

        yy += altura_linha + spacing

    return img


def track_mouse_movement_in_video(video_bytes, model, device, transform,
                                  target_classes=[1, 2, 3], batch_size=10,  fps_process=5 ):
    """
    Processa vídeo (bytes) com modelo UNet que retorna mask logits em shape [batch, num_classes, Hm, Wm].
    Calcula centróides por classe, converte para coordenadas do frame original e gera mapas de rastros.
    Retorna um dict com deslocamentos, mapa_base e mapa_realista numpy arrays BGR, e caminhos por classe.
    """
    # 1. Grava temporário
    print("⏳ Criando arquivo temporário de vídeo...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    # 2. Abre vídeo
    print("📹 Abrindo vídeo para leitura...")
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise ValueError("❌ Não foi possível abrir o vídeo.")

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ FPS do vídeo: {video_fps}, Total de frames: {total_frames}")
    print(f"🖼️ Resolução: {video_width}x{video_height}")
      # Calcula salto de frames
    frame_skip = max(1, math.floor(video_fps / fps_process))
    print(f"⏩ Processando 1 frame a cada {frame_skip} frames (FPS desejado: {fps_process})")

  
    positions_per_class = {cls: [] for cls in target_classes}
    frame_count = 0
    processed_frame_index = 0

    batch_frames = []
    batch_frame_indices = []

    # Para saber dimensões de saída da IA (mask)
    # Faz uma inferência de amostra no primeiro frame:
    ret0, frame0 = cap.read()
    if not ret0:
        cap.release()
        raise ValueError("Vídeo vazio ou não pôde ler primeiro frame.")
    # Prepara tensor de amostra
    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    pil0 = Image.fromarray(frame0_rgb).convert("RGB")
    tensor0 = transform(pil0).unsqueeze(0).to(device)
    with torch.no_grad():
        out0 = model(tensor0)
    # out0 shape: [1, num_classes, Hm, Wm]
    _, num_classes, mask_h, mask_w = out0.shape
    print(f"Saída do modelo (máscara): {out0.shape}")
    # Calcula fatores de escala de máscara para frame original:
    scale_x = video_width / mask_w
    scale_y = video_height / mask_h
    print(f"Scale mask->frame: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # Retorna ao início
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("🚀 Iniciando processamento dos frames em batches...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🏁 Fim do vídeo.")
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).convert("RGB")
        input_tensor = transform(pil_image)  # supõe resize interno para mask_w x mask_h
        batch_frames.append(input_tensor)
        batch_frame_indices.append(frame_count)

        if len(batch_frames) == batch_size:
            # Pilha e inferência
            input_batch = torch.stack(batch_frames).to(device)
            with torch.no_grad():
                output = model(input_batch)  # shape [B, C, mask_h, mask_w]
                probs = torch.softmax(output, dim=1).cpu().numpy()  # [B, C, Hm, Wm]

            # Para cada frame no batch
            for i in range(len(batch_frames)):
                for cls in target_classes:
                    # Máscara booleana no espaço mask_h x mask_w
                    pred_mask = (np.argmax(probs[i], axis=0) == cls)
                    # coords na máscara
                    coords = np.column_stack(np.where(pred_mask))
                    current_position = None
                    if len(coords) > 0:
                        y_mean, x_mean = coords.mean(axis=0)
                        # Converte para coordenada do frame original:
                        x_orig = int(x_mean * scale_x)
                        y_orig = int(y_mean * scale_y)
                        current_position = (x_orig, y_orig)
                        print(f"Classe {cls} - Frame {batch_frame_indices[i]}: Detectado em {current_position}")
                    else:
                        print(f"Classe {cls} - Frame {batch_frame_indices[i]}: Não detectado.")
                    positions_per_class[cls].append(current_position)

            batch_frames = []
            batch_frame_indices = []

        frame_count += 1

    # Processa últimos frames restantes
    if batch_frames:
        input_batch = torch.stack(batch_frames).to(device)
        with torch.no_grad():
            output = model(input_batch)
            probs = torch.softmax(output, dim=1).cpu().numpy()

        for i in range(len(batch_frames)):
            for cls in target_classes:
                pred_mask = (np.argmax(probs[i], axis=0) == cls)
                coords = np.column_stack(np.where(pred_mask))
                current_position = None
                if len(coords) > 0:
                    y_mean, x_mean = coords.mean(axis=0)
                    x_orig = int(x_mean * scale_x)
                    y_orig = int(y_mean * scale_y)
                    current_position = (x_orig, y_orig)
                    print(f"Classe {cls} - Frame {batch_frame_indices[i]}: Detectado em {current_position}")
                else:
                    print(f"Classe {cls} - Frame {batch_frame_indices[i]}: Não detectado.")
                positions_per_class[cls].append(current_position)

    cap.release()

    # Calcula deslocamentos totais
    deslocamentos_totais = {}
    for cls in target_classes:
        total_move = 0.0
        pos_list = positions_per_class[cls]
        for i in range(1, len(pos_list)):
            p0 = pos_list[i-1]
            p1 = pos_list[i]
            if p0 and p1:
                total_move += np.linalg.norm(np.array(p1) - np.array(p0))
        deslocamentos_totais[cls] = total_move
        print(f"Deslocamento total classe {cls}: {total_move:.2f} px")

    head_path = positions_per_class.get(1, [])
    body_path = positions_per_class.get(2, [])
    tail_path = positions_per_class.get(3, [])

    # Gera mapa de rastros com dimensões do vídeo
    print("🗺️ Gerando mapa de rastros realista...")
    map_img = np.ones((video_height, video_width, 3), dtype=np.uint8) * 255

    def draw_path(path, color, thickness=2):
        pts = [pt for pt in path if pt is not None]
        if len(pts) >= 2:
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(map_img, [arr], isClosed=False, color=color, thickness=thickness)


    def draw_endpoints(path, color):
        pts = [pt for pt in path if pt is not None]
        if pts:
            cv2.circle(map_img, pts[0], 5, color, -1)
            cv2.circle(map_img, pts[-1], 5, (0, 0, 0), -1)

    # Desenha em cores BGR: cabeça azul, corpo verde, cauda vermelho
    draw_path(head_path, (255, 0, 0))
    draw_path(body_path, (0, 255, 0))
    draw_path(tail_path, (0, 0, 255))
    draw_endpoints(head_path, (255, 0, 0))
    draw_endpoints(body_path, (0, 255, 0))
    draw_endpoints(tail_path, (0, 0, 255))

   
    # Adiciona textos de deslocamento numa caixa branca
    class_names = {1: "Head", 2: "Body", 3: "Tail"}
    class_colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
    map_img = add_displacement_text(map_img, deslocamentos_totais, class_names, class_colors)

    cv2.imwrite("rastro_mouse_com_deslocamento.png", map_img)

    # Sobrepõe ao último frame
    print("📷 Sobrepondo rastro ao último frame real do vídeo...")
    cap2 = cv2.VideoCapture(temp_video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, last_frame = cap2.read()
    cap2.release()

    if ret:
        overlay = last_frame.copy()
        # Desenha trajetórias no overlay
        def draw_on(img, path, color, thickness=2):
            pts = [pt for pt in path if pt is not None]
            if len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        def draw_ep(img, path, color):
            pts = [pt for pt in path if pt is not None]
            if pts:
                cv2.circle(img, pts[0], 5, color, -1)
                cv2.circle(img, pts[-1], 5, (0, 0, 0), -1)

        draw_on(overlay, head_path, (255, 0, 0))
        draw_on(overlay, body_path, (0, 255, 0))
        draw_on(overlay, tail_path, (0, 0, 255))
        draw_ep(overlay, head_path, (255, 0, 0))
        draw_ep(overlay, body_path, (0, 255, 0))
        draw_ep(overlay, tail_path, (0, 0, 255))

        # Usa transparência entre overlay e map_img, ambos mesmo tamanho
        mapa_realista = cv2.addWeighted(overlay, 0.5, map_img, 0.5, 0)
        cv2.imwrite("rastro_mouse_realista.png", mapa_realista)
    else:
        mapa_realista = map_img
    os.remove(temp_video_path)
    print("✅ Processamento concluído com sucesso!")
    return {
        "movimentos": deslocamentos_totais,
        "mapa_base": map_img,
        "mapa_realista": mapa_realista,
        "head_path": head_path,
        "body_path": body_path,
        "tail_path": tail_path,
         "video_info": {
            "fps_original": video_fps,
            "fps_processado": fps_process,
            "frame_skip": frame_skip,
            "resolution": (video_width, video_height),
            "total_frames": total_frames
        }
    }



def convert_ndarrays_to_lists(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarrays_to_lists(item) for item in data]
    else:
        return data


@app.post("/unet_freedom_track_image", summary="Track mouse in UNET image",  tags=["UNET"])
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(torch.softmax(output, dim=1),
                                dim=1).squeeze().cpu().numpy()

        unique, counts = np.unique(pred, return_counts=True)
        pixel_counts = {str(int(cls)): int(count)
                        for cls, count in zip(unique, counts)}

        # Mapeamento de cores e nomes das classes
        colors = {
            0: (0, 0, 0),        # Fundo
            1: (0, 0, 255),      # Cauda
            2: (0, 255, 0),      # Corpo
            3: (255, 0, 0),      # Cabeça
        }

        class_names = {
            0: "Fundo",
            1: "Cauda",
            2: "Corpo",
            3: "Cabeça"
        }

        # Criação da máscara
        mask_img = Image.new("RGBA", (pred.shape[1], pred.shape[0]))
        mask_pixels = mask_img.load()

        first_occurrence = {}

        for y in range(pred.shape[0]):
            for x in range(pred.shape[1]):
                cls = pred[y, x]
                mask_pixels[x, y] = colors[cls] + (120,)  # transparência
                if cls not in first_occurrence:
                    first_occurrence[cls] = (x, y)

        # Combinação com imagem original
        img_resized = img.resize((512, 512)).convert("RGBA")
        combined = Image.alpha_composite(img_resized, mask_img)
        # Criação de um painel moderno no canto superior direito
        # Criação de um painel lateral moderno com cantos arredondados
        panel_width = 240
        panel_height = 35 * len(pixel_counts) + 30
        combined_with_panel = Image.new(
            "RGBA", (combined.width + panel_width,
                     combined.height), (255, 255, 255, 0)
        )
        combined_with_panel.paste(combined, (0, 0))

        draw_panel = ImageDraw.Draw(combined_with_panel)

        panel_x = combined.width + 10
        panel_y = 20
        radius = 20

        # Desenho do painel com cantos arredondados (moderno)
        def draw_rounded_rectangle(draw, xy, radius, fill, outline=None, width=1):
            x1, y1, x2, y2 = xy
            draw.rounded_rectangle(
                xy, radius=radius, fill=fill, outline=outline, width=width)

        draw_rounded_rectangle(
            draw_panel,
            [panel_x, panel_y, panel_x + panel_width - 20, panel_y + panel_height],
            radius=15,
            fill=(25, 25, 35, 220),
            outline=(255, 255, 255, 80),
            width=1
        )

        # Fonte moderna
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        y_offset = panel_y + 15
        box_size = 16

        for cls in sorted(pixel_counts.keys(), key=int):
            color = colors[int(cls)]
            count = pixel_counts[cls]
            name = class_names[int(cls)]

            # Caixinha colorida
            draw_panel.rectangle(
                [panel_x + 15, y_offset,
                 panel_x + 15 + box_size, y_offset + box_size],
                fill=color + (255,)
            )

            # Texto da classe e contagem
            draw_panel.text(
                (panel_x + 15 + box_size + 10, y_offset),
                f"{name}: {count} px",
                fill=(240, 240, 240, 255),
                font=font
            )

            y_offset += box_size + 12

        buffered = io.BytesIO()
        combined_with_panel.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f"data:image/png;base64,{img_str}"
        # === Exportar máscara como imagem de rótulo ===
        # Cria uma imagem em tons de cinza com as classes
        label_mask = Image.fromarray(pred.astype(np.uint8), mode="L")

        # Salvar opcionalmente no disco (comentado)
        # label_mask.save("mask_label.png")

        # Codificar a imagem da máscara para base64
        buffered_mask = io.BytesIO()
        label_mask.save(buffered_mask, format="PNG")
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode()

        return JSONResponse(
            status_code=200,
            content={
                "filename": image.filename,
                "pixel_counts": pixel_counts,
                "image": href,
                "mask": f"data:image/png;base64,{mask_base64}"

            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/unet_freedom_track_video", summary="Track mouse in UNET video",  tags=["UNET"])
async def track_video(video: UploadFile = File(...),  fps: int = Query(5, ge=1, le=60, description="Frames per second to process (default 5, max 60)")):
    try:
        contents = await video.read()
        result = track_mouse_movement_in_video(
            contents, model, device, transform
        )

        # Converte todos os np.ndarray para listas
        serializable_result = convert_ndarrays_to_lists(result)

        return JSONResponse(status_code=200, content=serializable_result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# UNET - imagem


@app.post("/unet_housing_track_image", summary="Track mouse in UNET image", tags=["UNET"])
async def unet_track_image(
    image: UploadFile = File(..., description="Video file")
   
):
    """
    Processa o vídeo enviado para rastrear o deslocamento dos animais.
    Permite definir quantos FPS serão processados para reduzir custo computacional.
    """
    try:
        contents = await image.read()
        result = track_mouse_movement_in_video(
            contents, model, device, transform, fps=fps
        )
        serializable_result = convert_ndarrays_to_lists(result)
        return JSONResponse(status_code=200, content=serializable_result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/unet_housing_track_video", summary="Track mouse in UNET video", tags=["UNET"])
async def unet_track_video(video: UploadFile = File(..., description="Video file")):
    try:
        contents = await video.read()
        result = track_mouse_movement_in_video(
            contents, model, device, transform)
        serializable_result = convert_ndarrays_to_lists(result)
        return JSONResponse(status_code=200, content=serializable_result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# DETECTRON - imagem


@app.post("/detectron_track_image", summary="Track mouse in Detectron image", tags=["DETECTRON"])
async def detectron_track_image(image: UploadFile = File(..., description="Image file")):
    try:
        contents = await image.read()
        result = track_mouse_movement_in_video(
            contents, model, device, transform)
        serializable_result = convert_ndarrays_to_lists(result)
        return JSONResponse(status_code=200, content=serializable_result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# DETECTRON - vídeo


@app.post("/detectron_track_video", summary="Track mouse in Detectron video", tags=["DETECTRON"])
async def detectron_track_video(video: UploadFile = File(..., description="Video file")):
    try:
        contents = await video.read()
        result = track_mouse_movement_in_video(
            contents, model, device, transform)
        serializable_result = convert_ndarrays_to_lists(result)
        return JSONResponse(status_code=200, content=serializable_result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
