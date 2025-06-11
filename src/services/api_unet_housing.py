from PIL import ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import io
import torch.nn as nn
import base64

app = FastAPI(title="U-Net Thermal Mouse Segmentation API")

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
checkpoint = torch.load("unet_checkpoint_epoch_9.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])




@app.post("/predict/")
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
            "RGBA", (combined.width + panel_width, combined.height), (255, 255, 255, 0)
        )
        combined_with_panel.paste(combined, (0, 0))

        draw_panel = ImageDraw.Draw(combined_with_panel)

        panel_x = combined.width + 10
        panel_y = 20
        radius = 20

        # Desenho do painel com cantos arredondados (moderno)
        def draw_rounded_rectangle(draw, xy, radius, fill, outline=None, width=1):
            x1, y1, x2, y2 = xy
            draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

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
                "masked_image": href,
                "mask_unet": f"data:image/png;base64,{mask_base64}"

            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
