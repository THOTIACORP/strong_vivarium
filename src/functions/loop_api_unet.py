import requests
import base64
import os

def salvar_mascaras_da_pasta(api_url, pasta_imagens):
    extensoes_validas = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    arquivos_processados = []

    pasta_mascara = "./masks_thermal_images_unet"
    os.makedirs(pasta_mascara, exist_ok=True)

    for arquivo in os.listdir(pasta_imagens):
        caminho_completo = os.path.join(pasta_imagens, arquivo)

        if os.path.isfile(caminho_completo) and os.path.splitext(arquivo)[1].lower() in extensoes_validas:
            print(f"Processando: {arquivo}")

            with open(caminho_completo, "rb") as f:
                arquivos = {"image": (arquivo, f, "image/png")}
                resposta = requests.post(api_url, files=arquivos)

            if resposta.status_code == 200:
                dados = resposta.json()
                mask_base64 = dados.get("mask_unet", None)

                if mask_base64 is None or mask_base64 == "data:image/png;base64,...":
                    print(f"API não retornou a máscara base64 real para {arquivo}")
                    continue

                prefixo = "data:image/png;base64,"
                if mask_base64.startswith(prefixo):
                    mask_base64 = mask_base64[len(prefixo):]

                nome_mascara = os.path.splitext(arquivo)[0] + ".png"
                caminho_mascara = os.path.join(pasta_mascara, nome_mascara)

                with open(caminho_mascara, "wb") as f_mask:
                    f_mask.write(base64.b64decode(mask_base64))

                print(f"Mascara salva como: {caminho_mascara}")
                arquivos_processados.append(caminho_mascara)
            else:
                print(f"Erro na API para {arquivo}: {resposta.status_code} - {resposta.text}")

    return arquivos_processados


# Exemplo de uso:
api_url = "http://localhost:8000/predict/"
pasta = "./thermal_images_end"
salvar_mascaras_da_pasta(api_url, pasta)
