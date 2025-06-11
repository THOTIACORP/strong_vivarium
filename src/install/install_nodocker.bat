@echo off
setlocal

echo Iniciando instalação sem Docker...
cd /d %~dp0

:: Verifica se o Python está instalado
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python não encontrado. Baixando e instalando Python...

    :: Baixa o instalador do Python (ajustável para versão desejada)
    curl -o python-installer.exe https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe

    if exist python-installer.exe (
        echo Instalando Python em modo silencioso...
        python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

        :: Aguarda um pouco para garantir a instalação
        timeout /t 10 >nul

        :: Testa novamente se o Python foi instalado
        where python >nul 2>&1
        IF %ERRORLEVEL% NEQ 0 (
            echo ERRO: Python ainda não foi encontrado após a instalação.
            pause
            exit /b 1
        ) else (
            echo Python instalado com sucesso!
        )
    ) else (
        echo ERRO: Falha ao baixar o instalador do Python.
        pause
        exit /b 1
    )
) else (
    echo Python já está instalado.
)
echo Diretório atual:
cd
echo Listando arquivos no diretório atual:
dir
echo Listando arquivos em ..\..
dir ..\..
:: Instala dependências do projeto
echo Instalando dependências com pip...
python -m pip install --upgrade pip
pip install -r ..\..\requirements.txt

cd ..\
:: Verifica se a pasta "models" existe, senão cria
if not exist models (
    echo Criando pasta models...
    mkdir models
)

cd models
:: Baixar arquivos .pth se não existirem
if not exist sam_vit_h_4b8939.pth (
    echo Baixando sam_vit_h_4b8939.pth...
    curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
)

if not exist sam_vit_l_0b3195.pth (
    echo Baixando sam_vit_l_0b3195.pth...
    curl -L -o sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
)

if not exist sam_vit_b_01ec64.pth (
    echo Baixando sam_vit_b_01ec64.pth...
    curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
)


timeout /t 5 >nul
cd ..\
python functions/bioterio_tool_panel.py || (
    echo ERRO ao executar o script Python.
    pause
    exit /b 1
)

echo.
echo Instalação finalizada!
echo "Instalação binarizada"

pause
exit /b 0
endlocal
