# Strong Vivarium: Thermal Imaging Analysis Software Wistar *Rattus norvegicus* – Semantic and Instance Segmentation Software and Dataset
![Github version](https://img.shields.io/badge/version-0.0.4-blue)
![License](https://img.shields.io/badge/license-GNU-green)
![Status](https://img.shields.io/badge/status-development-yellow)


This dataset offers a **comprehensive collection of thermal images of *Rattus norvegicus* Wistar rats**,generated automatic  **segmentation masks** that were subsequently verified. It is specifically curated to support **deep learning tasks** such as **semantic segmentation** and **instance segmentation** in the context of biomedical image analysis **technical benchmarks**


**Image 1 -** Images that make up the database 

| 1x1| NxN |
| --- | --- |
| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fa33739c57711589dde37332acf4443b6%2Fframe_00090.jpg?generation=1748203874976823&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fe5428007ac6621ad734964147f00e54d%2FCaptura%20de%20tela%202025-05-24%20173325.png?generation=1748203895388385&alt=media) |


---

# ~~*Ethical Considerations / Animal Ethics Statement*~~
The animal observed in this study is a domestic pet owned by the author. There was no involvement of third-party animals or facilities. All procedures were purely observational and non-invasive, with no physical interaction of stimulation or alteration of the animal's natural behavior.

The animal had complete freedom of movement and no form of restraint, conditioning or experimental manipulation was applied at any stage. Filming was carried out in the animal's usual environment, ensuring minimal stress and maintaining natural conditions.

The author states that the animal's welfare was respected throughout the process, in accordance with commonly accepted ethical standards for non-invasive studies involving companion animals. As this study did not involve invasive procedures or external subjects, institutional ethics approval was not required.

## (c) **Open Licenses and Precedents**

This project aligns with widely accepted precedents in the field of computer vision and animal segmentation, which utilize publicly available datasets of domestic animals **without requiring animal ethics committee (CEUA/IACUC) approval**, due to their non-invasive nature and open data policies. Examples include:

* **Stanford Dogs Dataset**
  [http://vision.stanford.edu/aditya86/ImageNetDogs/](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  → Images of 120 breeds of dogs collected from ImageNet, used for classification and detection tasks.
* **Oxford-IIIT Pet Dataset**
  [https://www.robots.ox.ac.uk/\~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
  → Contains 37 categories of pet breeds (cats and dogs), with annotations for segmentation and classification.
* **Kaggle PetFinder Dataset**
  [https://www.kaggle.com/competitions/petfinder-adoption-prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction)
  → Used for predictive modeling of pet adoption; all data are collected from public pet adoption listings.

These datasets are considered **non-invasive, observational resources**. They demonstrate that **image-based studies involving domestic animals—without physical contact, experimental manipulation, or stress—are ethically acceptable and widely supported by the machine learning research community**. This project follows the same principle: the data involves a **single domestic pet** observed passively through thermal imaging in a home environment, with owner consent and no behavioral or physiological intervention.

> ![IMPORTANT](https://img.shields.io/badge/important-observations-blue) 
> <h2>⚠️ Important Ethical and Scientific Disclaimer  </h2>
> 1. We reiterate that the animal involved in this dataset is a domestic pet. No invasive procedures or confinement protocols were applied.
> 2. **If this benchmark is used in scientific or academic research, it is highly recommended to repeat the validation tests on real data**, since the environment and behavioral context of a domestic environment are not directly comparable to those of a controlled vivarium.
> 3. Caution should be exercised when generalizing or comparing results with studies that used laboratory-raised animals under standardized housing conditions.

----

# 📦 Dataset Highlights

* 🔬 Biomedical image analysis
* 🎯 Object detection and segmentation
* 🧠 Artificial intelligence applied to thermal data
* 🧪 Behavioral and physiological monitoring

This dataset contains over **1,200 thermal image frames, along with approximately 692 manually selected initial thermal images and their corresponding segmentation masks. These masks are subsequently used to segment and revise another 14,400 thermal frames**, which can be randomly combined to generate a virtually unlimited number of training samples. This structure makes the dataset particularly suitable for training deep learning models, including U-Net, Detectron2, and their respective variants. Additionally, the repository provides comprehensive resources, including scripts for model training, manual mask refinement, and deployment, as well as pre-trained U-Net and Detectron2 models for immediate application in research and development.

---

# 📁 Directory Structure

```
build/                                 → Generated Windows build
src/                                   → Development
    models/                               → Inference
        model_detectron
            output_masks_panoptic/             → Panoptic masks generated by Detectron2  
            output_images_panoptic/            → Corresponding images for panoptic segmentation
            panoptic_annotations.json          → Panoptic-style annotations used by Detectron2  
            metrics_multiclass_detectron.json  → Evaluation metrics for the Detectron2 panoptic model    
        model_unet_freedon
            thermal_images/                    → Raw thermal image frames background painel neutral
            masks_skeleton_thermal/            → Segmentation no IA masks (generated and annotated) 
            masks_thermal_images/              → Verified and corrected segmentation masks  
            metrics_multiclass_unet.json       → Evaluation metrics (per epoch) for the U-Net model  
            unet_checkpoint_epoch_{N}.pth      → Saved U-Net model checkpoints (one per epoch)
            model_unet_freedon.pth             → Trained model freedon 90% + mIoU
        model_unet_housing
            thermal_images_end/                → Raw thermal image frames housing modified 
            masks_thermal_images_unet/         → U-Net predicted segmentation   
            masks_thermal_images_end/          → UNet predicted and validated segmentation masks 
    functions/                            → Assistance
        opencv.py                              → Extract individual frames from thermal video sequences  
        pseud-class.py                         → Generating pseudo-classes for weak supervision 
        view_mask_debug.py                     → Visualization segmentation results over images
        edit_mask_app.py                       → GUI tool for manual refinement of segmentation masks   
        loop_api_unet.py                       → Inference loop using U-Net through API calls  
        transforms_fake_multiple.py            → Handling masks with multiple overlapping animals  
    install/                              → Installers Linux/macOS no GUI 
    services/                             → Offer
        api_unet_freedon.py                    → REST API to serve U-Net predictions  
    train/                                → Model training
        train_unet.py                          → Training script for U-Net segmentation  
        train_detectron.py                     → Training script for Detectron2 panoptic segmentation  
bioterio_forte.exe                     → Windows executable
bioterio_forte.spec                    → Helper windows executable
Dockerfile                             → Container compilation
icon.ico                               → Image start software
install.py                             → Installers with GUI
Readme.md                              → This file
requeriments.txt                       → Required libraries
```


---

# 👨‍🔬 Author

**Ronnei Borges Peres**
- 📧 [founder@thotiacorp.com.br](mailto:founder@thotiacorp.com.br)
- 🔗 [Kaggle](https://www.kaggle.com/ronneiborges) | [GitHub](https://github.com/THOTIACORP)

---

# 📜 License

This dataset is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
- ✅ Free for non-commercial use with attribution
- ❌ Commercial use is **not allowed** without prior permission

---

# 📢 Citation

```markdown
Peres RB. Thermal Images of Rattus norvegicus Wistar for Segmentation [dataset][strong_vivarium]. Kaggle; 2025. Available from: https://www.kaggle.com/ronneiborges
```


---

## 📑 Table of Contents
* [Strong Vivarium: Thermal Imaging Analysis Software Wistar *Rattus norvegicus* – Semantic and Instance Segmentation Software and Dataset](#strong-vivarium-thermal-imaging-analysis-software-wistar-rattus-norvegicus--semantic-and-instance-segmentation-software-and-dataset)
  * [Ethical Considerations / Animal Ethics Statement](#ethical-considerations--animal-ethics-statement)
    * [(c) **Open Licenses and Precedents**](#c-open-licenses-and-precedents)
    * [⚠️ **Important Ethical and Scientific Disclaimer**](#ethical-considerations--animal-ethics-statement)
  * [📦 Dataset Highlights](#-dataset-highlights) 
  * [📁 Directory Structure](#-directory-structure) 
  * [👨‍🔬 Author](#-author)
  * [📜 License](#-license)
  * [📢 Citation](#-citation)
* [1 - Installation and use](#1---installation-and-use)
  * [1.1 - 🎟️ `install.py`](#11---️-installpy-cross-platform-graphical-installer-and-launcher-using-tkinter-with-multi-language-support-enpt)
  * [1.2 - 🐁 `bioterio_tool_panel.py`](#12----bioterio_tool_panelpy--the-project-features-two-key-interfaces)
* [2 - 🎥 Data Source  & Generation](#2----data-source---generation)
* [3 - 🗺️ Justification, Methods and materials](#3---️-justification-methods-and-materials)
* [4 - Software construction](#-4---software-construction)
  * [4.1 - 🎞️ opencv.py – Video Frame Extraction Pipeline](#41---️-opencvpy--video-frame-extraction-pipeline)
  * [4.2 - 🔧 pseudo-class.py – Thermal Image Segmentation Pipeline](#42----pseudo-classpy--ai-free-thermal-image-segmentation-pipeline)
  * [4.3 - 🧪 `view_mask_debug.py` – **Mask Visualization and Debugging**](#43----view_mask_debugpy--mask-visualization-and-debugging)
  * [4.4 - 🧰 `edit_mask_app.py` – **Mask Editing GUI**](#7----edit_mask_apppy--mask-editing-gui)
  * [4.5 - `transforms_fake.py` – **Fake data for intensive training**](#8---transforms_fakepy--fake-data-for-intensive-training)
  * [4.6 - 🧠 `train_unet.py` – **U-Net Model Training**](#9----train_unetpy--u-net-model-training)
  * [4.7 - 🔌 `api_unet.py` – Inference API](#10----api_unetpy--inference-api)
  * [4.8 -  🧪 `transforms_fake_multiple.py` - Fake augmentation of semantic segmentation dataset to train instance segmentation](#12----transforms_fake_multiplepy---fake-augmentation-of-semantic-segmentation-dataset-to-train-instance-segmentation)
* [5 - 🌐 General Proposition: Why Segment *Rattus norvegicus*?](#5----general-proposition-why-segment-rattus-norvegicus)
  * [5.1 -📍 Tracking Movement: Why it Also Matters](#-tracking-movement-why-it-also-matters)
  * [5.2🎯 How the Computer Sees – Why is It Important?](#-how-the-computer-sees---why-is-it-important)
    * [5.2.1 - 🧩 What is a Kernel?](#-what-is-a-kernel)
    * [5.2.2 - 🧩 Kernel / CNN](#-kernel--cnn)
    * [5.2.3 - 📊 Comparison Table: Image AI Models](#-comparison-table-image-ai-models)
* [6 - ⚛️ Results](#6---️-results)
* [7 - ⚛️ Partial Conclusion](#7---️-partial-conclusion)
* [8 - ⚛️ Discussions](#8---️-discussions)
* [9 - 📚 References](#9----references)

---

# 1 - Installation and use
- This installer works on **Windows, Linux, and macOS**, with a localized UI and clear feedback.
Choose your installation method based on your operating system and environment:

### 🪟 **Windows (with GUI)**

```text
Just double-click the file:
bioterio_forte.exe
```


### 🐧💻 **Linux/macOS with GUI**

```bash
python3 install.py
```
or
```bash
python install.py
```

> ![IMPORTANT](https://img.shields.io/badge/important-observations-blue)
> * 📌 *Note: Linux/macOS must have Python with Tkinter support and required GUI packages installed.*
> * **Minimalist distributions** (e.g., Alpine Linux) or **headless environments** do **not support `tkinter` by default**.
> * In **environments without a GUI**, `tkinter` **will not open** and will raise an error like:
>   * *`_tkinter.TclError: no display name and no $DISPLAY environment variable`*
> * If your system does not appear to support a graphical user interface (GUI) or you are unsure, proceed using the GUI-less installation method as it will guide you through the installation


**Image 2 -**   🖥️ Software GUI – Overview Graphical User Interface
| Directory | Install Running |
| --- | --- |
| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fb47610ae06f1706529ef82dd88f8c598%2FCaptura%20de%20tela%202025-06-05%20154152.png?generation=1749152592594434&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F6f0134c6bb935dc9d5172e51e0763531%2FCaptura%20de%20tela%202025-06-05%20153017.png?generation=1749152194031618&alt=media) ||
|`bioterio_forte.exe`| First run|

<br>


* When already installed, the software launches a graphical interface (Tkinter) on:
  * ✅ **Windows**
  * ✅ **Linux**
  * ✅ **macOS**
  
**Image 3 -**   🖥️ Software GUI – Operation
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F8b0113a46ce1e75f381d93a5d128c968%2FCaptura%20de%20tela%202025-06-10%20223055.png?generation=1749609077503300&alt=media)

### 🐧🔧 **Linux/macOS (No GUI - Terminal, No Docker)**

```bash
cd src/install
chmod +x install_nodocker.sh
./install_nodocker.sh
```

### 🐳 **Linux/macOS  (No GUI - Terminal, with Docker)**

```bash
cd src/install
chmod +x install_docker.sh
./install_docker.sh
```

---

## &emsp; 1.1 - 🎟️ `install.py`: Cross-platform graphical installer and launcher using Tkinter with multi-language support (EN/PT).
- It lets the user choose Docker or no-Docker install mode, executes shell/batch scripts, shows real-time logs, displays license info, and checks for prior installation.

| Step                   | Description                                                          |
| ---------------------- | -------------------------------------------------------------------- |
| **OS Detection**       | Detects Windows or Linux/macOS to run correct scripts                |
| **Language Support**   | Shows UI text in English or Portuguese automatically                 |
| **Installation Check** | Skips install if already done (`installed.flag` exists)              |
| **GUI**                | Simple window with install buttons, log area, exit, and website link |
| **Run Scripts**        | Runs `.bat` on Windows, `.sh` on Linux/macOS                         |
| **Log Output**         | Shows real-time script output in scrollable text box                 |
| **Success/Error**      | Displays message boxes based on install result                       |
| **Website Link**       | Opens company website in browser on click                            |

<br>

### &emsp;&emsp; 1.1.1 - 🐁 `bioterio_forte.exe`: Windows Executable
- File with several functions but the summary is that it starts the software and checks if it has already been installed if there are new versions
- Created by compiling the parent file `install.py` any visual changes `install.py` must be recompiled with below command

```bash
pyinstaller --onefile --noconsole --name bioterio_forte --icon=icon.ico install.py
```
### &emsp;&emsp; 1.1.2 - 🐁 `bioterio_forte.spec`: Assistant `bioterio_forte.exe`
- Created by compiling the parent file `install.py`

> 📁 Control of files in the root in src/install

---

## &emsp; 1.2 - 🐁 `bioterio_tool_panel.py` : The project features two key interfaces

This PyQt5-based interface serves as a modular control panel for running a thermal image segmentation pipeline. It organizes six processing scripts—ranging from frame extraction to U-Net inference—into individual tabs with interactive execution and output logs. Each module operates independently, allowing flexibility in debugging and testing. The GUI supports light/dark themes and adapts to Windows or Linux environments. This structure mirrors typical steps used in scientific pipelines, like those from Kaggle's synthetic rodent dataset, providing clarity and control across the entire workflow.

### 📊 **Module Summary Table**

| **Icon** | **Module Name**     | **Function**                                     |
| -------- | ------------------- | ------------------------------------------------ |
| 🎞️      | Extract Frames      | Extracts frames from thermal video input.        |
| 🧬       | Pseudo-Classes      | Generates synthetic labels (pseudo-classes).     |
| 🕵️      | View Mask Debug     | Visualizes segmentation masks for QA.            |
| 🎨       | Edit Masks (GUI)    | GUI tool to manually edit segmentation.          |
| 🧠       | Inference API U-Net | Runs U-Net model inference on thermal data.      |
| 🦊       | Combine Animals     | Merges segmented animals for dataset generation. |

> 📁 Control src/functions

---

# 2 - 🎥 Data Source  & Generation

Step 1 Freedon:
Thermal images were extracted from the video file 20250524165426.mp4, included in the dataset. From the file `opencv.py`. Each frame was processed using the script `pseudo-class.py`, which implements an automated pipeline to segment and classify anatomical regions of the rat in each thermal frame: (background, head, body, tail). We then created a code to observe the masks and another to edit them check `edit_mask_app.py`. Then, we trained `train_unet.py`  and processed part of the images to verify the veracity of the scenario `metrics_multiclass_{N}.json`  scenario and create an initial semantic segmentation model; initial_model = `unet_checkpoint_epoch_9.pht` = `unet_checkpoint_epoch_initial.pth`

Step 2 Housing: Thermal images were extracted from the video file video_end.mp4, included in the dataset. From the `opencv.py` file. Each frame was processed using the `unet_checkpoint_epoch_initial.py` script, which implements an automated pipeline to segment and classify the anatomical regions of the rat in each thermal frame: (bottom, head, body, tail). We then created a code to observe the masks and another to edit them, checking `edit_mask_app.py`. We then trained `train_unet.py` and processed part of the images to verify the veracity of the `metrics_multiclass_{N}.json` scenario and created a final highly reliable semantic segmentation model

---

# 3 - 🗺️ Justification, Methods and materials

Despite being a domestic animal, food and health care follow the SOP Standard Operating Protocols for raising animals. Our domestic animals receive the strictest care with their food, housing and health. The big and main difference is that our animals are not euthanized "dead" at the end of the research. They are domestic animals and we have several animals and breeds and their lives always follow the natural cycle. 

We are involved in this technical benchmark because the potential of the product developed in its complete set can achieve a reduction of up to 3/4 of animals in research. Annually, an estimated **111 million rodents** (mice and rats) are utilized in U.S. biomedical research, comprising over 99% of all laboratory mammals . In Brazil, data from 2021 indicate that approximately **4.07 million animals** were authorized for research purposes, with nearly half being rodents. The study suggests that implementing automated, combined thermal and other imaging systems could potentially reduce the number of rodents used in research by up to **75%**. Applying this reduction:

* **United States**: A 75% decrease from 111 million equates to approximately **83 million fewer rodents** used annually.
* **Brazil**: Assuming half of the 4.07 million animals are rodents (approximately 2 million), a 75% reduction would result in **1.5 million fewer rodents** used each year.

These figures underscore the significant potential of such technologies to enhance ethical standards and reduce animal usage in scientific research.

**Image 2 -** Images of materials
| Sensor | Background | Animal | Housing | Housing modified |
| --- | --- | --- | --- | --- |
| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F1fcc63594ff7fa163c6a37c57916ee91%2Ftransferir%20(4).jpg?generation=1748208105746169&alt=media)<br> | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fd7b648cc7150c4ec4eb45297e13f0271%2Fimages.jpg?generation=1748208276983742&alt=media)<br> | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fe4712bc0a063dfee4167b8a9c2904dc4%2Ftransferir%20(5).jpg?generation=1748208345717146&alt=media)<br> |![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F5516c12b9593362714cda3389aaf1f0e%2FD_NQ_NP_2X_895055-MLB52281836968_112022-F-caixa-bioterio-para-ratos-e-camundongos-com-bebedouro-n3.webp?generation=1748736344606205&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F3c481a3a6eb730668855d85feb13aaec%2FTAMPA%20MORADIA%20-%20THOT%20IA%20CORP.png?generation=1748737380887401&alt=media) |
|Xinfrared  T2S+ 8mm Macro Thermal Imaging Camera| *Backside of a presentation panel* placed on the floor for a neutral thermal background| The Wistar lineage (albino rat) is one of the most widely used in scientific research worldwide [https://bioteriocentral.ufsc.br/rattus-norvergicus/](url)| Conventional housing: Bioterium Box for Rats and Mice with Drinker No. 3| Housing cover developed for remote thermographic monitoring of pets|

<br>

**Image 3 -** Animal care and housing
| Meat | Drink | Forage|
| --- | --- | --- |
| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fd798fbe3cd9313d81d184ed4408efd19%2FD_NQ_NP_819956-MLB54967502488_042023-O-labina-raco-para-roedores-1kkg-presense-isca-de-pesca.webp?generation=1748782967667702&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fb6247e41c9dff94ed4fcf1f19fa29827%2Ftorneira-e-fluxo-de-agua-no-banheiro_51195-23.jpg?generation=1748783206597389&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F7010cbc9fe535d5ece819c48f3dabbde%2FD_NQ_NP_852277-MLU74293215779_012024-O.webp?generation=1748783267445388&alt=media) |
|Food at will|Water as desired|Fragrance changed once a week single animal |


## &emsp; 3.1 - 🎥 Video Capture Setup

Step 1:
The video was recorded in a controlled environment set to a stable temperature of **25°C**. A **presentation panel** was placed **flat on the floor** to act as a neutral thermal background. The **Wistar rat (Rattus norvegicus)** was then positioned on top of the panel. To avoid external heat interference, **all lights were turned off** before recording began.

- **Capture Time**: 16:00  
- **Duration**: 00:00:51 (51 seconds)  
- **Frame Rate**: 25.25 frames per second (FPS)
- **Usage**: Facilitates the initial segmentation of the animal as the housing boxes have some reflection

Step 2: **Sampling Methodology for 24-Hour Animal Behavior Video Analysis**. Objective - To obtain a scientifically valid sample of animal behavior over a 24-hour period by reducing data volume and processing time, while ensuring representative coverage of different behavioral states throughout the day. **Sampling Approach: Stratified Time Sampling Design**- This methodology uses a **stratified temporal sampling** design, commonly applied in ethology and behavioral ecology. The central idea is to divide the 24-hour day into homogeneous periods (strata) based on known biological and environmental rhythms that may influence behavior.

We divide the day into four 6-hour time blocks:

| Period        | Time Range    | Sample Collection Times  |
| ------------- | ------------- | ------------------------ |
| **Night**     | 00:00 – 06:00 | 00:30–01:00, 03:00–03:30 |
| **Morning**   | 06:00 – 12:00 | 07:00–07:30, 09:30–10:00 |
| **Afternoon** | 12:00 – 18:00 | 13:00–13:30, 15:30–16:00 |
| **Evening**   | 18:00 – 24:00 | 19:00–19:30, 22:00–22:30 |

* Each time period contributes **two 30-minute samples**, totaling:
  * **8 video blocks**
  * **4 hours of observation**
  * **Balanced representation** of behavioral variation across circadian phases

#####  **Frame Extraction Plan**

* **Frame rate**: 1 frame per second (fps)
* **Each 30-minute segment** = 30 × 60 = **1,800 frames**
* **8 segments** = 8 × 1,800 = **14,400 frames total**

##### ✅ Total Frame Count

```bash
1 fps × 240 minutes × 60 seconds = 14,400 frames
```

- **Usage**: Highly reliable semantic segmentation with conventional housing environment study approach


> ### ⚠️ **Justification for Sampling Design Housing** – Why This Is a Sampling Strategy
> 
> This approach uses **temporal sampling** — we are **not analyzing the full 24 hours**, but a **representative 4-hour subset** of time. This sample is:
> 
> - **Systematic**: spans distinct periods of the day  
> - **Stratified**: aligned with biological activity cycles  
> - **Efficient**: reduces data volume by ~83% (4h out of 24h)
> 
> 📌 **Conclusion**: This is a **probabilistic time sample**, based on the assumption of **behavioral consistency within strata** and **variation across strata**.
> 
> ---
> 
> #### ⏱️ Why 4 Hours Matters
> 
> Behavioral studies (Altmann, 1974; Martin & Bateson, 2007) show that **4–6 hours** of strategically distributed observation can reliably reflect full-day patterns — **if properly stratified**.
> 
> Our 4-hour sample:
> 
> - Represents **all behavioral phases** (nocturnal, diurnal, crepuscular)  
> - Captures **cyclical behavior** with two observations per 6-hour block  
> - Ensures a **feasible and scalable** protocol for long-term studies

---

# 📀 4 - Software construction: Understand, use, improve

A modular software architecture was implemented to support the segmentation of thermal images in preclinical research workflows. The system is designed around a GUI-based control panel that orchestrates individual pipeline stages, including frame extraction, synthetic label generation, manual mask editing, and inference using convolutional neural networks (e.g., U-Net). Each processing unit operates independently, enabling flexible integration, debugging, and testing. Compatibility across platforms (Windows/Linux) is ensured through dynamic script execution, while the interface offers user-selectable themes and real-time logging to improve user interaction and traceability. This architecture aligns with established methodologies for biomedical image analysis, particularly in studies involving thermal imaging of laboratory rodents.

---

# &emsp; 4.1 - 🎞️ opencv.py – Video Frame Extraction Pipeline

This Python script uses OpenCV to extract all frames from a video file and save them as individual `.jpg` images. It automatically creates an output folder (`thermal_images`), processes the video frame-by-frame, and saves each frame with a sequential filename. Ideal for preparing datasets for computer vision tasks such as object detection or behavioral analysis.

**Image 4 -** Running module independently
| Step 1  |  Step 2 |
| --- | --- |
|  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fd3b035db676428a728202496cda5446c%2FCaptura%20de%20tela%202025-06-05%20104755.png?generation=1749134893128358&alt=media)| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F99ee4d0dddd7b16ea8967304bedd10ce%2FCaptura%20de%20tela%202025-06-05%20104923.png?generation=1749134994187220&alt=media)  |

<br>

**Note on Interference**: 

1. **Step 1 - Loose Environment**: An environment where the animal is freely moving and the background absorbs little of the animal's heat, resulting in minimal interference
2. **Step 2 - Living Environment**: In this case, heat reflection and absorption caused by the animal itself, as well as the presence of feces and urine, introduce significant interference. This makes it difficult to accurately identify and outline the animal’s edges in thermal images 

**Image 5 -** Running GUI
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fa192f95729bd2ba995a2aa2c5ab83ad4%2FCaptura%20de%20tela%202025-06-09%20132922.png?generation=1749490179589018&alt=media)

> 📁 Control src/functions
  
---

# &emsp; 4.2 - 🔧 pseudo-class.py – AI-free Thermal Image Segmentation Pipeline

The AI-less segmentation pipeline was designed to ensure anatomical accuracy while maintaining automation. We ran all videos through this script to generate initial masks, then checked for the easiest masks to correct, then trained each video set back and segmented them with the initial AI and retrained to create the final models. Below are the main steps involved:

### &emsp;&emsp; 4.2.1 - **Background Removal**

Each frame is converted to HSV color space, and a binary mask is generated by thresholding out the thermal background (typically in blue hues). The rat is isolated using inverse masking.

### &emsp;&emsp; 4.2.2 - **Noise Filtering**

Using `cv2.connectedComponentsWithStats`, only the largest connected component is retained—assumed to be the animal—effectively removing thermal noise and small artifacts.

### &emsp;&emsp; 4.2.3 - **Anatomical Axis Detection (PCA)**

Principal Component Analysis (PCA) is applied to the shape of the detected animal to determine its **longitudinal axis**. The two ends of this axis are used to infer the **tail** and **head** positions.

### &emsp;&emsp; 4.2.4 - **Region Classification**

Based on the pixel projections along the identified body axis, each rat is segmented into **three distinct regions**:

* `1` → Head
* `2` → Body
* `3` → Tail
* `0` → Background

### &emsp;&emsp; 4.2.5 - **Mask Generation**

The output is a semantic mask for each image, stored in the `masks_skeleton_thermal_{N}/` directory with filenames matching the original frames.

**Image 6 -** Running module independently
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F78dc1e3e8811d001ded9c860e6ec9c8f%2FCaptura%20de%20tela%202025-06-09%20154955.png?generation=1749498619645861&alt=media)
> - We use SAM (Segment Anything Model) as an auxiliary tool for universal segmentation, allowing the rapid and accurate identification of objects for later manipulation and recomposition in the library. This way we can duplicate the datasets with different corrections by the same researcher. Note: The image bank will be duplicated with SAM before the image name and the masks corresponding to the image name.

**Image 7 -** Running GUI
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F78509ab698973f2414206bcd3cd988b5%2FCaptura%20de%20tela%202025-06-09%20134748.png?generation=1749491286610868&alt=media)

**Image 8 -** Transforming images into discrete masks
| input | output |
| --- | --- |
| <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F0e822165581a64872e57b6e8abb4d739%2Fframe_00001.jpg?generation=1748194030779413&alt=media"> |<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F10ca94057ffb25bfc0660db27468b1da%2Fframe_00001.png?generation=1748194386327422&alt=media"> |

- Note on image visualization: Since masks are predominantly very dark (almost black), it can be difficult to visually inspect the details directly. To address this issue, we developed the `view_mask_debug.py` script — a mask visualization and debugging tool. This utility enhances the display of segmentation masks, making it easier to verify and debug the results during analysis.

> ⚙️ TODO: How can I help with this part?
> - better adjust the definition of the tail, body and head 
> - Improve the calculation of the percentage of the size of the animal's parts 
> - It works for any animal that has a similar structure, for example, cat, dog, fish
> - Make eleptic cuts on head and tail

<br>

> 📁 Control src/functions

---

# &emsp; 4.3 - 🧪 `view_mask_debug.py` – **Mask Visualization and Debugging**

**Purpose:**
Helper script to visualize segmentation masks for debugging.

**Functionality:**
* Overlays colored masks on discrete masks;
* Uses distinct colors to represent the background, head, body and tail regions;
* Facilitates visual inspection of mask quality.

**Usage:**
Use during development to visually evaluate the results of automatic segmentation and/or check the folder target

**Image 9 -** Running module independently - Transforming discreet masks into visible ones
| input | output |
| --- | --- |
| <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F10ca94057ffb25bfc0660db27468b1da%2Fframe_00001.png?generation=1748194386327422&alt=media">  |![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Ffb8e35ab05bedee99606fae98e4c16c1%2FCaptura%20de%20tela%202025-06-09%20161024.png?generation=1749499843341425&alt=media)|

**Image 10 -** Running GUI
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fdf0ddf72661caddbfd42525d569bed26%2FCaptura%20de%20tela%202025-06-10%20100152.png?generation=1749564240367013&alt=media)

> 📁 Control src/functions

---

# &emsp; 4.4 - 🧰 `edit_mask_app.py` – **Mask Editing GUI**

Purpose:
This script provides a graphical user interface (GUI) designed for the manual correction and refinement of segmentation masks generated by the automatic pipeline.

**Image 11 -** Running module independently - Software edit masks
| Software | Team Training |
| --- | --- |
| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fd4b52e5b959aa5132b88d2ad0802fffc%2FCaptura%20de%20tela%202025-05-27%20214221.png?generation=1748396563161100&alt=media) |![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F3f72a2e83546e4a0bfd8ca3e94bf1540%2Fsilhueta-de-rato.png?generation=1748543810392780&alt=media)  <br><br> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fbd8372ff7186cdc4a6b6c4ffa8c7ce98%2FHouse%20Mouse%20Anatomy.jpg?generation=1748549644579053&alt=media)|
|The masks automatically generated by skeleton help but contain many errors. It is only a support paper for a single researcher to be able to work with a larger number of images | Working with the images, we realized that to create a field that is more faithful to the researchers and the AI, it is necessary to increase the brush field to the maximum, lock the background and leave the head and tail as if they were helmets. See above how to refine the shell and below the anatomical connotation of the animal |


**Image 12 -**  Running GUI - Software edit masks
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Ffd195e0b6495c34c846ca425a44eeb86%2FCaptura%20de%20tela%202025-06-10%20145024.png?generation=1749581516041454&alt=media)

Functionality:

- Presents an intuitive and easy-to-use interface where users can visually inspect the masks overlaid on the thermal images
- Allows users to interactively paint or erase specific regions on the mask using mouse input, enabling fine-tuning of mask boundaries and correction of segmentation errors
- Supports working with the multiclass masks, so different anatomical regions (head, body, tail) can be edited accurately
- Improves the overall quality of the training dataset by allowing human-in-the-loop correction, which is crucial for supervised learning tasks where precise annotations impact model performance


Functions:
- Bidirectional Class Swap - Implemented a feature that allows users to swap two classes in the mask simultaneously. For example, swapping class 1 with class 2 will convert all pixels of class 1 to class 2, and all pixels of class 2 to class 1 in one operation.
- Interactive Multiclass Mask - Editing Enhanced the interface to support editing of multiple anatomical regions (head, body, tail) with distinct class assignments, improving precision in manual corrections.
- Brush Size Customization - Added a brush size slider that lets users adjust the painting tool's size dynamically, allowing finer or broader mask edits according to the user's needs.
- Background Locking Option - Introduced a toggle to "lock" the background class during painting, enabling users to restrict painting only to foreground regions and avoid accidental changes to the background.
- Undo Functionality - Added a history stack that allows undoing the last drawing operations, enhancing user control and reducing errors during mask editing.


Usage:

- Run the script when you want to manually review and correct any segmentation masks that may contain artifacts, misclassifications, or inaccuracies resulting from the automatic segmentation process
- Especially useful for preparing high-quality ground truth masks before training deep learning models like U-Net
- Helps ensure that the dataset’s annotations reflect true anatomical regions, leading to more robust and reliable model training and evaluation

> 📁 Control src/functions

---

# &emsp; 4.5 - `transforms_fake.py` – **Fake data for intensive training**

This Python tool generates new synthetic images of rats over background images using PyQt5 and OpenCV. It removes the original rat from thermal images, finds background patches to fill the gap, and pastes rotated rat crops into new positions. Each variation includes a new image and corresponding updated mask.
<br>

**Image 12 -**  Running GUI
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fdbf7d3ecce2d0576e4192ebdf93d45f6%2FCaptura%20de%20tela%202025-06-10%20220812.png?generation=1749607715756485&alt=media)



### 📊 **Functionality Table**

| Feature                 | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| GUI (PyQt5)             | Allows user to set number of backgrounds and rats per background. |
| Input folders           | `thermal_images_end` (images), `masks_thermal_images_end` (masks) |
| Output folders          | `ratos/fundos_sem_rato`, `ratos/novos_ratos`, `ratos/mascaras`    |
| Remove rat from image   | Identifies rat mask, replaces area with random background patch   |
| Mask processing         | Converts labels 1/2/3 to binary mask for rat area                 |
| Rotation                | Random angle between -180° and 180° for each rat variation        |
| Pasting rat             | Places rotated rat at a random location where it fits             |
| Generates updated masks | Original classes (1/2/3) retained in synthetic masks              |
| Logging                 | Real-time progress shown in GUI text area                         |

> 📁 Control src/functions

---


# &emsp; 4.6 - 🧠 `train_unet.py` – **U-Net Model Training**

**Purpose:**
Script to train a semantic segmentation model using the **U-Net architecture** to prepare data for instance segmentation

**Functionality:**

* Loads thermal images and segmentation masks;
* Splits data into training and validation sets;
* Trains a U-Net model using PyTorch;
* Saves the final trained model as `unet_checkpoint_epoch_{N}.pth`.

**Usage:**
Use this script for training, fine-tuning, or experimenting with the dataset.


| Step | Level             | Description                                                                                                                                                                  | Kernels?                                                          |
| ---- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1️⃣  | 🧱 Base           | **Import Modules**<br>Import libraries: `os`, `json`, `torch`, `PIL`, `sklearn`, etc.                                                                                        | ❌                                                                 |
| 2️⃣  | 📁 Dataset        | **Prepare Dataset**<br>- List image (`.jpg`) and mask (`.png`) files<br>- Filter valid pairs                                                                                 | ❌                                                                 |
| 2.1  | └──📦             | **Define `ThermalMouseDataset` Class**&lt;br&gt;- Load and resize images using `torchvision.transforms`&lt;br&gt;- Resize masks using `Image.NEAREST`<br>- Return image and mask tensors | ❌                                                                 |
| 3️⃣  | 🧠 Model          | **Build UNet Model**                                                                                                                                                         | ✅                                                                 |
| 3.1  | └──🧩             | **CBR Function**<br>Helper block: Conv2d → BatchNorm → ReLU                                                                                                                  | 🧠🧊 **YES**<br>`Conv2d` layers define the **kernels**            |
| 3.2  | └──🏗️            | **UNet Architecture**<br>- 4x Downsampling blocks<br>- Bottleneck<br>- 4x Upsampling blocks<br>- Final Conv2d to 4 output channels                                           | 🧠🧊 **YES**<br>Each block uses many **Conv2d layers** (kernels!) |
| 3.3  | └──✂️             | **Crop Center Function**<br>Crop feature maps to match sizes in decoder path                                                                                                 | ❌                                                                 |
| 4️⃣  | 📐 Metrics        | **Evaluation Metrics**                                                                                                                                                       | ❌                                                                 |
| 4.1  | └──📊             | **`evaluate_metrics` Function**<br>- Compute IoU, F1-score, accuracy per class<br>- Mean IoU, mean accuracy, global pixel accuracy                                           | ❌                                                                 |
| 5️⃣  | 🔄 Train/Validate | **`train_or_validate` Function**<br>- Switch between training or evaluation<br>- Forward pass, loss, and metrics<br>- Backpropagation if training                            | ✅ (usa kernels do modelo)                                         |
| 6️⃣  | 🚀 Main Pipeline  | **`main()` Function**                                                                                                                                                        | ✅                                                                 |
| 6.1  | └──⚙️             | **Setup**<br>- Define device (CPU/GPU)<br>- Set paths and transforms                                                                                                         | ❌                                                                 |
| 6.2  | └──📂             | **Create Dataset**<br>- Split into 80% train / 20% validation<br>- Wrap with `DataLoader`                                                                                    | ❌                                                                 |
| 6.3  | └──🔧             | **Initialize Model**<br>- Instantiate UNet<br>- Define optimizer (`Adam`) and loss (`CrossEntropyLoss`)                                                                      | ✅ (modelo inicializa os kernels)                                  |
| 6.4  | └──📈             | **Training Loop**<br>- For each epoch:<br>&nbsp;&nbsp;&nbsp;&nbsp;• Train<br>&nbsp;&nbsp;&nbsp;&nbsp;• Validate<br>&nbsp;&nbsp;&nbsp;&nbsp;• Save weights and metrics                                                                    | ✅ (usa e treina os kernels)                                       |
| 6.5  | └──💾             | **Save Metrics**<br>- Save results to `metrics_multiclass.json`                                                                                                              | ❌                                                                 |
| 7️⃣  | ▶️ Run            | **Execute Pipeline**<br>`if __name__ == "__main__": main()`                                                                                                                  | ✅                                                                 |

1. All initial models can be reviewed on the images with the model itself and rechecked to arrive at a highly reliable hit percentage
<br>
model_freedon = unet_checkpoint_epoch_9.pth :
- Train : 692 - 138 = 554 imagens
- Test : 138 imagens


| Métrica        | Época 9      | Época 10     |
| -------------- | ------------ | ------------ |
| `val_iou_mean` | **0.9041** ✅ | 0.9024       |
| `val_f1_mean`  | **0.9492** ✅ | 0.9482       |
| `val_loss`     | 0.0963       | **0.0825** ✅ |
| `val_accuracy` | 0.9970       | 0.9969       |

<br><br>
model_hounsing:
| Métrica        | Época       | Época     |
| -------------- | ------------ | ------------ |
| `val_iou_mean` | **0.** ✅ | 0.       |
| `val_f1_mean`  | **0.** ✅ | 0.       |
| `val_loss`     | 0.      | **0.** ✅ |
| `val_accuracy` | 0.| 0.      |

>### ⚠️ **Attention: note images are converted to grayscale**
>
>While the database derives multiple color models from images with their respective masks, we convert the thermal images to grayscale to remove artificial color information, align with sensor outputs, reduce noise, and improve model accuracy and efficiency. Segmentation is one thing, thermal analysis is another; here, we segment thermal images regardless of the sensor or thermal sensor configuration.

### 🔄 **Thermal Image Preprocessing: Grayscale Conversion**

| **Aspect**                  | **Before (RGB Input)**                           | **After (Grayscale Input)**                          | **Reason**                                                      |
| --------------------------- | ------------------------------------------------ | ---------------------------------------------------- | --------------------------------------------------------------- |
| **Image Format**            | RGB (possibly false-color or redundant channels) | Grayscale (single-channel “L” mode)                  | Removes unnecessary or misleading color data                    |
| **Sensor Alignment**        | May not match raw thermal sensor output          | Matches typical thermal sensor data (intensity only) | Ensures consistency with the physical nature of thermal imagery |
| **Model Input Consistency** | Risk of irrelevant color features                | Focus only on thermal intensity                      | Prevents confusion in model feature extraction                  |
| **Performance**             | 3-channel input, more computation                | 1-channel input, less memory & faster                | Efficient preprocessing and inference                           |

<br>

> 📁 Control src/train

---

# &emsp; 4.7 - 🔌 `api_{}.py` – Inference API


**Purpose:**
API interface to perform inference with the trained model.

**Functionality:**
* Loads the trained models;
* Accepts input images and videos through API calls; 
* Response:

```markdown
input: image
return JSONResponse(
  status_code=200,
  content={
    "filename": image.filename,
    "pixel_counts": pixel_counts,
    "masked_image": href,
    "mask_unet": f"data:image/png;base64,{mask_base64}"
  }
)

```

```markdown
input: video
return {
  "positions": positions_per_class,
  "movimentos": displaces_totais,
  "mapa_completo": map_img,
  "head_path": head_path,
  "body_path": body_path,
  "tail_path": tail_path,
}
```

**Usage:**
Integration into web or backend systems for automated analysis of new thermal images

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fba2e8d4a9750229cca5d7de29a2253e9%2Foiii.png?generation=1749570558585264&alt=media)
> 📁 Control src/services

---

# &emsp; 4.8 - ➰ `loop_api_unet.py` → Script for generating and saving segmentation masks via API

This script automates the process of sending a batch of thermal images to a U-Net model served through a REST API, receiving the predicted masks in base64 format, decoding them, and saving them locally. It ensures efficient batch processing without manual uploads.

### 🧩 Key Features:

* **API Integration**: Sends each image file to an API endpoint (`/predict/`) using a POST request with `multipart/form-data`.
* **Base64 Decoding**: The predicted mask is expected to be returned as a base64-encoded PNG image. The script decodes and saves it in the `masks_thermal_images_unet/` directory.
* **Error Handling**: Skips files if the response is malformed or if the mask is not returned properly.
* **File Type Filtering**: Only processes valid image formats such as `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`.

#### 🧪 Usage:
Semantic segmentation of all frames again `masks_thermal_images_unet`.
Make sure your API is up and running locally at the given endpoint before running the script.

---

# &emsp; 4.9 - 🧪 `transforms_fake_multiple.py` - Fake augmentation of semantic segmentation dataset to train instance segmentation

### 🎯 **Goal**:

Create synthetic images that simulate multiple rats (2 to 4) per image using existing thermal images and segmentation masks of single rats.
The final images and masks will preserve semantic segmentation labels (classes `1`, `2`, `3`, `4` for rats and `0` for background).


|        Type        | Input | Output |
| --- | --- | --- |
| image | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fd5b246244ad1aeb072d7d71630b696f4%2FCaptura%20de%20tela%202025-05-24%20173325.png?generation=1748620992450426&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Ff05cdad05ad0780aadb46b3ac89c2730%2FCaptura%20de%20tela%202025-05-30%20120354.png?generation=1748621065071118&alt=media) |
| masks | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F3923f2d058fd8132c4172d79ba8beb3f%2Fscreencapture-localhost-8501-2025-05-28-00_13_07.png?generation=1748621136753106&alt=media)| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Ff2991a8fcc70ccb0d078b678abe8ddb0%2Fscreencapture-localhost-8501-2025-05-30-12_06_07.png?generation=1748621190190081&alt=media)|

<br><br>
| Step | Description                                 | Details / Main Function                                                                                                                                                                                   |
| ---- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Import libraries                            | Imports required modules: `os`, `cv2` (OpenCV), `numpy`, `random`, `glob`, `json`, `tifffile`.                                                                                                            |
| 2    | Setup directories                           | Defines directories for original images, multiclass masks, and output folders for panoptic images and masks. Creates output folders if they don't exist.                                                  |
| 3    | Read image and mask files                   | Searches for images with common extensions inside the `thermal_images` folder and masks inside `masks_thermal_images_unet`. Maps base filenames to their paths.                                           |
| 4    | Pair images and masks                       | Creates a list of matching image-mask pairs based on filenames. Ensures at least one pair is found.                                                                                                       |
| 5    | Resize images and masks                     | Resizes all images and masks to a standard size (512x512). Uses appropriate interpolation for masks (`INTER_NEAREST`).                                                                                    |
| 6    | Random transform function                   | Applies random rotation, scaling, and translation to both image and mask for data augmentation.                                                                                                           |
| 7    | Combine multiple images function            | Selects 2 to 4 random image-mask pairs, applies random transforms, combines them into a single image and mask, avoiding overlapping instances. Saves combined files.                                      |
| 8    | Generate composite images                   | Runs the combination process 50 times to create the composite dataset.                                                                                                                                    |
| 9    | Define categories for panoptic segmentation | Sets categories for animal parts (`body`, `head`, `tail`) as "stuff" and for the whole animal (`rat`) as "thing".                                                                                         |
| 10   | Create panoptic mask function               | Takes the multiclass mask and generates: <br>- 2D panoptic mask encoding category and instance IDs. <br>- List of segment info (area, category, ID). <br> Detects rat instances via connected components. |
| 11   | Build panoptic JSON structure               | Creates the JSON structure with dataset info, including images and panoptic annotations.                                                                                                                  |
| 12   | Process combined masks                      | For each combined image, generates the panoptic mask, saves it as 32-bit TIFF, and appends annotation info to the JSON.                                                                                   |
| 13   | Save final JSON file                        | Saves the `panoptic_annotations.json` file containing all generated annotations.                                                                                                                          |
| 14   | Final confirmation message                  | Prints confirmation that the panoptic dataset was successfully generated.                                                                                                                                 |

- **Panoptic segmentation or simply instance segmentation**: The term 'Panopticon' comes from the Greek terms 'pan', which means “all”, juxtaposed with “optical”, linked to “vision”. Thus, Panopticon means 'the all-seeing' 

---


# 5 - 🌐 **General Proposition: Why Segment *Rattus norvegicus*?**

“The genetic proximity between rodents and humans is closer than previously believed: analysis of mouse chromosome 16 suggests a divergence of only ~2.5%” (Science 2002;296(5573):1661–71).

To understand diseases at a deeper level, it's not enough to observe the entire body — we need to analyze specific anatomical regions like the head, body, and tail. That's where semantic segmentation becomes essential.

By breaking down the rat into smaller, meaningful parts, we allow machine learning models to focus on localized thermal or motion-related changes — such as subtle temperature patterns linked to neurological disorders, inflammation, or circulatory issues.

---

## &emsp; 5.1 - 📍 **Tracking Movement: Why it also matters**

Beyond identifying physiological changes, segmentation and frame extraction also make it possible to **track the rat’s position and displacement over time**. By observing how different body parts move across frames, researchers can analyze motor behavior, detect abnormal locomotion, or assess responses to stimuli.

---

## &emsp; 5.2 - 🎯 **How ​​the computer sees - Why is it important?**

A neural network doesn't "see" like a human. It learns through *kernels* — small filters that slide across an image to detect patterns like edges, textures, or temperature gradients.

When these kernels process the entire body at once, vital details may get lost in the noise. But when the image is segmented into head, body, and tail, **each part is analyzed independently**, improving both **spatial focus** and **diagnostic accuracy**.

This approach supports both **biomedical insights** and **behavioral tracking** — a dual benefit that enhances the depth and precision of analysis in preclinical studies.

---

### &emsp;&emsp; 5.2.1 - 🧩 What is a Kernel?

A **kernel** is a tiny matrix (like 3x3 or 5x5) that scans over the image to detect features such as:

* Temperature changes
* Edges and boundaries
* Texture and shape

Think of it like a **magnifying glass** moving over different zones of the rat — trying to understand what’s happening **region by region**.


**Image 11 -** Representation of a kernel with and without limit
| Without Segmentation                   | With Segmentation                            |
| -------------------------------------- | -------------------------------------------- |
|![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F2356eeffb2778516c5807e702414630c%2FChatGPT%20Image%2025%20de%20mai.%20de%202025%2016_50_13.png?generation=1748376056296440&alt=media)| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F2e3e32e000612bed4eec40ee8c4880d9%2FChatGPT%20Image%2025%20de%20mai.%20de%202025%2016_350_13.png?generation=1748376021051645&alt=media)|
| Entire rat is analyzed as one big blob | Head, body, and tail are analyzed separately |
| Subtle patterns might be missed        | Local thermal anomalies become more visible  |
| Kernel has to guess context            | Kernel knows where it's looking              |

<br><br>
🔍 Visual Illustration - *(See the diagram above)*
Left: A kernel slides across the whole rat, confused by overlapping signals.
Right: The rat is segmented into regions. Now the same kernel focuses **only on the head**, or **only on the tail**, leading to **smarter learning** and **disease detection**.

---

### &emsp;&emsp; 5.2.2 - 🧩 Kernel / CNN

A CNN is an encoder, as the name itself suggests: convolution. The etymology of "convolution" comes from the Latin "convolutus, -a, -um", which is the past participle of the verb "convolvo, -ere", which means "to envelop" or "to roll up".


**Image 12 -** kernel enveloping, accounts
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F1126122f45a4bb74fee382f497f41c37%2FnvORWfF.png?generation=1748269536627460&alt=media)
- See more ref.: https://lamfo-unb.github.io/2020/12/05/Captcha-Break/  

---

### &emsp;&emsp; 5.2.3 - 📊 Comparison Table: Image AI Models

| Model Type                                 | Core Idea                                                               | Strengths                                                                            | Limitations                                                          | Typical Use Cases                                                            | Examples                            |
| ------------------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------- |
| **CNN**<br>(Convolutional Neural Network)  | Applies filters (kernels) over image pixels to extract spatial features | - Strong for local patterns<br>- Efficient with small datasets<br>- Well-established | - Limited global context<br>- Performance plateaus on large datasets | - Classification<br>- Object Detection<br>- Segmentation                     | ResNet, VGG, U-Net, EfficientNet    |
| **ViT**<br>(Vision Transformer)            | Splits image into patches and processes as a sequence using attention   | - Captures long-range dependencies<br>- Scales well<br>- Great for large datasets    | - Needs more data to train well<br>- Slower inference                | - Image classification<br>- Fine-grained tasks<br>- Self-supervised learning | ViT, Swin Transformer, DINOv2       |
| **Hybrid (CNN + Transformer)**             | Uses CNNs for local features + Transformer for context                  | - Best of both worlds<br>- High accuracy<br>- Efficient                              | - Can be complex<br>- Requires careful design                        | - Detection<br>- Segmentation<br>- Pose Estimation                           | DETR, Segmenter, BEiT               |
| **GAN / Diffusion**<br>(Generative Models) | Generates new image data by learning distributions                      | - Image generation<br>- Denoising<br>- Inpainting                                    | - Hard to train<br>- Not ideal for analysis tasks                    | - Image synthesis<br>- Super-resolution<br>- Restoration                     | StyleGAN, Stable Diffusion, Pix2Pix |
| **MLP-like**<br>(e.g. MLP-Mixer)           | Replaces convolutions and attention with dense layers                   | - Simplified design<br>- Competitive accuracy                                        | - Less intuitive<br>- Still experimental                             | - Classification<br>- Embedding learning                                     | MLP-Mixer, gMLP                     |

* 🧩 **CNNs** are ideal for ** classification, detection, segmentation, and small datasets.**
* 🔍 **Transformers** shine with **larger datasets and global context.**
* 🎨 **Generative models** focus on **image creation** and editing.
* 🧪 **Hybrids** combine strengths from both CNNs and Transformers.

---

# 6 - ⚛️ **Results**
In computer vision, models typically evolve from classification (what is in the image), to detection (where it is), and finally to segmentation (which pixels belong to each object), offering increasing spatial precision

|Classification|Localization / Detection   | Semantic Segmentation| Instance Segmentation |
| --- | --- | --- | --- |
| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fa2e490f4756f2670213580e11976b043%2Fframe_01021.jpg?generation=1748390297681988&alt=media)| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F142e37b3f7d886a90cb7a14e7c68de69%2Fframe_00024.jpg?generation=1748390491115766&alt=media)| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2F3a905df379c553b3bb03dbfad66979e8%2FCaptura%20de%20tela%202025-05-25%20152337.png?generation=1748391059905086&alt=media)| ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12221778%2Fac4f48fb5a37e18774e0af637ec0e5aa%2F19e5d359f09fb80af69e5e3cc90b6d1195e10aef5ff64f2bc81dd740.jpg?generation=1748612481868126&alt=media)|
|What is in the image?	The model identifies that the image contains a rat. No location or shape information.|Where is the rat?	The model draws a bounding box around the rat. You know its position, but not the exact body shape.|Which pixels belong to the rat (and to which part)? Each pixel is labeled as belonging to the rat's head, body, or tail — allowing for detailed thermal analysis of a single animal, even if there were two in the image it would treat rat as rat but not rat 1 and rat 2|Instance segmentation can label all the above information and tell you rat 1, rat 2, and so on|

<br><br>

| 📌 Type of Image AI       | 🧪 Main Validation Metrics                                                                 |
|---------------------------|--------------------------------------------------------------------------------------------|
| 🏷️ **Classification**         | *Accuracy* <br> *Precision / Recall / F1-score* <br> *Confusion Matrix* <br> *ROC-AUC* <br> *Cross-validation (k-Fold)* |
| 📍 Localization / Detection  | `IoU (Intersection over Union)` &lt;br&gt; `mAP (mean Average Precision)` &lt;br&gt; `Recall@IoU` &lt;br&gt; `FPS (real-time)` |
| 🎯 **Semantic Segmentation**  | *mIoU* <br> *Dice Coefficient* <br> *Pixel Accuracy* <br> *Boundary F1 Score* <br> *Visual overlay inspection* |
| 👤 Instance Segmentation  | `mAP@[.5:.95]` &lt;br&gt; `Per-instance IoU / Dice` &lt;br&gt; `Object Count Accuracy`               |

<br><br>

An instance segmentation AI can leverage all the previous metrics from classification and localization, in addition to its own specific segmentation metrics, depending on the scale and complexity of the task.

Since instance segmentation combines object detection (which involves classification and localization) with pixel-level segmentation, it naturally inherits the need to evaluate:
• Classification accuracy (correctly identifying object classes)
• Localization metrics like mAP (mean Average Precision) based on bounding boxes
• Plus its own specialized segmentation metrics, such as per-instance IoU or Dice coefficient, and boundary quality scores.

The exact set of metrics used depends on the scale of the model’s task—whether it focuses more on coarse detection or fine-grained segmentation—and the evaluation goals. Using this comprehensive set of metrics ensures a thorough assessment of the model’s performance across all relevant aspects: class recognition, object localization, and precise mask quality. 

**The results of this study can be viewed again in the training items `4.6` semantic segmentations [single animal]. In general, all metrics are in `model/metrics/`**. The architecture of this benchmark was built in such a way as to convey knowledge of the step-by-step process of rapidly assembling an AI by a single researcher.

---

# 7 - ⚛️ Partial Conclusion
Up to this point, the project has demonstrated a solid and coherent structure, effectively leveraging the "Thermal Images of Rats (Mice) for Segmentation" dataset . The utilization of synthetic thermal images has facilitated the development and preliminary validation of image segmentation models in biomedical research. These artificially generated datasets offer a scalable and ethical alternative to real animal data, reducing the need for live animal experimentation and accelerating the training process for deep learning models. The use of synthetic thermal image data, as done here for segmentation, presents an efficient and ethical alternative for the development and preliminary validation of image segmentation models in biomedical research. These artificially generated or pre-processed datasets reduce the need for manual mask verification and provide a scalable solution for training data-intensive deep learning models. However, while synthetic data speeds up development and reduces costs, it should be approached with caution to avoid overfitting models to unrealistic or oversimplified conditions. 

---

# 8 - ⚛️ Discussions
## &emsp; **Synthetic data**
Synthetic (or fake) data has become an increasingly valuable asset in medical imaging and preclinical research, particularly when real data is scarce, expensive, or ethically challenging to obtain. The dataset of thermal images of rats and mice serves as a useful benchmark for evaluating segmentation algorithms, offering controlled, annotated samples that can simulate real-world conditions to a certain extent.

Nevertheless, synthetic datasets often lack the full variability and unpredictability found in real clinical or experimental environments. This discrepancy can result in models that perform well during training but fail to generalize in real-world applications. As such, reliance solely on fake data may lead to biased results or inflated performance metrics.

To mitigate these issues, researchers should consider hybrid training strategies—combining synthetic data with a small amount of real data for fine-tuning. Moreover, rigorous validation using real-world samples remains crucial to verify the practical utility of any model trained on synthetic inputs.

In summary, while synthetic data like the one used in this project provides a powerful foundation for model prototyping, it should not replace real data when aiming for deployment in practical or clinical settings. Its greatest value lies in complementing real data, especially during the early phases of research and development.

## &emsp; **Synthetic data frameworks**


---

# 9 - 📚 References
## &emsp; **Animal Care**
- Andrade A, Pinto SC, Oliveira RS, editors. *Laboratory Animals: Breeding and Experimentation* \[Internet]. Rio de Janeiro: Editora FIOCRUZ; 2002 \[cited 2025 Jun 1]. 388 p. Available from: [https://books.scielo.org/id/xspz5](https://books.scielo.org/id/xspz5)

## &emsp; **Amount of rodent research**
- Carbone L. Estimating mouse and rat use in American laboratories by extrapolation from Animal Welfare Act-regulated species. *Sci Rep*. 2021;11:493. doi:10.1038/s41598-020-79961-0. PMID: 33431864; PMCID: PMC7805762.
- Science Magazine. How many mice and rats are used in U.S. labs? Controversial study says more than 100 million. *Science*. 2021 Jan 22. Available from: [https://www.science.org/content/article/how-many-mice-and-rats-are-used-us-labs-controversial-study-says-more-100-million](https://www.science.org/content/article/how-many-mice-and-rats-are-used-us-labs-controversial-study-says-more-100-million)
- Pesquisa FAPESP. Regulation establishes guidelines on the use of animals in research and teaching in Brazil. 2022 May. Available from: [https://revistapesquisa.fapesp.br/en/regulation-establishes-guidelines-on-the-use-of-animals-in-research-and-teaching-in-brazil](https://revistapesquisa.fapesp.br/en/regulation-establishes-guidelines-on-the-use-of-animals-in-research-and-teaching-in-brazil)

## &emsp; **Sampling Methods**
- Altmann J. Observational study of behavior: Sampling methods. *Behaviour*. 1974;49(3–4):227–66. doi:10.1163/156853974X00534
- Martin P, Bateson P. *Measuring behaviour: An introductory guide*. 3rd ed. Cambridge: Cambridge University Press; 2007. Available from: [https://www.cambridge.org/core/books/measuring-behaviour/889CBF92A1E5A4DC143BCC7D4DBA0D8C](https://www.cambridge.org/core/books/measuring-behaviour/889CBF92A1E5A4DC143BCC7D4DBA0D8C)
- Mural RJ, Adams MD, Myers EW, Smith HO, Gabor‑Miklos GL, Wides R, et al. A comparison of whole‑genome shotgun‑derived mouse chromosome 16 and the human genome. Science. 2002 May 31;296(5573):1661–71.

## &emsp; **Thermal Imaging / Infrared**
- Pereira CB, Kunczik J, Zieglowski L, Tolba R, Abdelrahman A, Zechner D, et al. Remote welfare monitoring of rodents using thermal imaging. *Sensors (Basel)*. 2018;18(11):3653. doi:10.3390/s18113653. PMID: 30373282; PMCID: PMC6263688
- Travain T, Valsecchi P. Infrared thermography in the study of animals' emotional responses: A critical review. *Animals (Basel)*. 2021;11(9):2510. doi:10.3390/ani11092510. PMID: 34573476; PMCID: PMC8464846.
- Mazur‑Milecka M, Kocejko T, Rumiński J. Deep Instance Segmentation of Laboratory Animals in Thermal Images. Applied Sciences. 2020;10(17):5979.
- vipulvs91. Neural Networks and Deep Learning Course Project: Thermal‑Image‑segmentation. GitHub; 2025. Available from: https://github.com/vipulvs91/Thermal-Image-segmentation
- Zheng S, Zhou C, Jiang X, Huang J, Xu D. Progress on infrared imaging technology in animal production: A review. *Sensors (Basel)*. 2022;22(3):705. doi:10.3390/s22030705. PMID: 35161450; PMCID: PMC8839879.
- Weimer SL, Wideman RF, Scanes CG, Mauromoustakos A, Christensen KD, Vizzier-Thaxton Y. Broiler stress responses to light intensity, flooring type, and leg weakness as assessed by heterophil-to-lymphocyte ratios, serum corticosterone, infrared thermography, and latency to lie. *Poultry Science*. 2020;99(7):3301–11. doi:10.1016/j.psj.2020.03.028. PMID: 32616223; PMCID: PMC7597826.
- Tabh JKR, Burness G, Wearing OH, Tattersall GJ, Mastromonaco GF. Infrared thermography as a technique to measure physiological stress in birds: Body region and image angle matter. *Physiological Reports*. 2021;9(11)\:e14865. doi:10.14814/phy2.14865. PMID: 34057300; PMCID: PMC8165734.

##  &emsp; **Other similar databases**
- RumpuDoggo. *mouse recog 2 Dataset* \[Internet]. Roboflow Universe: Roboflow; 2024 May \[cited 2025 Jun 7]. Available from: [https://universe.roboflow.com/rumpudoggo/mouse-recog-2](https://universe.roboflow.com/rumpudoggo/mouse-recog-2)
- dcwno. *mouse\_detection Dataset* \[Internet]. Roboflow Universe: Roboflow; 2024 May \[cited 2025 Jun 7]. Available from: [https://universe.roboflow.com/dcwno/mouse\_detection-em87f](https://universe.roboflow.com/dcwno/mouse_detection-em87f)
- MEL. *Mouse Dataset* \[Internet]. Roboflow Universe: Roboflow; 2024 May \[cited 2025 Jun 7]. Available from: [https://universe.roboflow.com/mel-mefb3/mouse-jmvzh](https://universe.roboflow.com/mel-mefb3/mouse-jmvzh)
- Arthur. *mouse\_detection Dataset* \[Internet]. Roboflow Universe: Roboflow; 2024 May \[cited 2025 Jun 7]. Available from: [https://universe.roboflow.com/arthur-mq9ig/mouse\_detection-7e034](https://universe.roboflow.com/arthur-mq9ig/mouse_detection-7e034)
- project. *Thermal\_mouse Dataset* \[Internet]. Roboflow Universe: Roboflow; 2024 May \[cited 2025 Jun 7]. Available from: [https://universe.roboflow.com/project-7lf4j/thermal\_mouse](https://universe.roboflow.com/project-7lf4j/thermal_mouse)
- rats. *rats Dataset* \[Internet]. Roboflow Universe: Roboflow; 2024 May \[cited 2025 Jun 7]. Available from: [https://universe.roboflow.com/rats-jjzwt/rats-naa55](https://universe.roboflow.com/rats-jjzwt/rats-naa55)


## &emsp; **Data Augmentation**
- Ghiasi G, Cui Y, Srinivas A, Qian R, Lin TY, Cubuk ED, et al. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. *arXiv \[Preprint]*. 2020 Dec 14 \[cited 2025 Jun 7]; arXiv:2012.07177. Available from: [https://arxiv.org/abs/2012.07177](https://arxiv.org/abs/2012.07177)


## &emsp; **Auxiliary tools**
- Kirillov A, Mintun E, Ravi N, Mao H, Rolland A, Gustafson L, et al. Segment Anything. arXiv preprint arXiv:2304.02643 [Internet]. 2023 [cited 2025 Jun 7]. Available from: https://arxiv.org/abs/2304.02643

