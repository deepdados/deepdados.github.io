---
layout: post
title: Modelo 2 - Tutorial 3 - Mapeamento de ativação de classe (CAM)
subtitle: Mapeamento de ativação de classe (CAM) - Modelo 2
tags: [COVID]
---

**ATUALIZANDO**


**Objetivo principal do projeto:** automatizar o processo de detecção de casos de COVID-19 a partir de imagens de radiografia de tórax, utilizando redes neurais convolucionais (RNC) por meio de técnicas de aprendizado profundo (deep learning). O projeto completo pode ser acessado [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/projetoCovid_Cesar_Lucas-Versao2.pdf)

**Etapas para alcançar o objetivo:**<br />
1- Mapeamento de ativação de classe (CAM)

**Mapeamento de ativação de classe (CAM)**

*Bases de dados utilizadas:*<br />
- Imagens de raio X e tomografias computadorizadas de tórax de indivíduos infectados com COVID-19 (COHE; MORRISON; DAO, 2020): [link](https://github.com/ieee8023/covid-chestxray-dataset)<br />
- Imagens de pulmões de indivíduos sem nenhuma infecção (KERMANY; ZHANG; GOLDBAUM, 2018): [link](https://data.mendeley.com/datasets/rscbjbr9sj/2)<br />

*Pacotes utilizados:*<br />
- Gradcam<br />
- PIL<br />
- Tensorflow<br />
- Numpy<br />
- Argparse<br />
- Imutils<br />
- Cv2<br />

*Código utilizado no projeto:*<br />
O notebook com todos os códigos utilizados nesta etapa está disponível [aqui](https://)<br />
**Observação:** a numeração e título de cada passo descrito neste tutorial, corresponde com a numeração e título contidos no notebook.

*Passos que serão seguidos:*<br />
**1º Passo** – [Importar as bibliotecas que serão utilizadas](#importar-as-bibliotecas-que-serão-utilizadas)<br />
**2º Passo** – [Carregar os arrays construídos na etapa referente ao pré-processamento de dados e normalizar os dados do input](#carregar-os-arrays-construídos-na-etapa-referente-ao-pré-processamento-de-dados-e-normalizar-os-dados-do-input)<br />
**3º Passo** – [Dividir os dados em dados de treinamento e dados de teste](#dividir-os-dados-em-dados-de-treinamento-e-dados-de-teste)<br />

**Tutorial 3:**

**1º Passo** 
#### Importar as bibliotecas que serão utilizadas

Importamos as bibliotecas Gradcam, PIL, Tensorflow, Numpy, Argparse, Imutils, Cv2, visto que nos apoiaremos nestas para realizar o mapeamento de ativação de classe (CAM) do [Modelo 2] (https://)

**Explicação biblioteca GradCam e o que é o mapeamento; continuação após o Modelo 2 (atualizando)**.

``` python
#importar pacote

from gradcam import GradCAM
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
from google.colab.patches import cv2_imshow
```

**Observação:** algumas bibliotecas não foram importadas completamente, como, por exemplo, o Tensorflow, pois não utilizaremos todas as funções ali contidas. Desta forma, facilita a utilização da biblioteca e o processamento dos códigos/dados.<br />

**2º Passo**
#### Carregar os arrays construídos na etapa referente ao pré-processamento de dados e dividir os dados em dados de treinamento e dados de teste na [Etapa 2](https://) do modelo 2. É importante executá-lo novamente, pois vamos precisar da imagem não normalizada no 

Este passo é similar ao [Passo 2](https://) e [Passo 3](https://) do [

Os arrays “X_Train” e “Y_Train” construídos na [Etapa 1](https://deepdados.github.io/2020-04-14-Modelo-1-COVID19-Pr%C3%A9-Processamento-dos-Dados/) do modelo 2 foram carregados e associados, respectivamente, às variáveis “X_train” e “Y_train”. Além disso, a variável X_train foi normalizada para os valores oscilarem entre 0 e 1.

``` python
X_train = np.load("/content/drive/My Drive/Python/COVID/Arrays/Modelo2/X_Train.npy")
X_train = X_train/255
Y_train = np.load("/content/drive/My Drive/Python/COVID/Arrays/Modelo2/Y_Train.npy")
```

**3º Passo**
#### Dividir os dados em dados de treinamento e dados de teste

20% dos dados referentes às imagens foram separados para o teste do modelo. A função abaixo retorna quatro valores que foram associados a quatro variáveis, a saber: “X_train”, “X_test”, “Y_train” e “Y_test”. Respectivamente, as duas primeiras foram usadas para o treino do modelo e as duas últimas para o teste.

É possível observar abaixo que a quantidade de casos é a mesma para o dataset referente ao treinamento (n = 117) e, também, para o teste (n = 30).

``` python
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train, test_size = 0.2, random_state = 40)

print(f"X_train shape: {X_train.shape} Y_train shape {Y_train.shape}")
print(f"X_test shape: {X_test.shape} Y_test shape {Y_test.shape}")
```

**Observação:** o parâmetro “random_state” faz com que a seleção aleatória de imagens seja a mesma toda vez que a função for executada.<br />


**Conclusão sobre do mapeamento de ativação de classe (CAM):** <br />
<br />
<br />
**Observação:** os resultados não apresentam caráter clínico, mas sim exploratório. Contudo, com o aperfeiçoamento dos modelos, estes podem trazer benefícios para o enfrentamento à COVID-19.
<br />
<br />
**Bibliografia** <br />
COHEN, Joseph; MORRISON, Paul; DAO, Lan. COVID-19 Image Data Collection. arXiv:2003.11597, 2020.<br />
<br />
KERMANY, Daniel; ZHANG, Kang; GOLDBAUM, Michael. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v.2, 2018. Disponível em: http://dx.doi.org/10.17632/rscbjbr9sj.2
