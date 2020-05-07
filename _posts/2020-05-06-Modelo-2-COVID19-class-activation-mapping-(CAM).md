---
layout: post
title: Modelo 2 - Tutorial 3 - Mapeamento de ativação de classe (CAM)
subtitle: Mapeamento de ativação de classe (CAM) - Modelo 2
tags: [COVID]
---

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
O notebook com todos os códigos utilizados nesta etapa está disponível [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/cam_modelo2_tutorial3.ipynb)<br />
**Observação:** a numeração e título de cada passo descrito neste tutorial, corresponde com a numeração e título contidos no notebook.

*Passos que serão seguidos:*<br />
**1º Passo** – [Importar as bibliotecas que serão utilizadas](#importar-as-bibliotecas-que-serão-utilizadas)<br />
**2º Passo** – [Carregar os arrays construídos na etapa referente ao pré-processamento de dados do Modelo 2 e dividir os dados em dados de treinamento e dados de teste](#carregar-os-arrays-construídos-na-etapa-referente-ao-pré-processamento-de-dados-do-modelo-2-e-dividir-os-dados-em-dados-de-treinamento-e-dados-de-teste)<br />
**3º Passo** – [Construir e salvar o mapa de ativação de classe (CAM)](#construir-e-salvar-o-mapa-de-ativação-de-classe-cam)<br />

**Tutorial 3:**

**1º Passo** 
#### Importar as bibliotecas que serão utilizadas

Importamos as bibliotecas Gradcam, PIL, Tensorflow, Numpy, Argparse, Imutils, Cv2, visto que nos apoiaremos nestas para realizar o mapeamento de ativação de classe (CAM) do [Modelo 2](https://deepdados.github.io/2020-04-21-Modelo-2-COVID19-Treinamento-e-Resultados/). <br />
<br />

Para a construção do mapeamento de ativação de classe adaptamos o código disponibilizado pelo [Pyimagesearch](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/) para o nosso [Modelo 2](https://deepdados.github.io/2020-04-21-Modelo-2-COVID19-Treinamento-e-Resultados/).<br />
<br />

Para rodar a biblioteca gradcam, é preciso carregar esta no ambiente virtual que você está executando o seu código. A biblioteca está disponível para download [aqui](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/).<br />

O mapeamento de ativação de classe facilita a interpretabilidade do modelo ao expor os locais da imagem que o modelo utilizou para realizar a classificação.<br />

O código aqui exposto para a construção do mapeamento deve ser executado após o treino do modelo. Neste tutorial, executamos apóst o treino do [Modelo 2](https://deepdados.github.io/2020-04-21-Modelo-2-COVID19-Treinamento-e-Resultados/).


``` python
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
#### Carregar os arrays construídos na etapa referente ao pré-processamento de dados do Modelo 2 e dividir os dados em dados de treinamento e dados de teste

Este passo é similar ao Passo 2 e Passo 3 do [Modelo 2](https://deepdados.github.io/2020-04-21-Modelo-2-COVID19-Treinamento-e-Resultados/) É importante executá-lo novamente, pois vamos precisar das imagens não normalizadas no mapeamento de ativação de classe (CAM).<br />

Os arrays “X_Train” e “Y_Train” construídos na [Etapa 1](https://deepdados.github.io/2020-04-20-Modelo-2-COVID19-Pr%C3%A9-Processamento-dos-Dados/) do Modelo 2 foram carregados e associados, respectivamente, às variáveis “X_train” e “Y_train”.<br />

20% dos dados referentes às imagens foram separados para o teste do modelo. A função abaixo retorna quatro valores que foram associados a quatro variáveis, a saber: “X_train”, “X_test”, “Y_train” e “Y_test”.

``` python
X_train = np.load("/content/drive/My Drive/Python/COVID/Arrays/Modelo2/X_Train.npy")
Y_train = np.load("/content/drive/My Drive/Python/COVID/Arrays/Modelo2/Y_Train.npy")
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train, test_size = 0.2, random_state = 40)
```

**3º Passo**
#### Construir e salvar o mapa de ativação de classe (CAM)

O código abaixo salva as imagens que foram utilizadas para o teste do [Modelo 2](https://deepdados.github.io/2020-04-21-Modelo-2-COVID19-Treinamento-e-Resultados/), cria o mapa de ativação de classe, sobrepõe este às imagens de pulmão utilizadas no teste e salva estas figuras.


``` python
for image in range(X_test.shape[0]):
  img = Image.fromarray(X_test[image])
  img.save(f"imagens_{image}.png")


for array in range(X_test.shape[0]):
  orig = cv2.imread(f"imagens_{array}.png")
  orig = cv2.resize(orig, (237, 237))
  image = load_img((f"imagens_{array}.png"), target_size=(237, 237))
  image = img_to_array(image)
  image = image/255
  image = np.expand_dims(image, axis=0)
  preds = model.predict(image)
  i = np.argmax(preds[0])
  if i == 0:
    label = f"COVID-19: {int(preds[0][i] * 100)}%"
  elif i == 1:
    label = f"NORMAL: {int(preds[0][i] * 100)}%"
  else:
    label = f"O.INF. : {int(preds[0][i] * 100)}%"
  print("[INFO] {}".format(label))
  cam = GradCAM(model, i)


  heatmap = cam.compute_heatmap(image)
  heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
  (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
  cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
  cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
  # #output = np.vstack([orig, heatmap, output])
  output = np.vstack([output])

  print(orig.shape,heatmap.shape,output.shape)

  # #output = imutils.resize(output, height=700)
  print(output.shape)
  cv2_imshow(output)
  cv2.imwrite(f"/content/drive/My Drive/Python/COVID/Arrays/Modelo2/image_{array}.png",output)
  # #cv2.waitKey(0)
```

![](/img/cam.png)

**Observação:** A partir dos arquivos salvos foi construída esta grade de imagens com alguns exemplos.<br />

**Conclusão sobre do mapeamento de ativação de classe (CAM):** É possível observar em cada grupo de imagem (Normal, COVID e outras infecções (O. Inf.)) as regiões mais importantes para o modelo para detectar cada classe. Quanto mais "quente" (amarelo), maior o grau de importância.<br />
<br />
<br />
**Observação:** os resultados não apresentam caráter clínico, mas sim exploratório. Contudo, com o aperfeiçoamento dos modelos, estes podem trazer benefícios para o enfrentamento à COVID-19.
<br />
<br />
**Bibliografia** <br />
COHEN, Joseph; MORRISON, Paul; DAO, Lan. COVID-19 Image Data Collection. arXiv:2003.11597, 2020.<br />
<br />
KERMANY, Daniel; ZHANG, Kang; GOLDBAUM, Michael. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v.2, 2018. Disponível em: http://dx.doi.org/10.17632/rscbjbr9sj.2
