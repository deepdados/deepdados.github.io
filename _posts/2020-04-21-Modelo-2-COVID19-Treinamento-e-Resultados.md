---
layout: post
title: Modelo 2 - Tutorial 2 - Detecção automática de casos de COVID-19 a partir de imagens de radiografia de tórax
subtitle: Treinamento do modelo e exposição dos resultados - Modelo 2
tags: [COVID]
---

**Objetivo principal do projeto:** automatizar o processo de detecção de casos de COVID-19 a partir de imagens de radiografia de tórax, utilizando redes neurais convolucionais (RNC) por meio de técnicas de aprendizado profundo (deep learning). O projeto completo pode ser acessado [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/projetoCovid_Cesar_Lucas.pdf)

**Etapas para alcançar o objetivo:**<br />
1- [Pré-processamento dos dados](https://deepdados.github.io/2020-04-20-Modelo-2-COVID19-Pr%C3%A9-Processamento-dos-Dados/)<br /> 
2- Treinamento do modelo e exposição dos resultados

**Etapa 2 - Treinamento do modelo e exposição dos resultados**

*Bases de dados utilizadas:*<br />
- Imagens de raio X e tomografias computadorizadas de tórax de indivíduos infectados com COVID-19 (COHE; MORRISON; DAO, 2020): [link](https://github.com/ieee8023/covid-chestxray-dataset)<br />
- Imagens de pulmões de indivíduos sem nenhuma infecção (KERMANY; ZHANG; GOLDBAUM, 2018): [link](https://data.mendeley.com/datasets/rscbjbr9sj/2)<br />

*Pacotes utilizados:*<br />
- Pandas<br />
- Os <br />
- PIL <br />
- Tensorflow<br />
- Sklearn<br />
- Imutils<br />
- Matplotlib<br />
- Numpy<br />
- Argparse<br />
- Cv2<br />
- Seaborn<br />


*Código utilizado no projeto:*<br />
O notebook com todos os códigos utilizados nesta etapa está disponível [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/treinamento_resultados_COVID_modelo2.ipynb)<br />
**Observação:** a numeração e título de cada passo descrito neste tutorial, corresponde com a numeração e título contidos no notebook.

*Passos que serão seguidos:*<br />
**1º Passo** – [Importar as bibliotecas que serão utilizadas](#importar-as-bibliotecas-que-serão-utilizadas)<br />
**2º Passo** – [Carregar os arrays construídos na etapa referente ao pré-processamento de dados e normalizar os dados do input](#carregar-os-arrays-construídos-na-etapa-referente-ao-pré-processamento-de-dados-e-normalizar-os-dados-do-input)<br />
**3º Passo** – [Dividir os dados em dados de treinamento e dados de teste](#dividir-os-dados-em-dados-de-treinamento-e-dados-de-teste)<br />
**4º Passo** – [Determinando a arquitetura do modelo (Xception) que será treinado](#determinando-a-arquitetura-do-modelo-xception-que-será-treinado)<br />
**5º Passo** – [Determinar os hyperparameters e compilar o modelo (Xception)](#determinar-os-hyperparameters-e-compilar-o-modelo-xception)<br />
**6º Passo** – [Treinar o modelo (Xception)](#treinar-o-modelo-xception)<br />
**7º Passo** – [Observar a acurácia do modelo (Xception) e a função de perda](#observar-a-acurácia-do-modelo-xception-e-a-função-de-perda)<br />
**8º Passo** – [Determinando a arquitetura do modelo (ResNet50V2) que será treinado](#determinando-a-arquitetura-do-modelo-resnet50v2-que-será-treinado)<br />
**9º Passo** – [Determinar os hyperparameters e compilar o modelo (ResNet50V2)](#determinar-os-hyperparameters-e-compilar-o-modelo-resnet50v2)<br />
**10º Passo** - [Treinar o modelo (ResNet50V2)](#treinar-o-modelo-resnet50v2)<br />
**11º Passo** - [Observar a acurácia do modelo (ResNet50V2) e a função de perda](#observar-a-acurácia-do-modelo-resnet50v2-e-a-função-de-perda)<br />
**12º Passo** - [Determinando a arquitetura (VGG16) do modelo que será treinado](#determinando-a-arquitetura-vgg16-do-modelo-que-será-treinado)<br />
**13º Passo** - [Determinar os hyperparameters e compilar o modelo (VGG16)](#determinar-os-hyperparameters-e-compilar-o-modelo-vgg16)<br />
**14º Passo** - [Treinar o modelo (VGG16)](#treinar-o-modelo-vgg16)<br />
**15º Passo** - [Observar a acurácia do modelo (VGG16) e a função de perda](#observar-a-acurácia-do-modelo-vgg16-e-a-função-de-perda)<br />
**16º Passo** - [Observar quais imagens o modelo (VGG16) acertou](#observar-quais-imagens-o-modelo-vgg16-acertou)<br />


**Tutorial 2:**

**1º Passo** 
#### Importar as bibliotecas que serão utilizadas

Importamos as bibliotecas Tensorflow, Sklearn, Imutils, Matplotlib, Numpy, Argparse, Cv2, Os, Pandas e Seaborn, visto que nos apoiaremos nestas para realizar o treinamento do modelo referente à COVID-19 e a análise dos resultados.

``` python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
%matplotlib inline
```

**Observação:** algumas bibliotecas não foram importadas completamente, como, por exemplo, o Tensorflow, pois não utilizaremos todas as funções ali contidas. Desta forma, facilita a utilização da biblioteca e o processamento dos códigos/dados.<br />

**2º Passo**
#### Carregar os arrays construídos na etapa referente ao pré-processamento de dados e normalizar os dados do input

Os arrays “X_Train” e “Y_Train” construídos na [Etapa 1](https://deepdados.github.io/2020-04-14-Modelo-1-COVID19-Pr%C3%A9-Processamento-dos-Dados/) foram carregados e associados, respectivamente, às variáveis “X_train” e “Y_train”. Além disso, a variável X_train foi normalizada para os valores oscilarem entre 0 e 1.

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

X_train shape: (117, 237, 237, 3) Y_train shape (117, 1)
X_test shape: (30, 237, 237, 3) Y_test shape (30, 1)
```

**Observação:** o parâmetro “random_state” faz com que a seleção aleatória de imagens seja a mesma toda vez que a função for executada.<br />

**4º Passo**
#### Determinando a arquitetura do modelo (Xception) que será treinado

Foi carregado os pesos da arquitetura Xception a partir do dataset “imagenet”, desconsiderando o topo da rede. Além disso, foi definido o input com a dimensão das imagens do banco de imagens que utilizaremos, a saber: 237 x 237px e 3 canais de cores como profundidade. Estas informações foram associadas à variável “bModel”.

Além disso, foi determinada a arquitetura do topo da rede, visto que foi retirado o topo da rede do dataset “imagenet”. Esta arquitetura foi associada à variável “tModel”.

Por último, foram unidas as variáveis “bModel” e “tModel” na variável “model”. Esta última variável representa o modelo que será treinado.

```python
bModel = Xception(weights="imagenet", include_top=False,
  	input_tensor=Input(shape=(237, 237, 3)))
tModel = bModel.output
tModel = AveragePooling2D(pool_size=(2, 2))(tModel)
tModel = Flatten(name="flatten")(tModel)
tModel = Dense(20, activation="relu")(tModel)
tModel = Dropout(0.2)(tModel)
tModel = Dense(3, activation="softmax")(tModel)

model = Model(inputs=bModel.input, outputs=tModel)
```

**5º Passo**
#### Determinar os hyperparameters e compilar o modelo (Xception)

Foram determinados os hyperparameters, em específico, o learning rate (“INIT_LR”), as epochs (“EPOCHS”) e o batch size (“BS”).

Posteriormente, foi definida a função de optimização Adam (“opt”), o modelo foi compilado considerando a função de perda “categorical_crossentropy” e como métrica de avaliação dos resultados, considerou-se a acurácia.

```python
INIT_LR = 1e-3
EPOCHS = 80
BS = 15

for layer in bModel.layers:
    layer.trainable = False

opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
```

**6º Passo**
#### Treinar o modelo (Xception)

A partir do comando abaixo o modelo foi treinado, deixando 10% das imagens para a validação. As informações foram salvas na variável “x” e o modelo foi salvo no computador como “modeloc_2.hdf5”.

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001, cooldown=5)

x = model.fit(X_train, Y_train, batch_size=BS,validation_split=0.1, epochs=EPOCHS,callbacks=[reduce_lr])

model.save("/content/drive/My Drive/Python/COVID/model/modeloc_2.hdf5")
```

**7º Passo**
#### Observar a acurácia do modelo (Xception) e a função de perda

Construímos um gráfico para analisar o histórico de acurácia dos dados de treinamento e de validação do modelo. Construímos, também, um gráfico que computa o erro da rede em relação aos dados de treinamento e validação.

Nota-se que a acurácia do modelo foi de 94%. Ou seja, o modelo acertou 94% das imagens utilizadas no teste.

``` python
plt.plot(x.history['accuracy'])
plt.plot(x.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(x.history['loss'])
plt.plot(x.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.evaluate(X_test,Y_test)
```
![](/img/xception_accuracy1_mod2.png)
<br />
<br />
![](/img/xception_accuracy2_mod2.png)
<br />
<br />
``` python
3/3 [==============================] - 1s 396ms/step - loss: 0.2876 - accuracy: 0.9467
[0.287587970495224, 0.9466666579246521]
```

**8º Passo**
#### Determinando a arquitetura do modelo (ResNet50V2) que será treinado

Foi carregado os pesos da arquitetura ResNet50V2 a partir do dataset “imagenet”, desconsiderando o topo da rede. Além disso, foi definido o input com a dimensão das imagens do banco de imagens que utilizaremos, a saber: 237 x 237px e 3 canais de cores como profundidade. Estas informações foram associadas à variável “bModel”.

Além disso, foi determinada a arquitetura do topo da rede, visto que foi retirado o topo da rede do dataset “imagenet”. Esta arquitetura foi associada à variável “tModel”.

Por último, foram unidas as variáveis “bModel” e “tModel” na variável “model”. Esta última variável representa o modelo que será treinado.

```python
bModel = ResNet50V2(weights="imagenet", include_top=False,
  	input_tensor=Input(shape=(237, 237, 3)))
tModel = bModel.output
tModel = AveragePooling2D(pool_size=(2, 2))(tModel)
tModel = Flatten(name="flatten")(tModel)
tModel = Dense(20, activation="relu")(tModel)
tModel = Dropout(0.2)(tModel)
tModel = Dense(3, activation="softmax")(tModel)

model = Model(inputs=bModel.input, outputs=tModel)
```

**9º Passo**
#### Determinar os hyperparameters e compilar o modelo (ResNet50V2)

Foram determinados os hyperparameters, em específico, o learning rate (“INIT_LR”), as epochs (“EPOCHS”) e o batch size (“BS”).

Posteriormente, foi definida a função de optimização Adam (“opt”), o modelo foi compilado considerando a função de perda “categorical_crossentropy” e como métrica de avaliação dos resultados, considerou-se a acurácia.

```python
INIT_LR = 1e-3
EPOCHS = 80
BS = 15

for layer in bModel.layers:
    layer.trainable = False

opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
```

**10º Passo**
#### Treinar o modelo (ResNet50V2)

A partir do comando abaixo o modelo foi treinado, deixando 10% das imagens para a validação. As informações foram salvas na variável “x” e o modelo foi salvo no computador como “modeloc_2.hdf5”.

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001, cooldown=5)

x = model.fit(X_train, Y_train, batch_size=BS,validation_split=0.1, epochs=EPOCHS,callbacks=[reduce_lr])

model.save("/content/drive/My Drive/Python/COVID/model/modeloc_2.hdf5")
```

**11º Passo**
#### Observar a acurácia do modelo (ResNet50V2) e a função de perda

Construímos um gráfico para analisar o histórico de acurácia dos dados de treinamento e de validação do modelo. Construímos, também, um gráfico que computa o erro da rede em relação aos dados de treinamento e validação.

Nota-se que a acurácia do modelo foi de 96%. Ou seja, o modelo acertou 96% das imagens utilizadas no teste.

``` python
plt.plot(x.history['accuracy'])
plt.plot(x.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(x.history['loss'])
plt.plot(x.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.evaluate(X_test,Y_test)
```
![](/img/resnet_accuracy1_mod2.png)
<br />
<br />
![](/img/resnet_accuracy2_mod2.png)
<br />
<br />
``` python
3/3 [==============================] - 1s 346ms/step - loss: 0.3698 - accuracy: 0.9600
[0.3697645366191864, 0.9599999785423279]
```

**12º Passo**
#### Determinando a arquitetura (VGG16) do modelo que será treinado

Foi carregado os pesos da arquitetura VGG16 a partir do dataset “imagenet”, desconsiderando o topo da rede. Além disso, foi definido o input com a dimensão das imagens do banco de imagens que utilizaremos, a saber: 237 x 237px e 3 canais de cores como profundidade. Estas informações foram associadas à variável “bModel”.

Além disso, foi determinada a arquitetura do topo da rede, visto que foi retirado o topo da rede do dataset “imagenet”. Esta arquitetura foi associada à variável “tModel”.

Por último, foram unidas as variáveis “bModel” e “tModel” na variável “model”. Esta última variável representa o modelo que será treinado.

```python
bModel = VGG16(weights="imagenet", include_top=False,classes=3,
	input_tensor=Input(shape=(237, 237, 3)))
  tModel = bModel.output
tModel = AveragePooling2D(pool_size=(2, 2))(tModel)
tModel = Flatten(name="flatten")(tModel)
tModel = Dense(20, activation="relu")(tModel)
tModel = Dropout(0.2)(tModel)
tModel = Dense(3, activation="softmax")(tModel)

model = Model(inputs=bModel.input, outputs=tModel)
```

**13º Passo**
#### Determinar os hyperparameters e compilar o modelo (VGG16)

Foram determinados os hyperparameters, em específico, o learning rate (“INIT_LR”), as epochs (“EPOCHS”) e o batch size (“BS”).

Posteriormente, foi definida a função de optimização Adam (“opt”), o modelo foi compilado considerando a função de perda “categorical_crossentropy” e como métrica de avaliação dos resultados, considerou-se a acurácia.

```python
INIT_LR = 1e-3
EPOCHS = 80
BS = 15

for layer in bModel.layers:
    layer.trainable = False

opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
```

**14º Passo**
#### Treinar o modelo (VGG16)

A partir do comando abaixo o modelo foi treinado, deixando 10% das imagens para a validação. As informações foram salvas na variável “x” e o modelo foi salvo no computador como “modeloc_2.hdf5”.

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001, cooldown=5)

x = model.fit(X_train, Y_train, batch_size=BS,validation_split=0.1, epochs=EPOCHS,callbacks=[reduce_lr])

model.save("/content/drive/My Drive/Python/COVID/model/modeloc_2.hdf5")
```

**15º Passo**
#### Observar a acurácia do modelo (VGG16) e a função de perda

Construímos um gráfico para analisar o histórico de acurácia dos dados de treinamento e de validação do modelo. Construímos, também, um gráfico que computa o erro da rede em relação aos dados de treinamento e validação.

Além disso, nota-se que a acurácia do modelo foi de 97%. Ou seja, o modelo acertou 97% das imagens utilizadas no teste.

``` python
plt.plot(x.history['accuracy'])
plt.plot(x.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(x.history['loss'])
plt.plot(x.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.evaluate(X_test,Y_test)
```
![](/img/vgg_accuracy1_mod2.png)
<br />
<br />
![](/img/vgg_accuracy2_mod2.png)
<br />
<br />
``` python
3/3 [==============================] - 1s 177ms/step - loss: 0.0941 - accuracy: 0.9733
[0.09413935989141464, 0.9733333587646484]
```

**16º Passo**
#### Observar quais imagens o modelo (VGG16) acertou

A partir da imagem abaixo é possível observar as imagens que o modelo acertou. Os “Labels” (Label Predict e Label Correct) que apresentam o mesmo nome indicam que o modelo acertou a predição. Exemplo: Label Predict = COVID e Label Correct = COVID.

Além disso, a figura foi salva como modelo_2.pdf no computador.

``` python
plt.figure(figsize=(20,20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=2.0, hspace=2.0)
i = 0
for i,image in enumerate(X_test):
    plt.subplot(9,9,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    img = np.expand_dims(X_test[i],axis = 0)
    x_pred = model.predict(img)[0]
    pred_covid = x_pred[0]
   
    pred_normal = x_pred[1]

    pred_infeccoes = x_pred[2]
    
    
    if pred_covid > pred_normal and pred_covid > pred_infeccoes:
      label = "COVID"
    elif pred_normal > pred_covid and pred_normal > pred_infeccoes:
      label = "NORMAL"
    elif pred_infeccoes > pred_covid and pred_infeccoes > pred_normal:
      label = "INFECÇÕES"
     
    
    if Y_test[i][0] == 1:
      label_test = "COVID"
    elif Y_test[i][1] == 1:
      label_test = "NORMAL"
    elif Y_test[i][2] == 1:
      label_test = "INFECÇÕES"
    plt.xlabel(f"Label Predict = {label} \n Label Correct = {label_test}")
    i += 1
plt.savefig('/content/drive/My Drive/Python/COVID/model/modelo_2.pdf')
```
![](/img/pulmao_mod2.png)
<br />
<br />

**Conclusão sobre o modelo 2:** A partir dos resultados preliminares é possível notar que o modelo apresenta uma acurácia elevada para classificar os pulmões normais, com COVID-19 e com outras infecções. Especialmente a partir da arquitetura VGG16. O próximo treino testará novas arquiteturas e parâmetros, com o intuito de aperfeiçoar o modelo.<br />
<br />
<br />
**Observação:** os resultados não apresentam caráter clínico, mas sim exploratório. Contudo, com o aperfeiçoamento dos modelos, estes podem trazer benefícios para o enfrentamento à COVID-19.
<br />
<br />
**Bibliografia** <br />
COHEN, Joseph; MORRISON, Paul; DAO, Lan. COVID-19 Image Data Collection. arXiv:2003.11597, 2020.<br />
<br />
KERMANY, Daniel; ZHANG, Kang; GOLDBAUM, Michael. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v.2, 2018. Disponível em: http://dx.doi.org/10.17632/rscbjbr9sj.2
