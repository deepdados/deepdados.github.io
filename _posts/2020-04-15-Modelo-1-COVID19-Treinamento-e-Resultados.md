---
layout: post
title: Modelo 1 - Tutorial 2 - Detecção automática de casos de COVID-19 a partir de imagens de radiografia de tórax
subtitle: Treinamento do modelo e exposição dos resultados - Modelo 1
tags: [COVID]
---

**Objetivo principal do projeto:** automatizar o processo de detecção de casos de COVID-19 a partir de imagens de radiografia de tórax, utilizando redes neurais convolucionais (RNC) por meio de técnicas de aprendizado profundo (deep learning). O projeto completo pode ser acessado [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/projeto_cesar_lucas_COVID.pdf)

**Etapas para alcançar o objetivo:**<br />
1- [Pré-processamento dos dados](https://deepdados.github.io/2020-04-14-Modelo-1-COVID19-Pr%C3%A9-Processamento-dos-Dados/)<br /> 
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
O notebook com todos os códigos utilizados nesta etapa está disponível [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/treinamento_resultados_COVID_modelo1.ipynb)<br />
**Observação:** a numeração e título de cada passo descrito neste tutorial, corresponde com a numeração e título contidos no notebook.

*Passos que serão seguidos:*<br />
**1º Passo** – [Importar as bibliotecas que serão utilizadas](#importar-as-bibliotecas-que-serão-utilizadas)<br />
**2º Passo** – [Carregar os arrays construídos na etapa referente ao pré-processamento de dados e normalizar os dados do input](#carregar-os-arrays-construídos-na-etapa-referente-ao-pré-processamento-de-dados-e-normalizar-os-dados-do-input)<br />
**3º Passo** – [Dividir os dados em dados de treinamento e dados de teste](#dividir-os-dados-em-dados-de-treinamento-e-dados-de-teste)<br />
**4º Passo** – [Determinando a arquitetura do modelo que será treinado](#determinando-a-arquitetura-do-modelo-que-será-treinado)<br />
**5º Passo** – [Determinar os hyperparameters e compilar o modelo](#determinar-os-hyperparameters-e-compilar-o-modelo)<br />
**6º Passo** – [Treinar o modelo](#treinar-o-modelo)<br />
**7º Passo** – [Observar a acurácia do modelo e a função de perda](#observar-a-acurácia-do-modelo-e-a-função-de-perda)<br />
**8º Passo** – [Observar quais imagens o modelo acertou](#observar-quais-imagens-o-modelo-acertou)<br />
**9º Passo** – [Construir uma matriz de confusão](#construir-uma-matriz-de-confusão)<br />

**Tutorial 2:**

**1º Passo** 
#### Importar as bibliotecas que serão utilizadas

Importamos as bibliotecas Tensorflow, Sklearn, Imutils, Matplotlib, Numpy, Argparse, Cv2, Os, Pandas e Seaborn, visto que nos apoiaremos nestas para realizar o treinamento do modelo referente à COVID-19 e a análise dos resultados.

``` python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import pandas as pd
import cv2
import numpy as np
import os
import seaborn as sn
%matplotlib inline
```

**Observação:** algumas bibliotecas não foram importadas completamente, como, por exemplo, o Tensorflow, pois não utilizaremos todas as funções ali contidas. Desta forma, facilita a utilização da biblioteca e o processamento dos códigos/dados.<br />

**2º Passo**
#### Carregar os arrays construídos na etapa referente ao pré-processamento de dados e normalizar os dados do input

Os arrays “X_Train” e “Y_Train” construídos na [Etapa 1](https://deepdados.github.io/2020-04-14-Modelo-1-COVID19-Pr%C3%A9-Processamento-dos-Dados/) foram carregados e associados, respectivamente, às variáveis “X_train” e “Y_train”. Além disso, a variável X_train foi normalizada para os valores oscilarem entre 0 e 1.

``` python
X_train = np.load("/content/drive/My Drive/Python/COVID/Arrays/X_Train.npy")
X_train = X_train/255
Y_train = np.load("/content/drive/My Drive/Python/COVID/Arrays/Y_Train.npy")
```

**3º Passo**
#### Dividir os dados em dados de treinamento e dados de teste

20% dos dados referentes às imagens foram separados para o teste do modelo. A função abaixo retorna quatro valores que foram associados a quatro variáveis, a saber: “X_train”, “X_test”, “Y_train” e “Y_test”.

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
#### Determinando a arquitetura do modelo que será treinado

Foi carregado os pesos da arquitetura VGG16 a partir do dataset “imagenet”, desconsiderando o topo da rede. Além disso, foi definido o input com a dimensão das imagens do banco de imagens que utilizaremos, a saber: 237 x 237px e 3 canais de cores como profundidade. Estas informações foram associadas à variável “bModel”.

Além disso, foi determinada a arquitetura do topo da rede, visto que foi retirado o topo da rede do dataset “imagenet”. Esta arquitetura foi associada à variável “tModel”.

Por último, foram unidas as variáveis “bModel” e “tModel” na variável “model”. Esta última variável representa o modelo que será treinado.

``` python
bModel = VGG16(weights="imagenet", include_top=False,
  input_tensor=Input(shape=(237, 237, 3)))

tModel = bModel.output
tModel = AveragePooling2D(pool_size=(4, 4))(tModel)
tModel = Flatten(name="flatten")(tModel)
tModel = Dense(64, activation="relu")(tModel)
tModel = Dropout(0.2)(tModel)
tModel = Dense(1, activation="sigmoid")(tModel)

model = Model(inputs=bModel.input, outputs=tModel)
```

**5º Passo**
#### Determinar os hyperparameters e compilar o modelo

Foram determinados os hyperparameters, em específico, o learning rate (“INIT_LR”), as epochs (“EPOCHS”) e o batch size (“BS”).

Posteriormente, foi definida a função de optimização Adam (“opt”), o modelo foi compilado considerando a função de perda “binary_crossentropy” e como métrica de avaliação dos resultados, considerou-se a acurácia.

``` python
INIT_LR = 1e-3
EPOCHS = 50
BS = 8

for layer in bModel.layers:
  layer.trainable = False

opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
  metrics=["accuracy"])
```

**6º Passo**
#### Treinar o modelo

A partir do comando abaixo o modelo foi treinado, deixando 10% das imagens para a validação. As informações foram salvas na variável “x” e o modelo foi salvo no computador como “modeloc_1.hdf5”.

``` python
x = model.fit(X_train, Y_train, batch_size=BS,validation_split=0.1, epochs=EPOCHS)

model.save("/content/drive/My Drive/Python/COVID/model/modeloc_1.hdf5")

```

**7º Passo**
#### Observar a acurácia do modelo e a função de perda

Construímos um gráfico para analisar o histórico de acurácia dos dados de treinamento e de validação do modelo. Construímos, também, um gráfico que computa o erro da rede em relação aos dados de treinamento e validação. Estes apontam que, aparentemente, não houve overfitting, visto que as linhas de treino e validação se aproximaram.

Além disso, nota-se que a acurácia do modelo foi de 98%. Ou seja, o modelo acertou 98% das imagens utilizadas no teste.

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
![](/img/acuracia1_modelo1.png)
<br />
<br />
![](/img/acuracia2_modelo1.png)
<br />
<br />
``` python
2/2 [==============================] - 1s 342ms/step - loss: 0.0345 - accuracy: 0.9818
[0.03453369066119194, 0.9818181991577148]
```

**8º Passo**
#### Observar quais imagens o modelo acertou

A partir da imagem abaixo é possível observar as imagens que o modelo acertou. Os “Labels” (Label Predict e Label Correct) que apresentam o mesmo nome indicam que o modelo acertou a predição. Exemplo: Label Predict = COVID e Label Correct = COVID. Nesse sentido, é possível observar que o modelo acertou 54 de 55 imagens totais.

Além disso, a figura foi salva como modelo_1.pdf no computador.

``` python
plt.figure(figsize=(20,20))
i = 0
for i,image in enumerate(X_test):
    plt.subplot(7,9,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    img = np.expand_dims(X_test[i],axis = 0)
    x_pred = model.predict(img)[0]
    x0 = x_pred[0]
    
    if x0 > 0.5:
      label = "COVID"
    else:
      label = "NORMAL"
    
    if Y_test[i] == 1:
      label_test = "COVID"
    else:
      label_test = "NORMAL"
    plt.xlabel(f"Label Predict = {label} \n Label Correct = {label_test}")
    i += 1
plt.savefig('/content/drive/My Drive/Python/COVID/model/modelo_1.pdf')
```
![](/img/pulmao_modelo1.png)
<br />
<br />

**9º Passo**
#### Construir uma matriz de confusão

O código abaixo cria uma matriz de confusão com os dados do modelo.

O modelo errou apenas uma classificação entre as 55 imagens utilizadas para o teste, apresentando uma acurácia de 98%. A matriz de confusão mostra que, dentre o total de imagens, 58% (n = 32) representam verdadeiros positivos, 40% (n = 22) verdadeiros negativos, 1,8% (n = 1) falsos negativos e 0% (n = 0) falsos positivos.

``` python
ypredict = model.predict(X_test)

ypredictc = []

for value in ypredict:
  x0 = value [0]
  # x1 = value [1]
  if x0 > 0.5:
    ypredictc.append(1)

  else:
    ypredictc.append(0)

resultado = np.array(ypredictc)


x = confusion_matrix(y_true=Y_test,y_pred=resultado)
x = x/X_test.shape[0]

y = pd.DataFrame(x,index = ["NORMAL","COVID"],columns=["NORMAL","COVID"])
plt.figure(figsize = (10,7))

fig = sn.heatmap(y, annot=True,cmap="Greens").get_figure()
fig.savefig("plot.jpg") 
```

![](/img/matriz_modelo1.png)
<br />
<br />

**Conclusão sobre o modelo 1:** A partir dos resultados preliminares é possível notar que o modelo apresenta uma acurácia elevada para classificar os pulmões normais e com COVID-19. O próximo modelo treinará com imagens de pulmões que apresentam outras infecções, com o intuito de obter um modelo capaz de diferenciar a COVID-19 de outras infecções.<br />
<br />
<br />
**Observação:** os resultados não apresentam caráter clínico, mas sim exploratório. Contudo, com o aperfeiçoamento dos modelos, estes podem trazer benefícios para o enfrentamento à COVID-19.
<br />
<br />
**Bibliografia** <br />
COHEN, Joseph; MORRISON, Paul; DAO, Lan. COVID-19 Image Data Collection. arXiv:2003.11597, 2020.<br />
<br />
KERMANY, Daniel; ZHANG, Kang; GOLDBAUM, Michael. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v.2, 2018. Disponível em: http://dx.doi.org/10.17632/rscbjbr9sj.2
