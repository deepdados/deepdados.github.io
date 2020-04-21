---
layout: post
title: Modelo 2 - Tutorial 1 - Detecção automática de casos de COVID-19 a partir de imagens de radiografia de tórax
subtitle: Pré-processamento dos dados - Modelo 2
tags: [COVID]
---

**Objetivo principal do projeto:** automatizar o processo de detecção de casos de COVID-19 a partir de imagens de radiografia de tórax, utilizando redes neurais convolucionais (RNC) por meio de técnicas de aprendizado profundo (deep learning). O projeto completo pode ser acessado [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/projeto_cesar_lucas_COVID.pdf)

**Etapas para alcançar o objetivo:**<br />
1- Pré-processamento dos dados<br />
[2- Treinamento do modelo e exposição dos resultados](https://)


**Etapa 1 - Pré-processamento dos dados**

*Bases de dados utilizadas:*<br />
- Imagens de raio X e tomografias computadorizadas de tórax de indivíduos infectados com COVID-19 (COHE; MORRISON; DAO, 2020): [link](https://github.com/ieee8023/covid-chestxray-dataset)<br />
- Imagens de pulmões de indivíduos sem nenhuma infecção (KERMANY; ZHANG; GOLDBAUM, 2018): [link](https://data.mendeley.com/datasets/rscbjbr9sj/2)<br />

*Pacotes utilizados:*<br />
- Pandas<br />
- Os<br />
- PIL<br />
- Numpy<br />
- CV2<br />

*Código utilizado no projeto:*<br />
O notebook com todos os códigos utilizados nesta etapa está disponível [aqui](https://github.com/deepdados/ProjetoCOVID/blob/master/preProcessamento_COVID_modelo2.ipynb)<br />
**Observação:** a numeração e título de cada passo descrito neste tutorial, corresponde com a numeração e título contidos no notebook.

*Passos que serão seguidos:*<br />
**1º Passo** – [Importar as bibliotecas que serão utilizadas](#importar-as-bibliotecas-que-serão-utilizadas)<br />
**2º Passo** – [Carregar o dataframe referente às imagens de pulmões de indivíduos com COVID-19](#carregar-o-dataframe-referente-às-imagens-de-pulmões-de-indivíduos-com-covid-19)<br />
**3º Passo** – [Análise do dataframe “df”](#análise-do-dataframe-df)<br />
**4º Passo** – [Selecionar os casos relacionados à COVID-19 no dataframe “df”](#selecionar-os-casos-relacionados-à-covid-19-no-dataframe-df)<br />
**5º Passo** – [Análise do dataframe “df_covid”](#análise-do-dataframe-df_covid)<br />
**6º Passo** – [Criar uma lista para adicionar os valores da variável/coluna “filename”](#criar-uma-lista-para-adicionar-os-valores-da-variávelcoluna-filename)<br />
**7º Passo** – [Criar uma lista apenas com os formatos de imagens que existem na pasta de imagem](#criar-uma-lista-apenas-com-os-formatos-de-imagens-que-existem-na-pasta-de-imagem)<br />
**8º Passo** – [Criar uma função para abrir as imagens, observar as suas dimensões e, posteriormente, salvar estes dados em um dataframe](#criar-uma-função-para-abrir-as-imagens-observar-as-suas-dimensões-e-posteriormente-salvar-estes-dados-em-um-dataframe)<br />
**9º Passo** – [Criar uma variável que contenha como valor o endereço da pasta onde as imagens estão salvas](#criar-uma-variável-que-contenha-como-valor-o-endereço-da-pasta-onde-as-imagens-estão-salvas)<br />
**10º Passo** – [Utilizar a função criada para observar a dimensão das imagens](#utilizar-a-função-criada-para-observar-a-dimensão-das-imagens)<br />
**11º Passo** – [Converter todas as imagens para 237 x 237px .png](#converter-todas-as-imagens-para-237-x-237px-png)<br />
**12º Passo** – [Criar uma lista com as imagens que serão deletadas da pasta](#criar-uma-lista-com-as-imagens-que-serão-deletadas-da-pasta)<br />
**13º Passo** – [Abrir as imagens de pulmões de indivíduos sem infecção e criar uma lista com o nome das imagens que existem na pasta de imagem](#abrir-as-imagens-de-pulmões-de-indivíduos-sem-infecção-e-criar-uma-lista-com-o-nome-das-imagens-que-existem-na-pasta-de-imagem)<br />
**14º Passo** – [Converter todas as imagens de pulmões de indivíduos não infectados para 237 x 237px .png](#converter-todas-as-imagens-de-pulmões-de-indivíduos-não-infectados-para-237-x-237px-png)<br />
**15º Passo** - [Abrir as imagens de pulmões de indivíduos com outras infecções e criar uma lista com o nome das imagens que existem na pasta de imagem](abrir-as-imagens-de-pulmões-de-indivíduos-com-outras-infecções-e-criar-uma-lista-com-o-nome-das-imagens-que-existem-na-pasta-de-imagem)
**16º Passo** - [Converter todas as imagens de pulmões de indivíduos com outras infecções para 237 x 237px .png](converter-todas-as-imagens-de-pulmões-de-indivíduos-com-outras-infecções-para-237-x-237px-png)
**17º Passo** – [Abrir as imagens dos pulmões de indivíduos infectados com COVID-19 em uma lista e transformar estas em um array (matriz de valores dos pixels que representam a imagem)](#abrir-as-imagens-dos-pulmões-de-indivíduos-infectados-com-covid-19-em-uma-lista-e-transformar-estas-em-um-array-matriz-de-valores-dos-pixels-que-representam-a-imagem)<br />
**18º Passo** – [Abrir as imagens dos pulmões de indivíduos sem infecções em uma lista e transformar estas em um array (matriz de valores dos pixels que representam a imagem)](#abrir-as-imagens-dos-pulmões-de-indivíduos-sem-infecções-em-uma-lista-e-transformar-estas-em-um-array-matriz-de-valores-dos-pixels-que-representam-a-imagem)<br />
**19º Passo** - [Abrir as imagens dos pulmões de indivíduos com outras infecções em uma lista e transformar estas em um array (matriz de valores dos pixels que representam a imagem)](abrir-as-imagens-dos-pulmões-de-indivíduos-com-outras-infecções-em-uma-lista-e-transformar-estas-em-um-array-matriz-de-valores-dos-pixels-que-representam-a-imagem)
**20º Passo** - [Agrupar os arrays em um único array contendo informações sobre as imagens de COVID-19, normal e com outras infecções](agrupar-os-arrays-em-um-único-array-contendo-informações-sobre-as-imagens-de-covid-19-normal-e-com-outras-infecções)
**21º Pasoo** - [Indicar os casos que são COVID-19, os que são normais e os casos com outras infecções e criar um array](indicar-os-casos-que-são-covid-19-os-que-são-normais-e-os-casos-com-outras-infecções-e-criar-um-array)
**22º Passo** – [Salvar os arrays em .npy](#salvar-os-arrays-em-npy)<br />

**Tutorial 1:**

**1º Passo** 
#### Importar as bibliotecas que serão utilizadas

Importamos as bibliotecas Pandas, Os, PIL, Numpy e CV2 visto que nos apoiaremos nestas para realizar o pré-processamento dos dados do modelo referente à COVID-19.

``` python
import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
```

**Observação:** a biblioteca “Pandas” foi importada como “pd”, com o intuito de agilizar a escrita do código. Ou seja, ao invés de digitar “pandas” ao usá-la, digitarei apenas “pd”. O mesmo foi feito com a biblioteca “numpy”. Além disso, a biblioteca "PIL" não foi importada completamente, pois não utilizaremos todas as funções ali contidas. Desta forma, facilita a utilização da biblioteca e o processamento dos códigos/dados.

**2º Passo**
#### Carregar o dataframe referente às imagens de pulmões de indivíduos com COVID-19

Carregamos o arquivo em .csv, chamado “metadata”, que acompanha o banco de imagens disponibilizado pelos pesquisadores (COHE; MORRISON; DAO, 2020).<br />
<br />
O comando abaixo nomeia este dataframe de “df” ao carregá-lo. Entre os parênteses, você deve colocar o endereço que se encontra este arquivo.

``` python
df = pd.read_csv("/Users/Neto/Desktop/Aprendizados/2020/Kaggle/corona_deep_learning/covid-chestxray-dataset-master/metadata.csv")
```

**3º Passo**
#### Análise do dataframe “df”

Geramos alguns dados descritivos com o intuito de descobrir quantas imagens de COVID-19 estão disponíveis no dataframe (df). Para tanto, pedimos uma contagem de valores a partir da variável/coluna “finding”. Esta variável contém o diagnóstico relacionado a cada imagem de pulmão.

``` python
df.finding.value_counts()

COVID-19          188
Streptococcus      17
Pneumocystis       15
SARS               11
E.Coli              4
ARDS                4
COVID-19, ARDS      2
Chlamydophila       2
No Finding          2
Legionella          2
Klebsiella          1
Name: finding, dtype: int64
```
É possível notar a partir dos dados que 188 imagens se referem à COVID-19.<br />

**4º Passo**
#### Selecionar os casos relacionados à COVID-19 no dataframe “df”

Separamos apenas os casos da variável/coluna “finding” no dataframe “df” que eram COVID-19, visto que utilizaremos apenas estes casos no modelo. Salvamos esta seleção em um novo dataframe nomeado “df_covid”.

``` python
df_covid = df[df["finding"] == "COVID-19"]
```

**5º Passo**
#### Análise do dataframe “df_covid”

Pedimos para observar o dataframe “df_covid”, com o intuito de analisar se a seleção dos casos de COVID-19 foi realizada corretamente. Para tanto, pedimos para ver o final deste dataframe. Além disso, solicitamos que apenas as variáveis/colunas “finding” e “filename” fossem mostradas. A “finding”se refere aos casos de COVID-19 selecionados e a “filename” indica o nome das imagens de radiografia de COVID-19 disponibilizadas pelos autores do banco em questão (COHE; MORRISON; DAO, 2020). Esta última informação foi solicitada, visto que será utilizada no próximo passo.

``` python
df_covid[["finding","filename"]].tail()
	finding	filename
307	COVID-19	covid-19-pneumonia-58-day-9.jpg
308	COVID-19	covid-19-pneumonia-58-day-10.jpg
309	COVID-19	covid-19-pneumonia-mild.JPG
310	COVID-19	covid-19-pneumonia-67.jpeg
311	COVID-19	covid-19-pneumonia-bilateral.jpg
```

**6º Passo**
#### Criar uma lista para adicionar os valores da variável/coluna “filename”

Criamos uma lista a partir da variável/coluna “filename” localizada no dataframe “df_covid”. Esta foi denominada “imagensCOVID”. Esta lista apresenta apenas os nomes das imagens com pulmões de indivíduos infectados pelo vírus COVID-19. Esta lista foi criada para facilitar a seleção das imagens que utilizaremos para treinar o modelo.

``` python
imagensCOVID = df_covid["filename"].tolist()
```

**7º Passo**
#### Criar uma lista apenas com os formatos de imagens que existem na pasta de imagem

Ao checar manualmente a pasta onde as imagens se encontram, notou-se apenas os formatos “.jpg” e “.png”. No entanto, a variável/coluna “filename” tem entre os seus valores, imagens com extensão “.gz”. Dessa forma, criamos uma lista (“imagensCovid”) apenas com o nome das imagens nos formatos existentes na pasta (“.jpg” e “.png”).

``` python
imagensCovid = []
for imagem in imagensCOVID:
    if imagem.endswith(".gz"):
        pass
    else:
        imagensCovid.append(imagem)
        
print(len(imagensCovid))
```

**8º Passo**
#### Criar uma função para abrir as imagens, observar as suas dimensões e, posteriormente, salvar estes dados em um dataframe

Sabendo que esta ação será utilizada frequentemente nas etapas de pré-processamento de dados dos modelos que serão treinados, criamos uma função para facilitar a realização deste processo. Assim, a função abaixo (“df_dimensao”) define a criação de um dataframe com as dimensões das imagens localizadas em uma determinada pasta.

``` python
def df_dimensao(folder_das_imagens, lista_nome_imagens):
    """Função para criar um dataframe com as dimensões das imagens de uma pasta.
    Parâmetros:
    
    folder_das_imagens(str): colocar a pasta onde as imagens estão salvas
    lista_nome_imagens(list): colocar a lista com o nome das imagens
    
    return
    
    df_dims(pd.DataFrame)
    
    """
    
    dic = {}
    dimensaoImagensLargura = []
    dimensaoImagensAltura = []
    nome = []
    
    if ".DS_Store" in lista_nome_imagens:
        lista_nome_imagens.remove(".DS_Store")
    for imagem in lista_nome_imagens:
        
        enderecoDaImagem = folder_das_imagens + "/" + imagem
        abrirImagem = Image.open(enderecoDaImagem)
        nome.append(imagem)
        dimensaoImagensLargura.append(abrirImagem.size[0])
        dimensaoImagensAltura.append(abrirImagem.size[1])

    dic["nome"] = nome
    dic["largura"] = dimensaoImagensLargura
    dic["altura"] = dimensaoImagensAltura
    df_dims = pd.DataFrame(dic)
    
    return df_dims
```

**9º Passo**
#### Criar uma variável que contenha como valor o endereço da pasta onde as imagens estão salvas

Com o intuito de utilizar a função criada no Passo 8, em específico o parâmetro “folder_das_imagens(str)”, devemos ter uma variável string que indique o endereço das imagens no computador. Para tanto, o código abaixo cria uma variável (“rootFolder”) indicando esta localização.

``` python
rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images"
```

**Observação:** em relação ao outro atributo da função denominado “lista_nome_imagens”, utilizaremos a lista criada no Passo 7 (“imagensCovid”).<br />

**10º Passo**
#### Utilizar a função criada para observar a dimensão das imagens

A partir da função criada, salvamos os valores na variável “dimensão”. Abaixo é possível observar os nomes de cada figura e a sua dimensão (largura x altura) em pixels.

``` python
dimensao = df_dimensao(rootFolder, imagensCovid)
print(dimensao)
                                                  nome  largura  altura
0    auntminnie-a-2020_01_28_23_51_6665_2020_01_28_...      882     888
1    auntminnie-b-2020_01_28_23_51_6665_2020_01_28_...      880     891
2    auntminnie-c-2020_01_28_23_51_6665_2020_01_28_...      882     876
3    auntminnie-d-2020_01_28_23_51_6665_2020_01_28_...      880     874
4                                nejmc2001573_f1a.jpeg     1645    1272
..                                                 ...      ...     ...
211                    covid-19-pneumonia-58-day-9.jpg     2267    1974
212                   covid-19-pneumonia-58-day-10.jpg     2373    2336
213                        covid-19-pneumonia-mild.JPG      867     772
214                         covid-19-pneumonia-67.jpeg      492     390
215                   covid-19-pneumonia-bilateral.jpg     2680    2276

[216 rows x 3 columns]
```

**Observação:** este passo é importante, pois para executar o modelo, todas as imagens devem ter a mesma dimensão.<br />

**11º Passo**
#### Converter todas as imagens para 237 x 237px .png

Visto que para rodar o modelo precisamos ter todas as imagens com a mesma dimensão, optamos por reduzir todas para a dimensão da menor figura disponível no banco de imagens. Além disso, para manter um padrão, alteramos o formato para “.png” de todas as figuras, visto que algumas eram “.jpg”.

O código abaixo redimensiona as imagens para 237 x 237px, salva elas em uma outra pasta e executa a função que construímos no Passo 8 para observar se todas as dimensões foram alteradas. 

``` python
for imagem in imagensCovid:
    enderecoDaImagem = rootFolder + "/" + imagem
    abrirImagem = Image.open(enderecoDaImagem)
    image_resize = abrirImagem.resize((237,237))
    os.chdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize")
    image_resize.save(f'{imagem}_resize_237_237.png')
    
    
rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize"
imagensDaPastaResize = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize")
df_redimensao = df_dimensao(rootFolder, imagensDaPastaResize)
print(df_redimensao)
                                                  nome  largura  altura
0    01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg_resi...      237     237
1    39EE8E69-5801-48DE-B6E3-BE7D1BCF3092.jpeg_resi...      237     237
2                 lancet-case2b.jpg_resize_237_237.png      237     237
3             nejmoa2001191_f4.jpeg_resize_237_237.png      237     237
4    7C69C012-7479-493F-8722-ABC29C60A2DD.jpeg_resi...      237     237
..                                                 ...      ...     ...
211  23E99E2E-447C-46E5-8EB2-D35D12473C39.png_resiz...      237     237
212  covid-19-pneumonia-43-day2.jpeg_resize_237_237...      237     237
213    radiol.2020201160.fig6b.jpeg_resize_237_237.png      237     237
214  8FDE8DBA-CFBD-4B4C-B1A4-6F36A93B7E87.jpeg_resi...      237     237
215      covid-19-pneumonia-7-L.jpg_resize_237_237.png      237     237

[216 rows x 3 columns]
```

**Observação:** como é possível notar, todas as figuras apresentam a mesma dimensão (largura x altura).<br />

**12º Passo**
#### Criar uma lista com as imagens que serão deletadas da pasta

Criamos uma lista com o nome das imagens que foram deletadas da pasta. Os autores deste modelo decidiram não incluir as imagens laterais e de tomografia computadorizada existentes no banco de imagens original. Assim, a variável “listaImagemDeletar” apresenta como valor uma lista com o nome destas imagens.

``` python
listaImagemDeletar = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/deletadas")
listaImagemDeletar = ['covid-19-pneumonia-30-L.jpg_resize_237_237.png',
 '396A81A5-982C-44E9-A57E-9B1DC34E2C08.jpeg_resize_237_237.png',
 'covid-19-infection-exclusive-gastrointestinal-symptoms-l.png_resize_237_237.png',
 'nejmoa2001191_f3-L.jpeg_resize_237_237.png',
 '3ED3C0E1-4FE0-4238-8112-DDFF9E20B471.jpeg_resize_237_237.png',
 'covid-19-pneumonia-38-l.jpg_resize_237_237.png',
 'a1a7d22e66f6570df523e0077c6a5a_jumbo.jpeg_resize_237_237.png',
 '254B82FC-817D-4E2F-AB6E-1351341F0E38.jpeg_resize_237_237.png',
 'covid-19-pneumonia-15-L.jpg_resize_237_237.png',
 'kjr-21-e24-g002-l-b.jpg_resize_237_237.png',
 'D5ACAA93-C779-4E22-ADFA-6A220489F840.jpeg_resize_237_237.png',
 'kjr-21-e24-g002-l-c.jpg_resize_237_237.png',
 'covid-19-pneumonia-14-L.png_resize_237_237.png',
 'kjr-21-e24-g004-l-a.jpg_resize_237_237.png',
 'nejmoa2001191_f1-L.jpeg_resize_237_237.png',
 'kjr-21-e24-g003-l-b.jpg_resize_237_237.png',
 'kjr-21-e24-g004-l-b.jpg_resize_237_237.png',
 'DE488FE1-0C44-428B-B67A-09741C1214C0.jpeg_resize_237_237.png',
 '191F3B3A-2879-4EF3-BE56-EE0D2B5AAEE3.jpeg_resize_237_237.png',
 '35AF5C3B-D04D-4B4B-92B7-CB1F67D83085.jpeg_resize_237_237.png',
 '6A7D4110-2BFC-4D9A-A2D6-E9226D91D25A.jpeg_resize_237_237.png',
 '4C4DEFD8-F55D-4588-AAD6-C59017F55966.jpeg_resize_237_237.png',
 'covid-19-caso-70-1-L.jpg_resize_237_237.png',
 '44C8E3D6-20DA-42E9-B33B-96FA6D6DE12F.jpeg_resize_237_237.png',
 'kjr-21-e24-g001-l-b.jpg_resize_237_237.png',
 'FC230FE2-1DDF-40EB-AA0D-21F950933289.jpeg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-a.jpg_resize_237_237.png',
 '925446AE-B3C7-4C93-941B-AC4D2FE1F455.jpeg_resize_237_237.png',
 'jkms-35-e79-g001-l-e.jpg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-b.jpg_resize_237_237.png',
 '21DDEBFD-7F16-4E3E-8F90-CB1B8EE82828.jpeg_resize_237_237.png',
 'covid-19-pneumonia-evolution-over-a-week-1-day0-L.jpg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-d.jpg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-c.jpg_resize_237_237.png',
 'nejmoa2001191_f5-L.jpeg_resize_237_237.png',
 'jkms-35-e79-g001-l-d.jpg_resize_237_237.png',
 'covid-19-pneumonia-22-day1-l.png_resize_237_237.png',
 'kjr-21-e24-g001-l-c.jpg_resize_237_237.png',
 '66298CBF-6F10-42D5-A688-741F6AC84A76.jpeg_resize_237_237.png',
 'covid-19-pneumonia-20-l-on-admission.jpg_resize_237_237.png',
 'covid-19-pneumonia-7-L.jpg_resize_237_237.png']
```

**13º Passo**
#### Abrir as imagens de pulmões de indivíduos sem infecção e criar uma lista com o nome das imagens que existem na pasta de imagem

Após criar uma variável denominada “pastaTreinoNormal” com o endereço da pasta com as imagens de pulmões de indivíduos sem infecção, criamos uma lista (“listaImagensTreino”) apenas com o nome e formato destas imagens.

``` python
pastaTreinoNormal = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL"

listaImagensTreino = os.listdir(pastaTreinoNormal)
```

**14º Passo**
#### Converter todas as imagens de pulmões de indivíduos não infectados para 237 x 237px .png

Foram redimensionadas as imagens de pulmões normais para a mesma dimensão das imagens dos pulmões com COVID-19: a saber, 237 x 237px. Para manter o mesmo padrão, alteramos o formato para “.png” de todas as figuras. É importante destacar que selecionamos através do código abaixo apenas as 100 primeiras imagens da pasta. Isto foi realizado para manter o treino com uma quantidade de imagem similar de indivíduos sem nenhuma infecção e com COVID-19. 

Além disso, executamos a função que construímos no Passo 8 para observar se todas as dimensões foram alteradas.

``` python
listaCemImagens = listaImagensTreino[0:100]
for imagem in listaCemImagens:
    enderecoDaImagem = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL"+ "/" + imagem
    abrirImagem = Image.open(enderecoDaImagem)
    image_resize = abrirImagem.resize((237,237))
    os.chdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal")
    image_resize.save(f'{imagem}_resize_237_237.png')
  
rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal"
imagensDaPastaResize = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal")
df_redimensao = df_dimensao(rootFolder, imagensDaPastaResize)
print(df_redimensao)
                                            nome  largura  altura
0   NORMAL2-IM-1196-0001.jpeg_resize_237_237.png      237     237
1   NORMAL2-IM-0645-0001.jpeg_resize_237_237.png      237     237
2           IM-0269-0001.jpeg_resize_237_237.png      237     237
3   NORMAL2-IM-1131-0001.jpeg_resize_237_237.png      237     237
4      IM-0545-0001-0002.jpeg_resize_237_237.png      237     237
..                                           ...      ...     ...
95  NORMAL2-IM-0592-0001.jpeg_resize_237_237.png      237     237
96  NORMAL2-IM-1167-0001.jpeg_resize_237_237.png      237     237
97  NORMAL2-IM-0741-0001.jpeg_resize_237_237.png      237     237
98  NORMAL2-IM-0535-0001.jpeg_resize_237_237.png      237     237
99          IM-0119-0001.jpeg_resize_237_237.png      237     237

[100 rows x 3 columns]
```

**Observação:** como é possível notar, todas as figuras apresentam a mesma dimensão (largura x altura).<br />

**15º Passo**
#### Abrir as imagens de pulmões de indivíduos com outras infecções e criar uma lista com o nome das imagens que existem na pasta de imagem

Após criar uma variável denominada “pastaTreinoOutrasInfeccoes” com o endereço da pasta com as imagens de pulmões de indivíduos outras infecções (isto é, sem ser a COVID-19), criamos uma lista (“listaImagensTreinoOutrasInfeccoes”) apenas com o nome e formato destas imagens.

``` python
pastaTreinoOutrasInfeccoes = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA"

listaImagensTreinoOutrasInfeccoes = os.listdir(pastaTreinoOutrasInfeccoes)
```

**16º Passo**
#### Converter todas as imagens de pulmões de indivíduos com outras infecções para 237 x 237px .png

Foram redimensionadas as imagens de pulmões com outras infecções (isto é, sem ser a COVID-19) para a mesma dimensão das imagens dos pulmões com COVID-19 e normal: a saber, 237 x 237px. Para manter o mesmo padrão, alteramos o formato para “.png” de todas as figuras. É importante destacar que selecionamos através do código abaixo apenas as 100 primeiras imagens da pasta. Isto foi realizado para manter o treino com uma quantidade de imagem similar de indivíduos com outras infecções, sem nenhuma infecção e com COVID-19.

Além disso, executamos a função que construímos no Passo 8 para observar se todas as dimensões foram alteradas.

```python
listaCemImagensOutrasInfeccoes = listaImagensTreinoOutrasInfeccoes[0:100]
for imagem in listaCemImagensOutrasInfeccoes:
    enderecoDaImagem = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA"+ "/" + imagem
    abrirImagem = Image.open(enderecoDaImagem)
    image_resize = abrirImagem.resize((237,237))
    os.chdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA/images_resize_infeccoes")
    image_resize.save(f'{imagem}_resize_237_237.png')

rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA/images_resize_infeccoes"
imagensDaPastaResize = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA/images_resize_infeccoes")
df_redimensao = df_dimensao(rootFolder, imagensDaPastaResize)
print(df_redimensao)

                                                nome  largura  altura
0    person890_bacteria_2814.jpeg_resize_237_237.png      237     237
1   person1016_bacteria_2947.jpeg_resize_237_237.png      237     237
2    person306_bacteria_1439.jpeg_resize_237_237.png      237     237
3    person472_bacteria_2015.jpeg_resize_237_237.png      237     237
4   person1491_bacteria_3893.jpeg_resize_237_237.png      237     237
..                                               ...      ...     ...
95   person364_bacteria_1660.jpeg_resize_237_237.png      237     237
96     person1455_virus_2489.jpeg_resize_237_237.png      237     237
97     person1238_virus_2098.jpeg_resize_237_237.png      237     237
98      person620_virus_1191.jpeg_resize_237_237.png      237     237
99     person26_bacteria_122.jpeg_resize_237_237.png      237     237

[100 rows x 3 columns]
```
**Observação:** como é possível notar, todas as figuras apresentam a mesma dimensão (largura x altura).<br />

**17º Passo**
#### Abrir as imagens dos pulmões de indivíduos infectados com COVID-19 em uma lista e transformar estas em um array (matriz de valores dos pixels que representam a imagem)

Primeiramente, a partir das imagens redimensionadas de pulmões de indivíduos com COVID-19 obtidas no Passo 11, criamos uma variável (“imagensCovid”) com a lista de nomes destas imagens. Em seguida, utilizando a lista de imagens que não foram utilizadas no modelo (laterais e tomografia computadorizada), referente ao Passo 12, estas foram deletadas dos valores da variável (“imagensCovid”).

Posteriormente, criamos uma lista com os arrays denominada “XTrainCovid” a partir das imagens redimensionadas, isto é, uma lista com os valores referentes aos pixels que representam as figuras de pulmões de indivíduos infectados pela COVID-19.

Por último, salvamos a lista “XTrainCovid” em um array denominado “xArrayCOVID”.

``` python
imagensCovid = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize")
imagensCovid = [x for x in imagensCovid if x not in listaImagemDeletar]

if ".DS_Store" in imagensCovid:
    imagensCovid.remove(".DS_Store")

xTrainCovid = []

for image in imagensCovid:
    x = cv2.imread("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize/" + image)
    x = np.array(x)
    xTrainCovid.append(x)

xArrayCOVID = np.array(xTrainCovid)
print(xArrayCOVID.shape)

(175, 237, 237, 3)
```

**Observação:** como é possível notar, o array construído (“xArrayCOVID”) possui quatro dimensões. A primeira (“175”) se refere à quantidade de casos, ou seja, de imagens de indivíduos com COVID-19; a segunda (“237”) se refere à largura da imagem; a terceira (“237”) se refere à altura da imagem e; a quarta (“3”), à quantidade de canais de cores existentes nas imagens.<br />

**18º Passo**
#### Abrir as imagens dos pulmões de indivíduos sem infecções em uma lista e transformar estas em um array (matriz de valores dos pixels que representam a imagem)

Primeiramente, a partir das imagens redimensionadas de pulmões de indivíduos sem infecções obtidas no Passo 13, criamos uma variável (“imagensNormal”) com a lista de nomes destas imagens.

Em um segundo momento, criamos uma lista com os arrays denominada “XTrainNormal” a partir das imagens redimensionadas, isto é, uma lista com os valores referentes aos pixels que representam as figuras de pulmões de indivíduos sem infecções.

Por último, salvamos a lista “XTrainNormal” em um array denominado “xArrayNormal”.

``` python
imagensNormal = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal")

if ".DS_Store" in imagensNormal:
    imagensNormal.remove(".DS_Store")

xTrainNormal = []

for image in imagensNormal:
    x = cv2.imread("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal/" + image)
    x = np.array(x)
    xTrainNormal.append(x)

xArrayNormal = np.array(xTrainNormal)
print(xArrayNormal.shape)

(100, 237, 237, 3)
```

**Observação:** como é possível notar, o array construído (“xArrayNormal”) possui quatro dimensões. A primeira (“100”) se refere à quantidade de casos, ou seja, de imagens de indivíduos sem infecções; a segunda (“237”) se refere à largura da imagem; a terceira (“237”) se refere à altura da imagem e; a quarta (“3”), à quantidade de canais de cores existentes nas imagens.<br />

**19º Passo**
#### Abrir as imagens dos pulmões de indivíduos com outras infecções em uma lista e transformar estas em um array (matriz de valores dos pixels que representam a imagem)

Primeiramente, a partir das imagens redimensionadas de pulmões de indivíduos com outras infecções obtidas no Passo 16, criamos uma variável (“imagensInfeccoes”) com a lista de nomes destas imagens.

Em um segundo momento, criamos uma lista com os arrays denominada “XTrainInfeccoes” a partir das imagens redimensionadas, isto é, uma lista com os valores referentes aos pixels que representam as figuras de pulmões de indivíduos com outras infecções.

Por último, salvamos a lista “XTrainInfeccoes” em um array denominado “xArrayInfeccoes”.

```python
imagensInfeccoes = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA/images_resize_infeccoes")

if ".DS_Store" in imagensNormal:
    imagensNormal.remove(".DS_Store")

xTrainInfeccoes = []

for image in imagensInfeccoes:
    x = cv2.imread("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/PNEUMONIA/images_resize_infeccoes/" + image)
    x = np.array(x)
    xTrainInfeccoes.append(x)

xArrayInfeccoes = np.array(xTrainInfeccoes)
print(xArrayInfeccoes.shape)

(100, 237, 237, 3)
```

**Observação:** como é possível notar, o array construído (“xArrayInfeccoes”) possui quatro dimensões. A primeira (“100”) se refere à quantidade de casos, ou seja, de imagens de indivíduos com outras infecções; a segunda (“237”) se refere à largura da imagem; a terceira (“237”) se refere à altura da imagem e; a quarta (“3”), à quantidade de canais de cores existentes nas imagens.<br />

**20º Passo**
#### Agrupar os arrays em um único array contendo informações sobre as imagens de COVID-19, normal e com outras infecções

Agrupamos o array das imagens de indivíduos com COVID-19 (“xArrayCOVID”), criado no Passo 17, com o array de imagens de indivíduos sem infecções (“xArrayNormal”), criado no Passo 18, e, por último, com o array de imagens de indivíduos com outras infecções (“xArrayInfeccoes”), criado no Passo 19. Este array foi salvo na variável “X_train”.

``` python
X_train = np.vstack((xArrayCOVID,xArrayNormal, xArrayInfeccoes))
```

**21º Passo**
#### Indicar os casos que são COVID-19, os que são normais e os casos com outras infecções e criar um array

Foram criados três arrays. O primeiro indica a variável “dfCOVID” com o valor “1”, indicando a presença de COVID-1, e a variável “dfNormal” e a variável "dfInfeccoes" com o valor “0”, indicando a ausência de imagens com estas características. O segundo indicando "1" para "dfNormal e "0" para "dfCOVID" e "dfInfeccoes". E o terceiro "1" para "dfInfeccoes" e "0" para "dfCOVID" e "dfNormal".

Por último, agrupamos o array das imagens de indivíduos com COVID-19 (“Y_train_COVID”) com o array de imagens de indivíduos sem infecções (“Y_train_NORMAL”) com o array de indivíduos com outras infecções ("Y_train_INFECCOES"). Este array foi salvo na variável “Y_train”.

``` python
dfCOVID = np.ones((xArrayCOVID.shape[0],1))
dfNormal = np.zeros((xArrayNormal.shape[0],1))
dfInfeccoes = np.zeros((xArrayInfeccoes.shape[0],1))

Y_train_COVID = np.vstack((dfCOVID,dfNormal, dfInfeccoes))

dfCOVID = np.zeros((xArrayCOVID.shape[0],1))
dfNormal = np.ones((xArrayNormal.shape[0],1))
dfInfeccoes = np.zeros((xArrayInfeccoes.shape[0],1))

Y_train_NORMAL = np.vstack((dfCOVID,dfNormal, dfInfeccoes))

dfCOVID = np.zeros((xArrayCOVID.shape[0],1))
dfNormal = np.zeros((xArrayNormal.shape[0],1))
dfInfeccoes = np.ones((xArrayInfeccoes.shape[0],1))

Y_train_INFECCOES = np.vstack((dfCOVID,dfNormal, dfInfeccoes))

Y_train = np.hstack((Y_train_COVID, Y_train_NORMAL, Y_train_INFECCOES))
```

**22º Passo**
#### Salvar os arrays em .npy

Para utilizar os arrays no treinamento do modelo, estes foram salvos em “X_Train.npy” e “Y_Train.npy”.

``` python
np.save("/Users/cesarsoares/Documents/Python/COVID/X_Train.npy",X_train)
np.save("/Users/cesarsoares/Documents/Python/COVID/Y_Train.npy", Y_train)
```

**Observação:** o X_Train será o input do modelo treinado e o Y_Train o target, ou seja, o resultado esperado do modelo.(br />

**Bibliografia** <br />
COHEN, Joseph; MORRISON, Paul; DAO, Lan. COVID-19 Image Data Collection. arXiv:2003.11597, 2020.<br />
<br />
KERMANY, Daniel; ZHANG, Kang; GOLDBAUM, Michael. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v.2, 2018. Disponível em: http://dx.doi.org/10.17632/rscbjbr9sj.2
