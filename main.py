#encoding: utf-8
from cmath import inf
import cv2
import pprint
import os, re, glob
from cv2 import kmeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


PATH_BASE = ".\\img\\Padroes\\"

def obter_espacos_vetoriais(histo=False):
    esps = []
    filenames = glob.glob(os.path.join(PATH_BASE,'*.png'))
    color = ('b','g','r')

    for filename in filenames:
        img = cv2.imread(filename)

        if(histo):
            plt.figure()

            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                length = None
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
            continue

        else:
            average_color_row = np.average(img, axis=0)
            histr = np.average(average_color_row, axis=0)
            length = np.linalg.norm((0,0,0) - histr)

        esps.append((float(re.sub("[^0-9]","",filename))/100, histr, length))

    return sorted(esps,key= lambda dado: dado[0])

def exibir_clusters(sc, pixels):
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1])
    labels = sc.labels_

    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=labels.astype(int), edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("R value")
    ax.set_ylabel("G value")
    ax.set_zlabel("B value")
    ax.dist = 12

    fig.add_axes(ax)
    plt.show()

def redesenhar_clusterizado(img, sc, lista_dados):
    _, axs = plt.subplots(1, 2)
    img3 = np.ones((img.shape[0],img.shape[1],3),np.float64) * 255
    COLORS = sc.cluster_centers_.astype(np.uint8)

    for i,classe in enumerate(sc.labels_):
        pixel = [x for x in np.array(COLORS[classe])]
        row, col = lista_dados[i]
        img3[row][col] = pixel 
        

    centers = list(sc.cluster_centers_)
    patches = []
    for k,center in enumerate(centers):
        espessura = obter_espessura_estimada(center)
        n_center = [x/255 for x in center]
        if espessura > 1 and espessura < inf:
            patches.append(mpatches.Patch(color=n_center,label=('%.2f µm' % espessura).replace(".",",")))

    axs[1].imshow(img3.astype(np.uint8))
    axs[0].imshow(img)
    plt.legend(handles=sorted(patches, key=lambda p: np.linalg.norm(np.array((0,0,0,0))-np.array(p._facecolor))),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
    plt.show()

def calcular_vetor_proximo(vetor):

    dist = []
    extrema = [np.linalg.norm((0,0,0)-x[1]) for x in [ESPACO_VETORIAL_ESPESSURAS[1], ESPACO_VETORIAL_ESPESSURAS[-2]]]
    vec_len = np.linalg.norm((0,0,0)-vetor)

    if vec_len > extrema[0]:
       return ESPACO_VETORIAL_ESPESSURAS[-1] 
    elif vec_len < extrema[1]:
       return ESPACO_VETORIAL_ESPESSURAS[0] 
    else:
        for ref in [x[1] for x in ESPACO_VETORIAL_ESPESSURAS]:
            dist.append(np.linalg.norm(ref-vetor))
        ind_espessura = np.argmin(dist)
        return ESPACO_VETORIAL_ESPESSURAS[ind_espessura]

def obter_espessura_estimada(vetor):
    #TODO: Implementar função de interpolação

    feicao_representativa = calcular_vetor_proximo(vetor)
    return feicao_representativa[0]


def avaliar_elbow_curve(dados, plot=False):

    res = []
    means = []
    for n in range(3,10):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(dados)
        means.append(kmeans)
        res.append(kmeans.inertia_)

    if plot:
        plt.plot(range(3,10), res)
        plt.title('elbow curve')
        plt.show()

    return means

def remover_nao_pixel(pixels, posicoes):

    classificador = KMeans(n_clusters=2)
    classificador.fit(pixels)    

    novo_pixels = []
    novo_pos = []
    for i in range(len(posicoes)):

        label = classificador.labels_[i]
        dist_min = distancia(classificador.cluster_centers_[label], (255,255,255)) 
        dist_max = distancia(classificador.cluster_centers_[label], (0,0,0)) 

        if(dist_max < dist_min):
            novo_pixels.append(pixels[i])
            novo_pos.append(posicoes[i])

    return (novo_pixels, novo_pos)
            

def distancia(a,b=(0,0,0)):
    return np.linalg.norm(b-a)

def processar_imagem(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (800,800), interpolation = cv2.INTER_AREA)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    IMG_WIDTH = img.shape[0]
    IMG_HEIGHT = img.shape[1]
    pixels = []
    pos = []
    for row in range(IMG_WIDTH):
        for col in range(IMG_HEIGHT):
            vector = img[row][col]
            pos.append((row,col))
            pixels.append((vector).astype(np.uint8))

    pixels = np.array(pixels)
    
    #pré-processamento: Remover não pixels da análise com base no Kmeans com n=2
    pixels, pos = remover_nao_pixel(pixels, pos)

    clusters = avaliar_elbow_curve(pixels, True)

    for sc in clusters:
        redesenhar_clusterizado(img, sc, pos)

POS_INF = (0,0,0) #Consideramos preto como somente óleo
ZERO = (255,255,255) #Consideramos branco como fundo de bequer. Pode ser diferente quando for água profunda.
BANCO_ESPESSURAS = obter_espacos_vetoriais(histo=False)
ESPACO_VETORIAL_ESPESSURAS = [(np.inf, POS_INF)] + BANCO_ESPESSURAS +[(0, ZERO)] 

pprint.pprint(BANCO_ESPESSURAS)
processar_imagem('.\\img\\Testes\\Quente\\100 microns.jpg')