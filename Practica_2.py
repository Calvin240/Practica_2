import numpy as np
from matplotlib import pyplot as plt
import cv2 #Opencv
import math
import skimage
from skimage import io

img_1 = cv2.imread('m_1.jpg')
img_2 = cv2.imread('m_2.jpg')

rimg_1 = cv2.resize(img_1, (300,300))
rimg_2 = cv2.resize(img_2, (300,300))

cv2.imshow('IMAGEN 1',rimg_1)
cv2.imshow('IMAGEN 2',rimg_2)

#SUMA

    #METODO 1
    
suma = cv2.add(rimg_1,rimg_2)
cv2.imshow('Suma 1',suma)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Suma 1')

    #METODO 2
    
suma_2 = cv2.add(rimg_1,rimg_2)
cv2.imshow('Suma 2',suma_2)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Suma 2')

    #METODO 3
    
suma_3 = cv2.addWeighted(rimg_1,0.5,rimg_2,0.5,0)
cv2.imshow('Suma 3',suma_3)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Suma 3')

#RESTA

    #METODO 1
    
resta = cv2.subtract(rimg_1,rimg_2)
cv2.imshow('Resta 1',resta)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Resta 1')

    #METODO 2
    
resta_2 = cv2.subtract(rimg_1,rimg_2)
cv2.imshow('Resta 2',resta_2)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Resta 2')

    #METODO 3
resta_3 = cv2.absdiff(rimg_1,rimg_2)
cv2.imshow('Resta 3',resta_3)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Resta 3')

#MULTIPLICACION

    #METODO 1
    
multiplicacion = cv2.multiply(rimg_1,rimg_2)
cv2.imshow('Multiplicacion 1',multiplicacion)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Multiplicacion 1')

    #METODO 2
multiplicacion_2 = cv2.multiply(rimg_1,rimg_2)
cv2.imshow('Multiplicacion 2',multiplicacion_2)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Multiplicacion 2')

#DIVISION

    #METODO 1
    
division = cv2.divide(rimg_1,rimg_2)
cv2.imshow('Division 1',division)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Division 1')

    #METODO 2

division_2 = cv2.divide(rimg_1,rimg_2)
cv2.imshow('Division 2',division_2)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Division 2')

#RAIZ
raiz = (rimg_1**(0.5))
cv2.imshow('Raiz',raiz)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Raiz')

#DERIVADA
derivada = cv2.Laplacian(rimg_1,cv2.CV_64F)
cv2.imshow('Derivada',derivada)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Derivada')

#POTENCIA

    #METODO 1
    
Potencia = np.zeros(rimg_1.shape, rimg_1.dtype)
g = 0.5
c = 1
Potencia = c * np.power(rimg_1,g)
maxi1 = np.amax(Potencia)
Potencia = np.uint8(Potencia/maxi1 * 255)
cv2.imshow('Potencia 1',Potencia)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Potencia 1')

    #METODO 2

Pow = cv2.pow(rimg_1,2)
cv2.imshow('Potencia 2',Pow)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Potencia 2')

#LOGARITMO NATURAL
logaritmo = np.zeros(rimg_1.shape, rimg_1.dtype)
c = 1
logaritmo = c * np.log(1+rimg_1)
maxi = np.amax(logaritmo)
logaritmo = np.uint8(logaritmo / maxi *255)
cv2.imshow('Logaritmo',logaritmo)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Logaritmo')

#CONJUNCION
opand = cv2.bitwise_and(rimg_1,rimg_2)
cv2.imshow('Conjuncion', opand)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Conjuncion')
  
#DISYUNCION
opor = cv2.bitwise_or(rimg_1,rimg_2)
cv2.imshow('Disyuncion', opor)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Disyuncion')

#NEGACION

    #METODO 1
    
height, width, _ = rimg_1.shape

for i in range(0, height - 1):
    for j in range(0, width -1):
        pixel = rimg_1[i,j]
        pixel[0] = 255 - pixel[0]
        pixel[1] = 255 - pixel[1]
        pixel[2] = 255 - pixel[2]
        rimg_1[i,j] = pixel
cv2.imshow('Negacion 1',rimg_1)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Negacion 1')

    #METODO 2

img_neg = 1 - rimg_2
cv2.imshow('Negacion 2',img_neg)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Negacion 2')

#TRASLACION

ancho = rimg_1.shape[1] #columnas
alto = rimg_1.shape[0] # filas
M = np.float32([[1,0,150],[0,1,100]]) #Construccion de la matriz
imageOut = cv2.warpAffine(rimg_1,M,(ancho,alto))
cv2.imshow('Traslacion',imageOut)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Traslacion')

#ESCALADO
    
eimg = cv2.resize(img_2, dsize=(200, 200))
cv2.imshow('Escalado',eimg)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Escalado')

#ROTACION

    #METODO 1
    
rimg = io.imread("m_1.jpg")
type(rimg)
rimg.shape
plt.imshow(rimg[::-1])###Invertir imagen
plt.show()

    #METODO 2

ancho = rimg_2.shape[1] #columnas
alto = rimg_2.shape[0] # filas
    
Rotacion = cv2.getRotationMatrix2D((ancho//2,alto//2),180,1)
imageOut = cv2.warpAffine(rimg_2,Rotacion,(ancho,alto))
cv2.imshow('Rotacion',imageOut)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Rotacion')

#TRASLACION A FIN

    #METODO 1
    
ancho = rimg_1.shape[1] #columnas
alto = rimg_1.shape[0] # filas
M = np.float32([[1,0,200],[0,1,150]]) #Construccion de la matriz
imageOut = cv2.warpAffine(rimg_1,M,(ancho,alto))
cv2.imshow('Traslacion a fin 1',imageOut)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Traslacion a fin 1')

    #METODO 2

rows, cols, ch = rimg_1.shape
pts1 = np.float32([[50, 50],
                    [200, 50], 
                    [50, 200]])
pts2 = np.float32([[10, 100],
                    [200, 50], 
                    [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(rimg_1, M, (cols, rows))

cv2.imshow('Traslacion a fin 2',dst)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Traslacion a fin 2')

    #METODO 3

rows,cols,ch = rimg_1.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
     
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(rimg_1,M,(300,300))

cv2.imshow('Traslacion a fin 3',dst)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Traslacion a fin 3')

#TRANSPUESTA

    #METODO 1

def transponer(rimg_1):
    t = []
    for i in range(len(rimg_1[0])):
        t.append([])
        for j in range(len(rimg_1)):
            t[i].append(rimg_1[j][i])
    return t
c = np.concatenate(transponer(rimg_1),axis=1)
#cv2.imshow('Transpuesta 1', c)
#cv2.waitKey(0) #Retardo
#cv2.destroyAllWindows()


    #METODO 2

transpuesta = cv2.transpose(rimg_1)
cv2.imshow('Transpuesta 2', transpuesta)
cv2.waitKey(0) #Retardo
cv2.destroyWindow('Transpuesta 2')
        
cv2.waitKey(0) #Retardo
cv2.destroyAllWindows()

