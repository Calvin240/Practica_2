import numpy as np
from matplotlib import pyplot as plt
import cv2 #Opencv
import math

img_1 = cv2.imread('m_1.jpg')
img_2 = cv2.imread('m_2.jpg')

rimg_1 = cv2.resize(img_1, (620,620))
rimg_2 = cv2.resize(img_2, (620,620))

cv2.imshow('IMAGEN 1',rimg_1)
cv2.imshow('IMAGEN 2',rimg_2)


K = cv2.waitKey (0) & 0xFF
if K == ord('k'):

    #SUMA
    suma = cv2.add(rimg_1,rimg_2)
    cv2.imshow('SUMA',suma)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #SUMA 2
    suma_2 = cv2.add(rimg_1,rimg_2)
    cv2.imshow('SUMA 2',suma_2)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #SUMA 3
    suma_3 = cv2.addWeighted(rimg_1,0.5,rimg_2,0.5,0)
    cv2.imshow('SUMA 3',suma_3)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #RESTA
    resta = cv2.subtract(rimg_1,rimg_2)
    cv2.imshow('RESTA',resta)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #RESTA 2
    resta_2 = cv2.subtract(rimg_1,rimg_2)
    cv2.imshow('RESTA 2',resta_2)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #RESTA 3
    resta_3 = cv2.absdiff(rimg_1,rimg_2)
    cv2.imshow('RESTA 3',resta_3)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #MULTIPLICACION
    multiplicacion = cv2.multiply(rimg_1,rimg_2)
    cv2.imshow('MULTIPLICACION',multiplicacion)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #MULTIPLICACION 2
    multiplicacion_2 = cv2.multiply(rimg_1,rimg_2)
    cv2.imshow('MULTIPLICACION 2',multiplicacion_2)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #DIVISION
    division = cv2.divide(rimg_1,rimg_2)
    cv2.imshow('DIVISION',division)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #DIVISION 2
    division_2 = cv2.divide(rimg_1,rimg_2)
    cv2.imshow('DIVISION 2',division_2)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #RAIZ
    raiz = (rimg_1**(0.5))
    cv2.imshow('Raiz',raiz)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #DERIVADA
    derivada = cv2.Laplacian(rimg_1,cv2.CV_64F)
    cv2.imshow('DERIVADA',derivada)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #POTENCIA
    Potencia = np.zeros(rimg_1.shape, rimg_1.dtype)
    g = 0.5
    c = 1
    Potencia = c * np.power(rimg_1,g)
    maxi1 = np.amax(Potencia)
    Potencia = np.uint8(Potencia/maxi1 * 255)
    cv2.imshow('POTENCIA',Potencia)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    Pow = cv2.pow(rimg_1,2)
    cv2.imshow('POTENCIA 2',Pow)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

    #LOGARITMO NATURAL
    logaritmo = np.zeros(rimg_1.shape, rimg_1.dtype)
    c = 1
    logaritmo = c * np.log(1+rimg_1)
    maxi = np.amax(logaritmo)
    logaritmo = np.uint8(logaritmo / maxi *255)
    cv2.imshow('LOGARITMO',logaritmo)
    cv2.waitKey(0) #Retardo
    cv2.destroyAllWindows()

cv2.waitKey(0) #Retardo
cv2.destroyAllWindows()

