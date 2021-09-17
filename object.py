# Cada objeto na cena do simulador sera um objeto da classe definida aqui

import math
import numpy as np

class Object:
    def __init__(self, name):
        self.name = name

        self.handler = -1
        self.position = None
        self.orientation = None

        self.transforms = None

# Achei que faria sentido ter Transforms como subclasse de Object, ja que essas transformacoes estao 
# associadas a cada objeto
class Transforms(Object):
    # Contem as matrizes de transformacao de cada objeto em relacao ao referencial global
    def __init__(self, theta, translation):
        # self.Rx = self.Rx(theta)
        # self.Ry = self.Ry(theta)
        self.Rz = self.Rz(theta)
        self.homogeneous_tf = self.homogeneous(theta, translation)
        self.inv_homogeneous_tf = self.inv_homogeneous(theta, translation)
    
    # Retorna a matriz de rotacao para theta em torno do eixo X
    def Rx(self, theta):
        return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]])

    
    # Retorna a matriz de rotacao para theta em torno do eixo Y
    def Ry(self, theta):
        return np.array([[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.cos(theta), 0.0, math.sin(theta)]])

    # Retorna a matriz de rotacao para teta em torno do eixo Z
    def Rz(self, theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])

    # Retorna a matriz de transformacao homogenea de {B} em relacao a {A}, para um theta e translacao dados
    def homogeneous(self, theta, translation):
        # Chamei de Rz_submatriz a parte esquerda da matriz homogenea que contem [[Rz], [0, 0, 0]]
        Rz_submatrix = np.vstack((self.Rz, np.zeros(3)))

        # Chamei de T_submatriz a parte direita da matriz de tf homogenea que contem [[translacao], [1]]
        # Obs.: como o vetor de ^AP_BORG vem como um array (vetor linha), tive que transforma-lo num vetor coluna 
        # antes usando o np.c_[x]
        T_submatriz = np.vstack((np.c_[translation], [1]))
        
        # Por fim, junta as duas submatrizes em uma matriz homogenea
        return np.hstack((Rz_submatrix, T_submatriz))

    def inv_homogeneous(self, theta, translation):
        Rz_submatrix = np.vstack((np.transpose(self.Rz), np.zeros(3)))
        T_submatriz = np.vstack((np.dot(-np.transpose(self.Rz), np.c_[translation]), [1]))

        return np.hstack((Rz_submatrix, T_submatriz))
