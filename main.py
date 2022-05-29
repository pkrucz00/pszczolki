import numpy as np
from nutrinator import Nutrinator


# Prezentacja działania wynalazku
if __name__ == "__main__":
    # Poniższe 3 zmienne nie są potrzebne do Nutrinatora,
    # ale w celach prezentacji je tu zostawiłem, bo łatwiej to ogarnąć z analizą potem
    d = 7
    n = 3
    R = [0, 1, 2, 3]

    N = np.array([[20, 30, 20, 10],
                 [40, 50, 20, 20],
                 [50, 10, 10, 30],
                 [10, 40, 30, 10]])

    # Wektory przykładowe wg analizy. Zrobione bardzo leniwe, ale kształt się zgadza:)
    n_b = np.array([[100, 100, 100, 100],
                    [100, 100, 100, 100],
                    [100, 100, 100, 100],
                    [100, 100, 100, 100],
                    [100, 100, 100, 100],
                    [100, 100, 100, 100],
                    [100, 100, 100, 100]])
    n_s = np.zeros((d, n, 4))

    # Rozpiska dań na 7 dni po 3 na każdy dzień
    A = np.array([[2, 3, 1],
                  [1, 3, 0],
                  [3, 2, 0],
                  [2, 3, 2],
                  [0, 1, 3],
                  [1, 2, 3],
                  [1, 2, 3]])

    P = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]])

    # Konstrukcja
    ntr = Nutrinator(N)
    # Dodanie wektorów
    ntr.fit(n_b, n_s)
    # obliczenie S dla danej macierzy
    print(ntr.compute(A, P))
