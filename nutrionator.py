import numpy as np
import numpy.typing as npt

'''
Chciałem zrobić klasę z interfejsem podobnym do klasyfikatorów z scikit learna.
Oczywiście udało się bardziej mniej niż więcej, ale oceńcie sami.

Nutrinator to nowy wynalazek dr Dundersztyca. 
Dzięki niemu będzie w stanie się dobrze odżywiać,
wykształci dobrze zbudowaną sylwetkę
i będzie mógł walczyć z PP Panem Dziobakiem pięść w pięść ... czy tam łapę ... czy co tam mają dziobaki

Dundersztyc będzie niepokonany - nie ma tutaj nawet przycisku autodestrukcji! 
'''
class Nutrinator:

    def __init__(self, N: dict, gamma=None):
        self.N = N
        self.gamma = gamma.T if gamma else np.array([1, 1, 1, 1]).T
        self.n_b = None
        self.n_s = None

    def fit(self, n_b, n_s=None):
        self.n_b = n_b
        self.n_s = n_s if n_s is not None else np.zeros(n_b.shape)

    def compute(self, A):
        return np.abs(self.gamma @ self.__d_n(A))

    # strasznie nie podoba mi się, że musiałem dodać tutaj pętle z pythona
    # Jeżeli macie pomysł na to, jak to przyśpieszyć, to dajcie znać,
    # mi się nie udało wykminić
    def __d_n(self, A):
        n_p_d = np.array([self.__n_p(A, d) for d in range(A.shape[0])])
        return np.sum(np.abs(n_p_d - self.n_b), axis=0)

    def __n_p(self, A, d):
        A_d = A[d, :]
        n_s_d = self.n_s[d, :, :]
        ni_A_d = np.array([self.__ni(A_d[i]) for i in range(A_d.shape[0])])
        return np.sum(ni_A_d - n_s_d, axis=0)

    def __ni(self, ind):
        return self.N[ind]


# Prezentacja działania wynalazku
if __name__=="__main__":
    # Poniższe 3 zmienne nie są potrzebne do Nutrinatora,
    # ale w celach prezentacji je tu zostawiłem, bo łatwiej to ogarnąć z analizą potem
    d = 7
    n = 3
    R = [0, 1, 2, 3]

    # Tę mapę dajemy w konstruktorze.
    # Kminiłem, jak to zamienić na NDArray, ale i tak w implementacji wyszło,
    # żę pętli for nie jestem w stanie uniknąć (patrz komentarz nad `__d_n`)
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

    # Konstrukcja
    ntr = Nutrinator(N)
    # Dodanie wektorów
    ntr.fit(n_b, n_s)
    # obliczenie S dla danej macierzy
    print(ntr.compute(A))
