import numpy as np


class Nutrinator:
    """
    Chciałem zrobić klasę z interfejsem podobnym do klasyfikatorów z scikit learna.
    Oczywiście udało się bardziej mniej niż więcej, ale oceńcie sami.

    Nutrinator to nowy wynalazek dr Dundersztyca.
    Dzięki niemu będzie w stanie się dobrze odżywiać,
    wykształci dobrze zbudowaną sylwetkę
    i będzie mógł walczyć z PP Panem Dziobakiem pięść w pięść ... czy tam łapę ... czy co tam mają dziobaki

    Dundersztyc będzie niepokonany - nie ma tutaj nawet przycisku autodestrukcji!
    """

    def __init__(self, N: np.ndarray, gamma=None):
        self.N = N
        self.gamma = gamma.T if gamma is not None else np.array([1, 1, 1, 1]).T
        self.n_b = None
        self.n_s = None

    def fit(self, n_b: np.ndarray, n_s: np.ndarray = None) -> None:
        self.n_b = n_b
        self.n_s = n_s if n_s is not None else np.zeros(n_b.shape)

    def compute(self, A: np.ndarray, P: np.ndarray) -> float:
        return np.abs(self.gamma @ self.__d_n(A, P))

    # strasznie nie podoba mi się, że musiałem dodać tutaj pętle z pythona
    # Jeżeli macie pomysł na to, jak to przyśpieszyć, to dajcie znać,
    # mi się nie udało wykminić
    def __d_n(self, A, P):
        n_p_d = np.array([self.__n_p(A, d, P) for d in range(A.shape[0])])
        return np.sum(np.abs(n_p_d - self.n_b), axis=0)

    def __n_p(self, A, d, P):
        A_d = A[d, :]
        n_s_d = self.n_s[d, :, :]
        ni_A_d = np.array([self.__ni(A_d[i]) * P[d, i] for i in range(A_d.shape[0])])
        return np.sum(ni_A_d - n_s_d, axis=0)

    def __ni(self, ind):
        return self.N[ind]
