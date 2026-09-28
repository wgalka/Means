import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

'''
Feature selection based on fuzzy combination entropy considering
global and local feature correlation.

Python implementation based on the MATLAB code supplied by the author.
'''


class FSmFCE:
    def __init__(self, alfa=1, beta=1, gamma=1, lambd=0.5, k=0.001, verbose=0):
        self.alfa = alfa
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self.k = k
        self.verbose = verbose

        self.S = set()

    def fit(self, X, y=None):
        self.S = self.FSmFCE(
            X,
            y,
            alfa=self.alfa,
            beta=self.beta,
            gamma=self.gamma,
            lambd=self.lambd,
            k=self.k,
            verbose=self.verbose,
        )
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, sorted(self.S)]
        return X[:, sorted(self.S)]

    def get_support(self):
        return self.S

    def fuzzy_similarity(self, X, verbose=0):
        m_objects, n_features = X.shape  # m - liczba obiektów, n - liczba cech
        similarity_matrix = np.zeros((n_features, m_objects, m_objects))

        # Odchylenie standardowe z próby, tak jak w kodzie MATLAB autora.
        std_a = np.std(X, axis=0, ddof=1)
        if verbose > 0:
            print("delta (std) for each feature:\n", std_a)

        for feature_index in range(n_features):
            column = X[:, feature_index].reshape(-1, 1)
            diff_squared = pairwise_distances(column, metric="euclidean") ** 2
            similarity_matrix[feature_index, :, :] = np.exp(
                -diff_squared / (2 * std_a[feature_index] ** 2)
            )

        if verbose > 0:
            print("similarity matrix for each feature:", similarity_matrix.shape)
        return similarity_matrix

    def fuzzy_com_entropy(self, similarity_matrix, R, x, verbose=0):
        """
        Oblicza fuzzy combination entropy CE(R).

        :param similarity_matrix: macierze podobieństwa dla poszczególnych cech
        :param R: zbiór cech, zapisany jako lista indeksów
        :param x: próg lambda używany do przejścia z relacji rozmytej do klasycznej
        """
        # MATLAB: n = size(similarity_matrix, 1).
        # W Pythonie similarity_matrix ma układ:
        # (liczba_cech, liczba_obiektów, liczba_obiektów),
        # dlatego liczba obiektów znajduje się w osi 1.
        m_objects = similarity_matrix.shape[1]

        # MATLAB: tmp_matrix = ones(n,n).
        # Tworzymy macierz relacji pełnej, czyli wszędzie ustawiamy 1.
        tmp_matrix = np.ones((m_objects, m_objects))

        # MATLAB: if at_num == 0, fuzzy_com_ent = 1.
        # Dla pustego zbioru cech nie wykonujemy części wspólnej macierzy.
        if len(R) == 0:
            return 1.0

        for feature_index in R:
            # MATLAB: tmp_matrix = min(tmp_matrix, ...).
            # Dla każdej cechy w R wykonujemy przecięcie rozmyte
            # (t-norma minimum) macierzy podobieństwa.
            # Po tej linii:
            # tmp_matrix(i,j) = min_{r in R} similarity_matrix(r,i,j).
            tmp_matrix = np.minimum(
                tmp_matrix, similarity_matrix[feature_index, :, :].squeeze()
            )

        if verbose > 0:
            print("t-norm matrix (minimum from compared pairs):", tmp_matrix)

        # MATLAB: if tmp_matrix(i,j) >= x, tmp_matrix(i,j)=1, else 0.
        # Próg lambda: relację rozmytą zamieniamy na macierz klasyczną.
        tmp_matrix = np.where(tmp_matrix >= x, 1, 0)
        if verbose > 0:
            print("after applying lambda", tmp_matrix)

        fuzzy_com_ent = 0.0
        for object_index in range(m_objects):
            # MATLAB: sum_i = sum(tmp_matrix(i,:)).
            # sum_i jest liczbą obiektów pozostających w tej samej klasie
            # podobieństwa co obiekt object_index.
            sum_i = np.sum(tmp_matrix[object_index, :])
            if sum_i > 0:
                # MATLAB: fuzzy_com_ent +=
                #   (sum_i*(sum_i-1))/(n*(n-1)).
                # To jest wkład jednej klasy ziarnistości do sumy w CE.
                fuzzy_com_ent += (
                    sum_i * (sum_i - 1) / (m_objects * (m_objects - 1))
                )

        # MATLAB: fuzzy_com_ent = 1 - fuzzy_com_ent/n.
        # Ostatecznie:
        # CE(R) = 1 - (1/m) * SUM_i[
        #   n_i(n_i-1)/(m(m-1))
        # ].
        return 1 - fuzzy_com_ent / m_objects

    def fuzzy_com_conditional_entropy(
        self, similarity_matrix, C, D, x, verbose=0
    ):
        # C - zbiór warunkujący, D - zbiór warunkowany.
        # Funkcja zwraca CE(D | C), a nie CE(C | D).
        # MATLAB: Fuzzy_Com_Conditional_Entropy(similarity_matrix,C,D,x)
        #         oblicza Disc(D|C).
        if len(C) == 0:
            # MATLAB: if size(C,1)==0, f_com_con_ent = CE(D).
            # Warunkowanie pustym zbiorem nie zmienia entropii.
            return self.fuzzy_com_entropy(
                similarity_matrix, R=D, x=x, verbose=verbose
            )

        # MATLAB: [C;D] oznacza konkatenację zbiorów cech.
        # Python: list(C) + list(D) realizuje to samo działanie.
        CD = list(C) + list(D)

        # Zgodnie z definicją z artykułu:
        # CE(D|C) = CE(C ∪ D) - CE(C).
        # Pierwsze wywołanie realizuje CE(C ∪ D),
        # drugie wywołanie realizuje CE(C).
        return self.fuzzy_com_entropy(
            similarity_matrix, R=CD, x=x, verbose=verbose
        ) - self.fuzzy_com_entropy(
            similarity_matrix, R=list(C), x=x, verbose=verbose
        )

    def mutual_inf_com(self, similarity_matrix, C, D, x, verbose=0):
        # MATLAB: mutual_com_ent = CE(D) - CE(D|C).
        # W Pythonie pierwsza linia poniżej realizuje CE(D),
        # a drugie wywołanie realizuje CE(D|C).
        #
        # ICE(C;D) = CE(D) - CE(D|C).
        # CE(D|C) = CE(C ∪ D) - CE(C).
        # Zatem ICE(C;D) = CE(D) - CE(C ∪ D) + CE(C).
        # Wzajemna informacja jest przemienna: ICE(C;D) = ICE(D;C).
        return self.fuzzy_com_entropy(
            similarity_matrix, R=list(D), x=x, verbose=verbose
        ) - self.fuzzy_com_conditional_entropy(
            similarity_matrix, C=list(C), D=list(D), x=x, verbose=verbose
        )

    def MGRMLR(
        self,
        similarity_matrix,
        fk,
        D,
        S,
        alfa,
        beta,
        gamma,
        lambd,
        verbose=0,
    ):
        """
        Oblicza MGRMLR dla kandydata fk.

        fk - aktualnie rozpatrywana cecha,
        D - cecha decyzyjna,
        S - cechy wybrane wcześniej.
        """

        # Pomocniczo zapisujemy ICE(F;G) jako wzajemną informację.
        #
        # Wzór podstawowy:
        # ICE(F;G) = CE(G) - CE(G|F).
        #
        # Wywołanie mutual_inf_com(F,G) realizuje dokładnie:
        # self.fuzzy_com_entropy(G)
        # - self.fuzzy_com_conditional_entropy(F,G).
        def ice(F, G):
            return self.mutual_inf_com(
                similarity_matrix,
                list(F),
                list(G),
                x=lambd,
                verbose=verbose,
            )

        # Twierdzenie 8 z artykułu:
        # ICE(F;G|H) = ICE(F ∪ H;G) - ICE(H;G).
        #
        # Po rozpisaniu obu składników:
        # ICE(F ∪ H;G) = CE(G) - CE(G|F ∪ H),
        # ICE(H;G)     = CE(G) - CE(G|H).
        #
        # Po odjęciu:
        # ICE(F;G|H) = CE(G|H) - CE(G|F ∪ H).
        #
        # W kodzie:
        # ice(list(F)+list(H),G) realizuje ICE(F ∪ H;G),
        # ice(H,G)             realizuje ICE(H;G).
        # H pozostaje warunkiem; przemienność dotyczy tylko pierwszych dwóch zbiorów.
        def ice_cond(F, G, H):
            return ice(list(F) + list(H), G) - ice(H, G)

        # D - decision
        # 𝐽𝐺𝑟𝑒𝑙(𝑓𝑘,𝑆,𝐷) = 𝐼𝐶𝐸(𝑓𝑘;𝐷) + SUM 𝐼𝐶𝐸(𝑓𝑠;D|𝑓𝑘)
        #   𝐼𝐶𝐸(fk;D) = 𝐶𝐸(D) − 𝐶𝐸(D|fk) = 𝐶𝐸(fk)−𝐶𝐸(fk|D)
        #   𝐼𝐶𝐸(fs;D|fk)= 𝐼𝐶𝐸(fs;D) − 𝐼𝐶𝐸(D;fk;fs)
        #     𝐼𝐶𝐸(fs;D) = 𝐶𝐸(fs)−𝐶𝐸(fs|D)
        #     𝐼𝐶𝐸(D;fk;fs) = 𝐼𝐶𝐸(fk;D)+ 𝐼𝐶𝐸(fs;fk)−𝐼𝐶𝐸(fk;D∪fs)
        #       𝐼𝐶𝐸(fs;fk) = 𝐶𝐸(fk) − 𝐶𝐸(fk|fs)
        #       𝐼𝐶𝐸(fk;D∪fs) = 𝐶𝐸(fk)−𝐶𝐸(fk|D∪fs)

        # x = 𝐶𝐸(fk)−𝐶𝐸(fk|D)
        # 𝐽𝐺𝑟𝑒𝑙(𝑓𝑘,𝑆,𝐷) = x + SUM (𝐶𝐸(fs)−𝐶𝐸(fs|D)) - (x + (𝐶𝐸(fk) − 𝐶𝐸(fk|fs)) - 𝐶𝐸(fk)−𝐶𝐸(fk|D∪fs))
        #
        # W kodzie pierwsza linia realizuje ICE(fk;D),
        # a każda iteracja pętli dodaje ICE(fs;D|fk).
        jgrel = ice(fk, D)
        for fs in S:
            jgrel += ice_cond([fs], D, fk)

        # 𝐽𝑁𝑟𝑒𝑙(𝑓𝑘,𝑆,𝐷) = SUM 𝐼𝐶𝐸(fk;D|fs)
        #   𝐼𝐶𝐸(fk;D|fs) = 𝐼𝐶𝐸(fk∪fs;D) − 𝐼𝐶𝐸(fs;D)
        #     𝐼𝐶𝐸(fk∪fs;D) = 𝐶𝐸(D) − 𝐶𝐸(D|fk∪fs)
        #     𝐼𝐶𝐸(fs;D) = 𝐶𝐸(D) − 𝐶𝐸(D|fs)
        #   𝐼𝐶𝐸(fk;D|fs) = 𝐶𝐸(D|fs) − 𝐶𝐸(D|fk∪fs)
        #
        # 𝐽𝑁𝑟𝑒𝑙 = SUM [𝐶𝐸(D|fs) − 𝐶𝐸(D|fk∪fs)]
        # Każda iteracja pętli dodaje jeden składnik tej sumy.
        jnrel = 0.0
        for fs in S:
            jnrel += ice_cond(fk, D, [fs])

        # 𝐽𝑅𝑟𝑒𝑙(𝑓𝑘,𝑆,𝐷) = SUM 𝐼𝐶𝐸(D;fs|fk)
        #   𝐼𝐶𝐸(D;fs|fk) = 𝐼𝐶𝐸(D∪fk;fs) − 𝐼𝐶𝐸(fk;fs)
        #     𝐼𝐶𝐸(D∪fk;fs) = 𝐶𝐸(fs) − 𝐶𝐸(fs|D∪fk)
        #     𝐼𝐶𝐸(fk;fs) = 𝐶𝐸(fs) − 𝐶𝐸(fs|fk)
        #   𝐼𝐶𝐸(D;fs|fk) = 𝐶𝐸(fs|fk) − 𝐶𝐸(fs|D∪fk)
        #
        # 𝐽𝑅𝑟𝑒𝑙 = SUM [𝐶𝐸(fs|fk) − 𝐶𝐸(fs|D∪fk)]
        # Każda iteracja pętli dodaje jeden składnik tej sumy.
        jrrel = 0.0
        for fs in S:
            jrrel += ice_cond(D, [fs], fk)

        # 𝐌𝐆𝐑𝐌𝐋𝐑(𝑓𝑘,𝑆,𝐷) = 𝛼𝐽𝐺𝑟𝑒𝑙 + 𝛽𝐽𝑁𝑟𝑒𝑙 + 𝛾𝐽𝑅𝑟𝑒𝑙
        # W kodzie: alfa*JGrel + beta*JNrel + gamma*JRrel.
        return alfa * jgrel + beta * jnrel + gamma * jrrel

    def FSmFCE(
        self,
        X,
        y,
        alfa=0.33,
        beta=0.33,
        gamma=0.33,
        lambd=0.5,
        k=0.001,
        verbose=0,
    ):
        # MATLAB: FDS = (U, C ∪ D, V, f).
        # W Pythonie y dokładamy jako ostatnią cechę macierzy podobieństwa.
        # Ostatni indeks będzie więc indeksem cechy decyzyjnej D.
        X_with_y = np.column_stack((X, y))
        Fmatrix = self.fuzzy_similarity(X_with_y, verbose=verbose)

        # Indeks decyzji jest równy liczbie cech wejściowych.
        # Dla X o wymiarze (liczba_obiektów, liczba_cech) nie jest to
        # Fmatrix.shape[1]-1, bo shape[1] oznacza liczbę obiektów.
        decision_index = X.shape[1]

        # C - cechy warunkowe, bez dołączonej cechy decyzyjnej D.
        # D - cecha decyzyjna, czyli ostatnia cecha w Fmatrix.
        C = set(range(decision_index))
        D = [decision_index]
        S = set()

        # MATLAB: Co_prev = 10; Co = 0.
        # Co oznacza CE(D|S), czyli entropię decyzji warunkowaną
        # aktualnie wybranym zbiorem cech.
        Co_prev = 10.0
        Co = 0.0

        # Algorytm 1 z artykułu:
        # while Co_prev - Co > k
        # Kontynuujemy wybór, dopóki zmniejszenie CE(D|S) jest większe k.
        # Tolerancja służy tylko do stabilnego rozstrzygania remisów
        # wynikających z niedokładności obliczeń zmiennoprzecinkowych.
        tie_tol = 1e-12
        while Co_prev - Co > k:
            best_score = -np.inf
            best_feature = None

            # MATLAB: for every fi in C-S compute MGRMLR(fi).
            # Sprawdzamy każdą jeszcze niewybraną cechę w stałej kolejności.
            for fi in sorted(C - S):
                score = self.MGRMLR(
                    similarity_matrix=Fmatrix,
                    fk=[fi],
                    D=D,
                    S=S,
                    alfa=alfa,
                    beta=beta,
                    gamma=gamma,
                    lambd=lambd,
                    verbose=verbose,
                )

                # MATLAB: a = max{fi | MGRMLR(fi)}.
                # Zapamiętujemy cechę o największej wartości MGRMLR.
                # Przy remisie zachowujemy cechę o niższym indeksie.
                is_better = (
                    best_feature is None
                    or score > best_score + tie_tol
                )
                is_tie = (
                    best_feature is not None
                    and np.isclose(
                        score,
                        best_score,
                        rtol=tie_tol,
                        atol=tie_tol,
                    )
                )
                if is_better or (is_tie and fi < best_feature):
                    best_score = score
                    best_feature = fi

            if best_feature is None:
                break  # wszystkie cechy wejściowe zostały już wybrane

            # MATLAB: Co_prev = Co.
            Co_prev = Co

            # MATLAB: Co = CE(D | S ∪ {a}).
            # C w funkcji warunkowej to zbiór warunkujący S ∪ {a},
            # a D pozostaje zbiorem warunkowanym.
            Co = self.fuzzy_com_conditional_entropy(
                similarity_matrix=Fmatrix,
                C=list(S | {best_feature}),
                D=D,
                x=lambd,
                verbose=verbose,
            )

            # MATLAB: S = S ∪ {a}.
            S = S | {best_feature}

            if verbose:
                print(f"Selected feature: {best_feature}, Co: {Co}")

        # Końcowa redukcja z Algorytmu 1:
        # if CE(D | S - {a}) <= CE(D | S),
        #     S = S - {a}.
        # Przeglądamy indeksy malejąco, aby przy równoważnych cechach
        # najpierw usunąć większy indeks i zachować mniejszy.
        for feature in sorted(S, reverse=True):
            # MATLAB: CE(D | S - {a}).
            # C = S - {a}, D = decyzja.
            ce_without_feature = self.fuzzy_com_conditional_entropy(
                similarity_matrix=Fmatrix,
                C=list(S - {feature}),
                D=D,
                x=lambd,
                verbose=verbose,
            )
            # MATLAB: CE(D | S).
            # C = S, D = decyzja.
            ce_with_feature = self.fuzzy_com_conditional_entropy(
                similarity_matrix=Fmatrix,
                C=list(S),
                D=D,
                x=lambd,
                verbose=verbose,
            )

            # MATLAB: if CE(D | S - {a}) <= CE(D | S), S = S - {a}.
            if ce_without_feature <= ce_with_feature:
                S = S - {feature}

        return S


if __name__ == "__main__":
    X = np.array(
        [
            [0.3, 1.0, 0.4, 0.8, 1],
            [0.4, 0.6, 0.3, 0.9, 2],
            [0.7, 0.0, 1.0, 0.0, 2],
            [1.0, 0.2, 0.1, 0.2, 3],
        ]
    )
    fs_object = FSmFCE()
    Fmatrix = fs_object.fuzzy_similarity(X, verbose=0)
    print(Fmatrix)
    entropy1 = fs_object.fuzzy_com_entropy(Fmatrix, [0], 0.5)  # CE(a1)
    print(entropy1)
    entropy12 = fs_object.fuzzy_com_entropy(Fmatrix, [0, 1], 0.5)  # CE(a1 ∪ a2)
    condition = fs_object.fuzzy_com_conditional_entropy(
        Fmatrix, [0, 1, 2, 3], [4], 0.5
    )  # CE(a5 | a1,a2,a3,a4)
    print(entropy12)
    print(condition)
