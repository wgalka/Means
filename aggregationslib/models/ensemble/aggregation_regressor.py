import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from copy import copy

# Import default aggregation method from aggregationslib.aggregations
from aggregationslib.aggregations import A_ar

class AggregationRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, n_subarrays=5, aggregation_method=None, random_state=None):
        self.base_estimator = base_estimator
        self.n_subarrays = n_subarrays
        self.aggregation_method = aggregation_method if aggregation_method is not None else A_ar()
        self.random_state = random_state
        self.models_ = []
        self.attribute_sets_ = []

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_features = X_arr.shape[1]

        # Pobranie indeksów cech
        feature_indices = np.arange(n_features)
        
        # Jeśli podano random_state, losowo tasujemy indeksy kolumn przed podziałem
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(feature_indices)

        # Podział cech na podtabele
        self.attribute_sets_ = np.array_split(feature_indices, self.n_subarrays)

        # Trenowanie osobnego modelu na każdym podzbiorze cech
        self.models_ = []
        for attr_set in self.attribute_sets_:
            model = copy(self.base_estimator)
            model.fit(X_arr[:, attr_set], y_arr)
            self.models_.append(model)
        
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]

        # Generowanie prognoz cząstkowych z każdego modelu
        subsets_predictions = np.zeros((n_samples, self.n_subarrays))
        for idx, (model, attr_set) in enumerate(zip(self.models_, self.attribute_sets_)):
            subsets_predictions[:, idx] = model.predict(X_arr[:, attr_set])

        # Agregacja prognoz wiersz po wierszu
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = self._aggregate(subsets_predictions[i, :])
        
        return y_pred

    def _aggregate(self, values):
        """Wywołuje przekazany obiekt agregacji na predykcjach cząstkowych."""
        if not callable(self.aggregation_method):
            raise TypeError("aggregation_method must be a callable object.")
        
        # Wymagane rzutowanie wartości na typ float (wymóg niektórych funkcji w aggregationslib)
        return self.aggregation_method(values.astype(float))
