---
layout: default
title: "Means - Aplikacja Shiny"
lang: pl
---
# Aplikacja Shiny

Graficzny interfejs do obliczania funkcji agregacji na danych tabelarycznych oraz oceny klasyfikacji binarnej.

*English version: [English](https://wgalka.github.io/Means/en/shiny/)*

## Uruchomienie

Zainstaluj bibliotekę z katalogu głównego repozytorium:

```powershell
python -m pip install .
```

Uruchom aplikację poleceniem:

```powershell
agglib start
```

Jeśli w środowisku brakuje pakietów wymaganych przez interfejs (`shiny`, `pandas`, `matplotlib`, `scikit-learn`, `pygments`), CLI zapyta o ich doinstalowanie:

```text
Install the graphical interface? [Y/n]
```

Możesz też zainstalować wszystkie zależności od razu:

```powershell
python -m pip install ".[app]"
```

Polecenie `agglib serve` działa analogicznie do `agglib start`. Serwer zatrzymuje się skrótem `Ctrl+C`.

> **Tryb testowy:** Dopisanie `?tester=1` do adresu w przeglądarce (np. `http://127.0.0.1:8000/?tester=1`) automatycznie wczytuje przykładowy zbiór `example.csv` wraz z wektorem decyzji.

## Praca z danymi

1. **Wczytanie pliku** – wgraj plik CSV i dobierz separator kolumn oraz separator dziesiętny. Każdy wiersz traktowany jest jako osobny wektor do agregacji.
2. **Wybór kolumn** – w podglądzie danych zaznacz checkboxy w nagłówkach kolumn, które mają wejść do obliczeń (kolumny tekstowe, np. identyfikatory, należy odznaczyć). Kliknięcie samej nazwy kolumny sortuje podgląd tabeli.

Podgląd domyślnie pokazuje 5 pierwszych i 5 ostatnich wierszy (`5 … 5`), z możliwością przełączenia na pełną tabelę (`All rows`, ograniczoną w widoku przeglądarki do 2 000 wierszy). Sortowanie i skracanie widoku nie wpływa na kolejność wierszy w obliczeniach ani w eksporcie.

Wgrywane pliki CSV są weryfikowane jeszcze przed wczytaniem do pamięci (limit 20 MB oraz 100 000 wierszy), a przed dodaniem zadania do kolejki aplikacja szacuje zapotrzebowanie na pamięć i złożoność obliczeniową wybranej metody. Obliczenia wykonują się asynchronicznie w ograniczonej kolejce zadań z paskiem postępu.

## Agregacje i wagi

Dostępnych jest 21 funkcji agregacji z modułu `aggregationslib.ndarray`, w tym całki rozmyte Choqueta (`A_ch`) i Sugeno (`A_sg`) z sześcioma rodzajami miar:

- **weighted** oraz **k-fuzzy** – pola na wagi pojawiają się bezpośrednio nad zaznaczonymi kolumnami (dla miary ważonej wagi muszą sumować się do 1).
- **owa** – wagi przypisywane są kolejnym pozycjom po uporządkowaniu wartości od największej (suma wag równa 1).
- **exponential_cardinality** – przyjmuje pojedynczy parametr $q > 0$.
- **uniform** oraz **relative** – miary niewymagające dodatkowych wag.

Ze względu na generowanie podzbiorów w konstruktorach miar, dla wariantów `weighted`, `owa` i `k-fuzzy` w interfejsie obowiązuje limit 12 zaznaczonych kolumn. Dane wejściowe nie są automatycznie normalizowane, a braki danych (`NaN`) lub wartości nieskończone w wybranych kolumnach zostaną zgłoszone jako błąd.

Na dole formularza można pobrać wynikowy plik CSV (zawierający oryginalne kolumny powiększone o kolumnę z wynikiem) oraz gotowy, samodzielny kod w Pythonie odtwarzający obliczenia.

## Ocena klasyfikacji

W sekcji *Ground truth labels* można podać rzeczywiste etykiety klas (wklejając je jako tekst lub wgrywając z pliku CSV), aby ocenić jakość klasyfikacji opartej na wynikach agregacji:

- Etykiety muszą odpowiadać kolejnym wierszom danych (wliczając dopisane ręcznie wektory) i zawierać dokładnie dwie klasy.
- Należy wskazać etykietę klasy dodatniej (domyślnie `1`) oraz określić, czy wyższy wynik agregacji oznacza klasę dodatnią, czy ujemną.

Po podaniu etykiet aplikacja wyznacza:
- **Krzywą ROC** wraz z polem pod krzywą (ROC AUC),
- **Optymalny próg odcięcia ($t$)** według wybranego kryterium: indeksu Youdena ($J = \text{TPR} - \text{FPR}$), dokładności (Accuracy) lub miary F1. Próg można też ustawić ręcznie w trybie *Manual*,
- **Zestaw metryk**: Average Precision, Accuracy, Balanced Accuracy, Precision, Recall, Specificity, NPV, F1 (dla klasy dodatniej oraz warianty micro/macro/weighted), MCC i współczynnik $\kappa$ Cohena (nazwy metryk w tabeli linkują do dokumentacji `scikit-learn`),
- **Macierze pomyłek** w postaci map ciepła (liczności oraz udziały procentowe w klasach rzeczywistych).
