# Całki rozmyte Choqueta i Sugeno

Funkcje `choquet_integral` i `sugeno_integral` przyjmują niepusty wektor
liczb z `[0, 1]` i zwracają jedną wartość typu `float`.

Dla wektora wejściowego $(x_1,\dots,x_n)\in[0,1]^n$ niech $\sigma$
porządkuje indeksy tak, aby $x_{\sigma(1)}\leq\dots\leq x_{\sigma(n)}$.
Definiujemy zbiory poziomicowe $A_i=\{\sigma(i),\dots,\sigma(n)\}$ oraz
$x_{\sigma(0)}=0$. Całki mają postać:

$$C_\mu(x_1,\dots,x_n)=\sum_{i=1}^{n}(x_{\sigma(i)}-x_{\sigma(i-1)})\,\mu(A_i)$$

$$S_\mu(x_1,\dots,x_n)=\max_{1\leq i\leq n}\min(x_{\sigma(i)},\mu(A_i))$$

`mu` musi być miarą rozmytą: `mu(empty) = 0`, `mu(full) = 1`, a dodanie
elementu do zbioru nie może zmniejszyć jego miary.

## Domyślna miara

Jeśli nie podasz miary, obie funkcje użyją miary kardynalności
`mu(A) = |A| / n`:

Przy tej domyślnej mierze całka Choqueta jest równa średniej arytmetycznej:

$$C_\mu(x_1,\dots,x_n)=\sum_{i=1}^{n}(x_{\sigma(i)}-x_{\sigma(i-1)})\frac{n-i+1}{n}
=\frac{1}{n}\sum_{i=1}^{n}x_i$$

Równość jest matematyczna. W wersji `float` wyniki mogą różnić się na
ostatnich bitach z powodu zaokrągleń.

```python
from aggregationslib import A_ch, A_sg
from aggregationslib.aggregation import choquet_integral, sugeno_integral

values = [0.2, 0.7, 0.9]
choquet_value = choquet_integral(values)  # 0.6
sugeno_value = sugeno_integral(values)    # 2 / 3

# Callable aggregation objects; measures can also be set in the constructor.
choquet_object = A_ch()
sugeno_object = A_sg()
choquet_value = choquet_object(values)
sugeno_value = sugeno_object(values)
```

## Wersja `Decimal`

Jeśli potrzebujesz wyniku typu `Decimal`, użyj odpowiedników z modułu
`decimalaggregations`. Całki wykonują obliczenia w standardowym Pythonie,
używając `Decimal` i sortowania list; nie korzystają z NumPy ani obliczeń
zmiennoprzecinkowych. Ten sam moduł udostępnia funkcje i obiekty callable:

```python
from decimal import Decimal as D
from aggregationslib.decimalaggregations import (
    A_ch, A_sg, DecimalFuzzyMeasure, choquet_integral, sugeno_integral,
)

values = [D("0.2"), D("0.7"), D("0.9")]
choquet_value = choquet_integral(values)  # Decimal("0.6")
sugeno_value = sugeno_integral(values)    # Decimal z bieżącą precyzją kontekstu

choquet_object = A_ch()
sugeno_object = A_sg()
choquet_object(values)
sugeno_object(values)
```

Domyślnie przekazuj wartości wektora i miary jako `Decimal`, całkowite albo
tekstowe zapisy dziesiętne, na przykład `D("0.1")`. Wejściowe `float` są
domyślnie odrzucane. Parametr `allow_float=True` włącza ich konwersję przez
`Decimal(str(value))`:

```python
choquet_integral([0.1, 0.7], allow_float=True)
A_ch(allow_float=True)([0.1, 0.7])

# Parametr obejmuje też wartości miary.
sugeno_integral([0.2, 0.8], measure=[0.0, 0.25, 1.0], allow_float=True)
```

Konwersja zachowuje dziesiętny zapis `float`, ale nie odtwarza precyzji,
która mogła zostać utracona wcześniej. Wynik całki ma typ `Decimal`; jego
precyzję dla dzielenia wyznacza bieżący kontekst modułu `decimal`.

Do miary niesymetrycznej użyj `DecimalFuzzyMeasure` i wartości `Decimal` dla
każdego podzbioru.

Dla miary symetrycznej można przekazać listę długości `n + 1`, zawierającą
wartości `mu` dla zbiorów o rozmiarach od `0` do `n`:

```python
choquet_integral(values, measure=[D("0"), D("0.2"), D("0.6"), D("1")])

# The same measure can be stored in callable aggregation objects.
measure = [D("0"), D("0.2"), D("0.6"), D("1")]
choquet_object = A_ch(measure=measure)
sugeno_object = A_sg(measure=measure)
choquet_object(values)
sugeno_object(values)
```

Wektor miary musi zaczynać się od `0`, kończyć na `1` i być niemalejący.

## Miara niesymetryczna

Aby różne podzbiory cech miały różną wagę, przekaż pełną miarę jako mapę.
Klucze to `frozenset` z indeksami pozycji wektora, numerowanymi od zera. Dla
trzech cech mapa musi zawierać miarę wszystkich ośmiu podzbiorów:

```python
from aggregationslib.aggregation import FuzzyMeasure, choquet_integral

measure = FuzzyMeasure(3, {
    frozenset(): 0.0,
    frozenset({0}): 0.2,
    frozenset({1}): 0.3,
    frozenset({2}): 0.4,
    frozenset({0, 1}): 0.6,
    frozenset({0, 2}): 0.7,
    frozenset({1, 2}): 0.8,
    frozenset({0, 1, 2}): 1.0,
})

result = choquet_integral([0.2, 0.7, 0.9], measure=measure)
```

Konstruktor sprawdza kompletność, normalizację i monotoniczność miary.

## Walidacja wejścia

Wektor musi być jednowymiarowy, niepusty, rzeczywisty, skończony i zawarty
w `[0, 1]`. Niepoprawny wektor lub miara powoduje `ValueError` albo
`TypeError` z opisem problemu.

## Przypadki brzegowe i testy

Z definicji całki Choqueta jako sumy poziomic oraz własności całek rozmytych
wynikają testy: wektor jednoelementowy, same zera, same jedynki, wartości
powtórzone (w tym remisy podczas sortowania), wektory 0/1 oraz miary
niesymetryczne. Dla wektora wskaźnikowego `1_A` obie całki powinny zwrócić
`mu(A)`. Dla miary znormalizowanej wynik powinien też leżeć pomiędzy minimum
a maksimum wektora; zwiększenie dowolnych składowych nie może obniżyć wyniku.
Zestaw testów skupia się na tych własnościach, niezmienniczości przy
jednoczesnym przestawieniu cech i miary oraz porównaniu z 100 rekordami
obliczonymi w Wolfram Language.

W tym pakiecie obie całki mają kontrakt agregacji na `[0, 1]^n`; choć
literatura opisuje także rozszerzenia całki Choqueta dla wartości ujemnych,
nie należą one do tego interfejsu. Sugeno wymaga zakresu `[0, 1]` oraz
znormalizowanej miary.

Wzorcowe wyniki 100 przypadków, wcześniej obliczone w Wolfram Language,
są zapisane w `tests/data/wolfram_integral_reference.json`. Testy porównują
z nimi obie implementacje (`float` i `Decimal`), zarówno funkcje, jak i obiekty
`A_ch` oraz `A_sg`. Testy odczytują gotowy plik JSON — nie uruchamiają Wolframa
i nie wymagają jego instalacji.

### Ręczne obliczenie przykładu

Dla wektora $(0.7,0.2,0.9)$ i miary z poprzedniego przykładu sortowanie daje
wartości $(0.2,0.7,0.9)$ oraz zbiory poziomicowe
$A_1=\{0,1,2\}$, $A_2=\{0,2\}$, $A_3=\{2\}$. Ich miary wynoszą odpowiednio
$1$, $0.7$ i $0.4$. Stąd:

$$
C_\mu(0.7,0.2,0.9)=(0.2-0)\cdot1+(0.7-0.2)\cdot0.7+(0.9-0.7)\cdot0.4=0.63.
$$

$$
S_\mu(0.7,0.2,0.9)=\max\{\min(0.2,1),\min(0.7,0.7),\min(0.9,0.4)\}=0.7.
$$

## Źródła

- G. Beliakov, H. Bustince Sola, T. Calvo Sánchez, [Fuzzy Integrals](https://doi.org/10.1007/978-3-319-24753-3_4), w: *A Practical Guide to Averaging Functions*, Studies in Fuzziness and Soft Computing, vol. 329, Springer, 2016, s. 145–181. Rozdział przedstawia całki Choqueta i Sugeno oraz własności miar rozmytych.
- T. Murofushi, M. Sugeno, [Fuzzy Measures and Fuzzy Integrals](https://ikojadin.perso.univ-pau.fr/kappalab/pub/MurSugCOLL2000.pdf), 2000. Zawiera definicje skończonej całki Choqueta i Sugeno oraz własności: monotoniczność, zachowanie wartości miary na wektorach wskaźnikowych i ograniczenie wyniku przez minimum i maksimum.
- J.-L. Marichal, [On Sugeno Integral as an Aggregation Function](https://ikojadin.perso.univ-pau.fr/kappalab/pub/MarFSS2000.pdf), 2000. Podaje równoważną postać Sugeno typu max–min opartą na statystykach pozycyjnych.
- M. Sugeno, [Fuzzy Measure and Fuzzy Integral](https://doi.org/10.9746/sicetr1965.8.218), 1972. Oryginalna praca o miarze rozmytej i całce Sugeno.
