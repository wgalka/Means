# Klasy funkcji agregujących

## Funkcje agregujące na `[0,1]`

```mermaid
flowchart TD
    A[Wszystkie funkcje agregujące] --> C[Conjunctive<br/>A <= min]
    A --> V[Averaging / internal<br/>min <= A <= max]
    A --> D[Disjunctive<br/>A >= max]
    A --> H[Hybrid / mixed<br/>pozostałe funkcje]

    C --> C1[Rozmyte AND<br/>min, iloczyn, t-normy]
    V --> V1[Średnie i operatory kompensacyjne<br/>arytmetyczna, geometryczna, OWA]
    D --> D1[Rozmyte OR<br/>max, suma probabilistyczna, t-konormy]
    H --> H1[Uninormy i nullnormy]
```

Interpretacja klas:

- `conjunctive`: wynik nie przekracza minimum argumentów;
- `averaging`: wynik leży między minimum i maksimum;
- `disjunctive`: wynik jest co najmniej tak duży jak maksimum argumentów;
- `hybrid/mixed`: funkcja nie należy do żadnej z trzech pozostałych klas.

## Operatory dla wartości przedziałowych

```mermaid
flowchart TD
    A[Interval-valued aggregation functions] --> P[Klasyczny porządek częściowy]
    A --> Q[Pos-aggregation<br/>possible]
    A --> N[Nec-aggregation<br/>necessary]
    A --> L[Dopuszczalne porządki liniowe]
    A --> O[IVOWA]

    P --> R[Representable]
    P --> PM[Pseudomax-representable]
    P --> PN[Pseudomin-representable]

    L --> L1[Lexicographic]
    L --> L2[XY]
    L --> L3[Inne porządki dopuszczalne]
```

Diagram przedstawia główne rodziny wymienione w rozdziale dotyczącym
agregacji dla wartości przedziałowych. Nie jest to hierarchia zawierania się
wszystkich klas: część rodzin może się przecinać.

## Powierzchnie 3D

![Powierzchnie 3D wszystkich funkcji agregujących](assets/aggregation-surfaces-all.png)

Plansza pokazuje, jak różne funkcje przekształcają parę wartości `(x,y)` w
pojedynczy wynik `A(x,y)`. Można ją ponownie wygenerować skryptem
`generate_aggregation_plots.py`.
