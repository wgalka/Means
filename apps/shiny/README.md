# Aplikacja Shiny

Interfejs jest formularzem pośrodku strony, bez panelu bocznego.
Kolejne sekcje prowadzą przez dane wejściowe, opcjonalne decyzje,
wybór agregacji, wybór kolumn i wag, wyniki oraz ocenę klasyfikacji.
Dodatkowe wektory są w rozwijanej sekcji przy danych wejściowych;
ocena z ROC i progiem pojawia się po wybraniu źródła decyzji.

Zainstaluj bibliotekę z katalogu głównego repozytorium.

```powershell
python -m pip install .
```

Po instalacji uruchom nakładkę z dowolnego katalogu w tym samym środowisku Pythona.

```powershell
agglib start
```

Jeśli brakuje zależności nakładki, `agglib start` wyświetli pytanie
`Install the graphical interface? [Y/n]`.
Naciśnij Enter lub wpisz `y`, aby zainstalować zależności i otworzyć aplikację
w przeglądarce. Wpisz `n`, aby zakończyć.

`agglib serve` działa tak samo. Serwer zatrzymasz w terminalu skrótem `Ctrl+C`.
Instalacja z `".[app]"` pozwala przygotować zależności nakładki z wyprzedzeniem.

Wgraj CSV, ustaw separator i klikaj checkboxy bezpośrednio w nagłówkach kolumn.
Kliknięcie nazwy kolumny sortuje tabelę, a checkbox zmienia udział w agregacji.
Domyślnie wszystkie kolumny są zaznaczone; kolumny tekstowe (np. identyfikatory)
należy odznaczyć. Dostępne są też przyciski zaznaczenia i odznaczenia wszystkich.
Kolejność wybranych kolumn jest zgodna z kolejnością w pliku.
Bez pliku aplikacja czeka na wgranie danych.
Ukryty tryb testera można uruchomić przez dopisanie `?tester=1` do adresu
aplikacji, np. `http://127.0.0.1:8000/?tester=1` (użyj portu swojego serwera).
Ładuje on tabelę `example.csv` z 20 wierszami i pasujący wektor decyzji,
aby można było od razu sprawdzić agregacje, ROC i metryki. Nie ma przełącznika
testera ani przycisków pobierania przykładów w interfejsie.
Każdy wiersz wybranych kolumn jest osobnym
wektorem. Dodatkowe wektory wpisuj po jednym na linię; liczby rozdzielaj
spacjami i używaj kropki dziesiętnej. Są dopisywane jako nowe wiersze,
z pustymi komórkami w pozostałych kolumnach.

Wybierz agregację i parametry. Wyniki oraz kod aktualizują się automatycznie.
Podgląd kodu ma kolorowanie składni Pythona przez Pygments, bez zewnętrznych usług.
Tabele korzystają z natywnego DataGrid Shiny. Domyślnie pokazują pierwszych
5 i ostatnich 5 wierszy, a pomiędzy nimi wiersz „…” (dla tabel do 10 wierszy
wyświetlane są wszystkie). W prawym górnym rogu każdej tabeli znajduje się
wybór widoku „5 … 5” / „All rows”, z podświetloną aktywną opcją.
Widoki tabel działają niezależnie; pełny widok korzysta z wirtualnego przewijania.
Kliknięcie nagłówka kolumny sortuje pełne dane
na serwerze, a dopiero potem skraca podgląd: „…” zawsze pozostaje pośrodku.
Kolejne kliknięcia zmieniają kierunek sortowania; strzałka wskazuje kierunek.
Tabele wejściowa i wynikowa mają niezależne sortowanie. Numer wiersza
pozwala odnaleźć pierwotną pozycję po sortowaniu. Sortowanie podglądu nie
zmienia kolejności obliczeń ani eksportu. Obliczenia i eksport obejmują wszystkie wiersze.
CSV zachowuje kolumny źródłowe i dodaje wynik; kod można skopiować i uruchomić
bez Shiny. W kodzie `input.csv` oznacza wgrany plik. Eksport używa wybranego
separatora pól i separatora dziesiętnego.

Dostępne są wszystkie 20 agregacji obiektowych z modułu `ndarray`, w tym
Choquet i Sugeno z sześcioma rodzajami miar. Wagi odpowiadają kolejności
wybranych kolumn; dla OWA odpowiadają wartościom od największej.
Po wybraniu miary weighted, OWA lub k-fuzzy pola wag pojawiają się nad
nazwami zaznaczonych kolumn. Weighted i k-fuzzy przypisują wagę do konkretnej
kolumny; OWA oznacza pola numerami pozycji od największej wartości.
Odznaczenie kolumny ukrywa jej pole. Weighted i OWA wymagają sumy wag 1;
k-fuzzy przyjmuje wagi singletonów bez takiego wymagania.
Exponential cardinality wymaga jednego parametru q>0 zamiast wektora wag.
Wprowadzanie własnych funkcji `quasi_overlap` i składanie dwóch agregacji
nie jest częścią formularza.

Braki danych i nieskończone liczby w wybranych kolumnach wymagają poprawienia.
Aplikacja zgłasza błędy obliczeń i odrzuca wyniki zespolone lub nieskończone.
Nie normalizuje automatycznie danych ani wag. Miary weighted, OWA i k-fuzzy
mają w interfejsie limit 12 kolumn ze względu na enumerację podzbiorów
w istniejących konstruktorach biblioteki.

Pliki są przetwarzane lokalnie przez uruchomiony serwer Shiny.

## Decyzje i ocena klasyfikacji

W sekcji „Ground truth labels (optional)” można wkleić wektor etykiet
(spacje, przecinki, średniki lub nowe linie) albo wgrać CSV, ustawić jego separator,
nagłówek i wybrać kolumnę decyzji. Etykiety muszą zawierać dwie klasy.
Wskaż etykietę klasy dodatniej (domyślnie `1`) i kierunek wyniku agregacji.
Każda decyzja odpowiada wierszowi w pierwotnej kolejności danych, włącznie
z dopisanymi wektorami. Aplikacja sprawdza zgodność liczby decyzji i wyników.
Ukryty tryb testera automatycznie wpisuje przykładowe decyzje do pola wektora.

Raport pokazuje ROC i ROC AUC, AP, ACC, balanced ACC, precision, recall,
specificity, NPV, F1 klasy dodatniej, F1 micro/macro/weighted, MCC i Cohen kappa.
Nazwa każdej miary w raporcie jest linkiem do dokumentacji odpowiedniej funkcji
scikit-learn, gdzie znajduje się też odnośnik `[source]` do implementacji.
Podpowiedź przy linku wskazuje wariant uśredniania lub związek z funkcją
(np. specificity odpowiada recall klasy ujemnej).
Macierze pomyłek są mapami ciepła: liczności oraz udziały w klasach rzeczywistych.
Komórki mają oznaczenia TN/FP/FN/TP i wartości liczbowe (procenty dla udziałów).
Oś pionowa oznacza klasę rzeczywistą, a pozioma przewidzianą.
Obie mapy reagują na zmianę progu; skala udziałów pozostaje stała od 0 do 100%.
Predykcje i wyniki można pobrać jako CSV. Kod pod spodem zawiera też obliczenia
progu, ocenę oraz wykres ROC; etykiety są w nim zapisane jako wektor binarny.

Próg jest optymalny według wybranego kryterium: Youden J (równoważny maksimum
balanced accuracy), accuracy lub F1 klasy dodatniej. Rozpatrywane są wszystkie
progi ROC, bez pomijania punktów pośrednich, również wariant wszystkich decyzji
ujemnych (próg nieskończony). Remisy rozstrzyga mniej predykcji dodatnich.
ROC AUC ocenia ranking wyników i nie zależy od wybranego progu.
Indeks Youdena to `J = TPR - FPR`; próg `t` jest odrębną wartością, dla której
to kryterium osiąga maksimum. Raport podaje aktualne J i t. Tryb „Manual”
pozwala edytować t (początkowo ustawione na próg automatyczny); metryki,
punkt na ROC, macierze, eksport i kod reagują na tę zmianę.
Przycisk „Reset to Youden threshold” wraca do automatycznej maksymalizacji J.
Próg oraz metryki są wyznaczane na tych samych danych; do oceny generalizacji
potrzebne są osobne dane do doboru progu i do końcowego testowania.

Definicje API: [scikit-learn ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).
