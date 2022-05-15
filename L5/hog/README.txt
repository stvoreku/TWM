

# Instrukcja uruchomienia
1. Uruchomić L5_trainSVM.m
2. Uruchomić L5_test.m

W skrypcie testowym można zmieniać parametry bez potrzeby ponownego trenowania SVM.
Na początku skryptu można wybrać przykład wpisując odpowiednią liczbę 1-5 (jedynie przykład 4 z jakiejś przyczyny nie działa).

# Obserwacje wstępne
Kod faworyzuje wybór małych detekcji, więc jeżeli zaczynamy od małych okien to "zjadają" one duże detekcje, co objawia się tym że zamiast zaznaczenia całej osoby mamy w jej miejscu np. 3 małe prostokąty.
W praktyce scale > 1.0 psuje więc detekcje większych osób.
Ze względu na ograniczenia czasowe nie naprawialiśmy tego zachowania i skupiliśmy się na wykrywaniu mniejszego zakresu osób.
Innymi słowy, osoby w głębszej dali nie będą wykrywane.
Z tej samej przyczyny nie ma sensu ustawiać "levels" na większe wartości, ponieważ w każdym kroku oglądamy coraz mniejszy obraz, a więc szukamy coraz większych ludzi - co zostanie odrzucone jak wyjaśniono wyżej.

# Komentarz do implementacji
Aby móc nauczyć SVM HOGami, muszą one być tego samego rozmiaru.
Dlatego też cellSize w liczeniu HOGów jest ustawiany dynamicznie w taki sposób aby każdy obraz dawał HOG o tych samych rozmiarach.
Kiedy jednak ustawiamy dowolny rozmiar siatki powyżej 8, pojawiają się błędy i np. zamiast 12x18 mamy 13x18 lub 12x19.
Próbowaliśmy zmienić zaokrąglanie z funkcji floor na zaokrąglenie do najbliższej liczby całkowitej, jednak poskutkowało to jedynie odwróceniem problemu.
Finalnie więc używamy siatki komórek 8x8.

Ponieważ w oryginalnym kodzie wdrożono minimalizację, zamiast maksymalizować pewność wykrycia osoby minimalizujemy pewność nie-wykrycia osoby.
Wyniki SVM zostały "znormalizowane" (fitSVMPosterior) do 0..1.
Powyższe wyniki zdaję się dopełniać do jedynki, więc ta metoda nie powinna wprowadzać błędów - nie jest to jednak pewne.

# Ograniczenia naszej metody
Nasz SVM używający HOGów mocno skupia się na obszarach których dolna połowa przypomina nogi.
Być może jest to wada użytego zestawu danych lub poniekąd małego rozmiaru HOGów.
Może to też jednak sugerować że sieć splotowa byłaby lepsza do wykrywania ludzi, a SVM nadaje się lepiej do obiektów o relatywnie stałym kształcie.

# Wyniki
Najlepsze wyniki dla przykładu 1 (people_1) uzyskaliśmy dla
scale = 0.8;
True positive: 4/5
False positive: 3

Przykład 2:
scale = 0.5;
tp 4/4
fp 1

Przykład 3:
scale = 0.5;
tp 1/1
fp 0

Przykład 4:
błąd z chodzeniem po obrazie

Nowe przykłady:

Przykład 5:
scale = 0.5;
True positive: w zasadzie 5/5 (dziecko na rowerze jest wykryte dobrze ale nie koloruje się na zielono)
False positive: kilkanaście - trochę nie rozumiemy czemu SVN wykrywa te fragmenty jako osoby

Przykład 6 - usunięty, z jakiś przyczyn jego gTruth błędnie się eksportowało

Gdyby nie wspomniany problem z faworyzowaniem mniejszych detekcji, prawdopodobnie obyłoby się bez manualnego zmieniania początkowej skali obrazu.
Być może zmniejszyłoby to również liczbę false positive, które przeważnie są małymi prostokątami.