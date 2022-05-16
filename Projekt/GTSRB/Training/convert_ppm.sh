# Pierwsze polecenie zmienia *.ppm w PNG o nazwach *.ppm.png
# Drugie usuwa wszystkie *.ppm
# find . -name *.ppm -type f -exec convert {} {}.png \;
find . -name *.ppm -type f -exec rm {} \;
