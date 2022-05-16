# Pierwsze polecenie zmienia *.ppm w PNG o nazwach *.ppm.png
# Drugie usuwa wszystkie *.ppm
# find ./test -name *.ppm -type f -exec convert {} {}.png \;
# find ./test -name *.ppm -type f -exec rm {} \;
