# CaminoMasCorto.mod
# Modelo de camino más corto entre un origen s y un destino t

set NODES;
set ARCS within {NODES, NODES};

param c{ARCS} >= 0;
param s symbolic in NODES;
param t symbolic in NODES;

var x{ARCS} binary;

minimize Distancia_Total:
    sum{(i,j) in ARCS} c[i,j] * x[i,j];

subject to Flujo{k in NODES}:
    sum{(k,j) in ARCS} x[k,j] - sum{(i,k) in ARCS} x[i,k] =
        if k = s then 1 else if k = t then -1 else 0;

solve;

display Distancia_Total;
display x;
