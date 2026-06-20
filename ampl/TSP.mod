# TSP.mod
# Formulación TSP con eliminación de subciclos MTZ

set NODES;
param n := card(NODES);
param d{NODES, NODES} >= 0;
param depot symbolic in NODES default 0;

var x{i in NODES, j in NODES: i <> j} binary;
var u{i in NODES: i <> depot} >= 1 <= n-1;

minimize Distancia_Total:
    sum{i in NODES, j in NODES: i <> j} d[i,j] * x[i,j];

subject to Salida_Unica{i in NODES}:
    sum{j in NODES: j <> i} x[i,j] = 1;

subject to Entrada_Unica{j in NODES}:
    sum{i in NODES: i <> j} x[i,j] = 1;

subject to Eliminar_Subciclos{i in NODES, j in NODES: i <> j and i <> depot and j <> depot}:
    u[i] - u[j] + n * x[i,j] <= n - 1;

solve;

display Distancia_Total;
display x;
display u;
