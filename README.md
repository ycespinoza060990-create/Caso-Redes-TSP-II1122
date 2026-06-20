# Camino más corto + TSP en Streamlit

Aplicación desarrollada para el caso de redes de II-1122 Optimización Industrial. La app calcula la matriz Desde-Hasta mediante caminos más cortos, genera archivos AMPL y permite intentar resolver el TSP con formulación MTZ desde Streamlit.

## Archivos incluidos

- `app.py`: aplicación principal en Streamlit.
- `requirements.txt`: librerías necesarias.
- `data/Red1.nf`: red vial.
- `data/Clientes1.txt`: puntos de entrega.
- `ampl/CaminoMasCorto.mod`: modelo AMPL de camino más corto.
- `ampl/TSP.mod`: modelo AMPL del TSP con eliminación de subciclos MTZ.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Flujo de uso

1. Abrir la app.
2. Usar los archivos incluidos o subir la red y clientes.
3. Calcular la matriz Desde-Hasta.
4. Descargar el paquete con `.mod`, `.dat` y matriz.
5. Intentar resolver el TSP en Streamlit o correr formalmente en AMPL.
6. Reportar ruta óptima, distancia total, tiempo, variables y restricciones.

## Nota

La instancia puede ser grande para resolver de forma exacta dentro de una app web. Para la evidencia formal del TSP se recomienda correr `TSP.mod` y `TSP.dat` en AMPL con CPLEX, Gurobi o HiGHS.
