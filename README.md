# Camino más corto + TSP en Streamlit

Aplicación para el caso de redes: calcula la matriz Desde-Hasta mediante caminos más cortos, genera archivos AMPL y permite intentar resolver el TSP con formulación MTZ desde la app.

## Archivos incluidos

- `app.py`: aplicación principal en Streamlit.
- `requirements.txt`: librerías necesarias.
- `data/Red1.nf`: red vial.
- `data/Clientes1.txt`: clientes.
- `ampl/CaminoMasCorto.mod`: modelo AMPL de camino más corto.
- `ampl/TSP.mod`: modelo AMPL del TSP con MTZ.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Flujo de uso

1. Abrir la app.
2. Usar los archivos incluidos o subir la red y clientes.
3. Calcular la matriz Desde-Hasta.
4. Descargar el ZIP con `.mod`, `.dat` y matriz.
5. Resolver formalmente en AMPL y guardar evidencia.
6. Reportar ruta óptima, distancia total, tiempo, variables y restricciones.

## Nota importante

La app intenta resolver el TSP con SciPy MILP, pero para la entrega formal se recomienda correr `TSP.mod` y `TSP.dat` en AMPL con un solver como CPLEX, Gurobi o HiGHS.
