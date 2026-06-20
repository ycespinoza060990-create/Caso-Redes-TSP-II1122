# Caso 2 · TSP con matriz 13x13

Aplicación Streamlit para el Caso 2. Trabaja con la matriz de distancias completa de 13 puntos: depósito 0 + 12 clientes.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Incluye

- Resolución exacta del TSP con SciPy/HiGHS usando formulación MTZ.
- Modelo AMPL `TSP_13.mod`.
- Datos AMPL `TSP_13.dat`.
- Matriz de distancias `matriz_13.csv`.
- Respuestas guía a las preguntas de las partes I, II y III.
