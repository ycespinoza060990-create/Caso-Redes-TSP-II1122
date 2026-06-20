import io
import time
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="Caso 2 · TSP", page_icon="🚚", layout="wide")

CSS = """
<style>
    .main {background-color: #ffffff;}
    h1, h2, h3 {color: #2f3a4f;}
    .subtitle {font-size: 18px; color: #64748b; font-style: italic; margin-bottom: 20px;}
    .card {background:#f3f6fb; border:1px solid #d5dde8; border-radius:14px; padding:22px; box-shadow:0 2px 8px rgba(15,23,42,.06); min-height:110px;}
    .answer {background:#ffffff; border-left:6px solid #242c66; border-radius:10px; padding:18px; margin:12px 0; box-shadow:0 1px 5px rgba(15,23,42,.08);}
    .pill {display:inline-block; background:#242c66; color:white; border-radius:10px; padding:8px 14px; font-weight:700;}
    .yellow {background:#f2aa00; color:#14213d;}
    .route {background:#eef6ff; border:1px solid #bfdbfe; border-radius:12px; padding:18px; font-size:18px; line-height:1.8;}
    code {font-size: 15px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

NODES = [0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77]
D = np.array([
[0,29,18,16,24,32,17,28,29,27,27,33,34],
[29,0,38,40,29,48,15,57,27,53,55,24,44],
[18,38,0,31,42,49,22,27,21,19,36,50,16],
[16,40,31,0,21,18,32,25,45,30,16,34,47],
[24,29,42,21,0,20,30,46,46,48,36,14,56],
[32,48,49,18,20,0,45,40,60,47,25,33,65],
[17,15,22,32,30,45,0,44,16,39,45,32,30],
[28,57,27,25,46,40,44,0,48,11,17,59,41],
[29,27,21,45,46,60,16,48,0,40,54,48,18],
[27,53,19,30,48,47,39,11,40,0,26,60,30],
[27,55,36,16,36,25,45,17,54,26,0,50,52],
[33,24,50,34,14,33,32,59,48,60,50,0,62],
[34,44,16,47,56,65,30,41,18,30,52,62,0]
], dtype=float)

@st.cache_data
def solve_tsp(dist):
    start = time.time()
    n = len(dist)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    mx = len(arcs)
    mu = n - 1
    total_vars = mx + mu

    c = np.zeros(total_vars)
    for k, (i, j) in enumerate(arcs):
        c[k] = dist[i, j]

    constraints = []
    lb = []
    ub = []

    for i in range(n):
        row = np.zeros(total_vars)
        for k, (a, b) in enumerate(arcs):
            if a == i:
                row[k] = 1
        constraints.append(row); lb.append(1); ub.append(1)

    for j in range(n):
        row = np.zeros(total_vars)
        for k, (a, b) in enumerate(arcs):
            if b == j:
                row[k] = 1
        constraints.append(row); lb.append(1); ub.append(1)

    arc_index = {arc: k for k, arc in enumerate(arcs)}
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            row = np.zeros(total_vars)
            row[mx + i - 1] = 1
            row[mx + j - 1] = -1
            row[arc_index[(i, j)]] = n
            constraints.append(row); lb.append(-np.inf); ub.append(n - 1)

    integrality = np.ones(total_vars)
    lb_var = np.zeros(total_vars)
    ub_var = np.ones(total_vars)
    for i in range(1, n):
        idx = mx + i - 1
        lb_var[idx] = 1
        ub_var[idx] = n - 1
        integrality[idx] = 0

    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb_var, ub_var),
        constraints=LinearConstraint(np.vstack(constraints), lb, ub),
        options={"time_limit": 300, "mip_rel_gap": 0}
    )

    elapsed = time.time() - start
    if not res.success:
        return {"success": False, "message": res.message, "time": elapsed}

    x = res.x[:mx]
    selected_idx = [(i, j) for k, (i, j) in enumerate(arcs) if x[k] > 0.5]
    succ = {i: j for i, j in selected_idx}
    route_idx = [0]
    current = 0
    while True:
        current = succ[current]
        route_idx.append(current)
        if current == 0:
            break
        if len(route_idx) > n + 1:
            break

    route_nodes = [NODES[i] for i in route_idx]
    selected_arcs = [(NODES[i], NODES[j], int(dist[i, j])) for i, j in selected_idx]

    return {
        "success": True,
        "objective": float(res.fun),
        "route": route_nodes,
        "selected_arcs": selected_arcs,
        "time": elapsed,
        "variables": total_vars,
        "binary_variables": mx,
        "mtz_variables": mu,
        "constraints": len(constraints),
        "out_constraints": n,
        "in_constraints": n,
        "mtz_constraints": (n - 1) * (n - 2),
        "message": res.message,
    }

def make_dat():
    lines = []
    lines.append("set NODES := " + " ".join(map(str, NODES)) + ";")
    lines.append("param depot := 0;")
    lines.append("param d : " + " ".join(map(str, NODES)) + " :=")
    for node, row in zip(NODES, D.astype(int)):
        lines.append(str(node) + " " + " ".join(map(str, row)))
    lines.append(";")
    return "\n".join(lines)

def make_mod():
    return """set NODES;
param n := card(NODES);
param d {NODES, NODES} >= 0;
param depot symbolic in NODES default 0;

var x {i in NODES, j in NODES: i <> j} binary;
var u {i in NODES: i <> depot} >= 1 <= n-1;

minimize Distancia_Total:
    sum {i in NODES, j in NODES: i <> j} d[i,j] * x[i,j];

subject to Salida_Unica {i in NODES}:
    sum {j in NODES: j <> i} x[i,j] = 1;

subject to Entrada_Unica {j in NODES}:
    sum {i in NODES: i <> j} x[i,j] = 1;

subject to Eliminar_Subciclos {i in NODES, j in NODES: i <> j and i <> depot and j <> depot}:
    u[i] - u[j] + n * x[i,j] <= n - 1;

solve;
display Distancia_Total;
display x;
display u;
"""

def package_zip(result):
    mem = io.BytesIO()
    df = pd.DataFrame(D.astype(int), index=NODES, columns=NODES)
    route_text = " → ".join(map(str, result["route"])) if result.get("success") else "Sin solución"
    arcs_df = pd.DataFrame(result.get("selected_arcs", []), columns=["origen", "destino", "distancia"])
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data/matriz_13.csv", df.to_csv(index=True))
        z.writestr("ampl/TSP_13.mod", make_mod())
        z.writestr("ampl/TSP_13.dat", make_dat())
        z.writestr("resultados/ruta_optima.txt", route_text)
        z.writestr("resultados/arcos_seleccionados.csv", arcs_df.to_csv(index=False))
        z.writestr("resultados/resumen.txt", f"Distancia total: {result.get('objective')}\nTiempo: {result.get('time'):.4f} segundos\nVariables: {result.get('variables')}\nRestricciones: {result.get('constraints')}\n")
    mem.seek(0)
    return mem

st.title("Caso 2 · Camino más corto + TSP")
st.markdown('<div class="subtitle">Instancia reducida de 13 puntos: depósito 0 + 12 clientes</div>', unsafe_allow_html=True)

tabs = st.tabs(["📌 Enfoque", "📊 Matriz", "🧮 Formulación", "🚚 Resolver TSP", "📝 Respuestas", "📦 Entregables"])

with tabs[0]:
    st.header("Caso 2 · enfoque por etapas")
    c1, c2 = st.columns([1, 1.35])
    with c1:
        st.markdown('<div class="card"><h3>El problema</h3><p>Una empresa sale del depósito 0, visita 12 clientes y regresa al depósito, recorriendo la menor distancia total.</p><p><b>13 puntos relevantes:</b><br>0 · 3 · 10 · 21 · 22 · 35 · 40 · 41 · 47 · 65 · 71 · 76 · 77</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>Flujo en dos etapas</h3><p><b>Red vial original</b> → caminos más cortos → <b>Matriz Desde-Hasta 13×13</b> → TSP exacto → ruta óptima.</p><p>La matriz ya resume la distancia mínima entre cada par de puntos, por eso el TSP se resuelve sobre esta red reducida.</p></div>', unsafe_allow_html=True)

with tabs[1]:
    st.header("Matriz de distancias completa")
    df = pd.DataFrame(D.astype(int), index=NODES, columns=NODES)
    st.dataframe(df, use_container_width=True)
    st.download_button("Descargar matriz CSV", df.to_csv(index=True).encode("utf-8"), "matriz_13.csv", "text/csv")

with tabs[2]:
    st.header("Formulación matemática del TSP")
    st.markdown(r"""
**Función objetivo:** minimizar la distancia total recorrida.

$$\min \sum_{i \in N}\sum_{j \in N, j \ne i} d_{ij}x_{ij}$$

**Salida única:** cada punto debe tener exactamente una salida.

$$\sum_{j \in N, j \ne i} x_{ij}=1 \quad \forall i \in N$$

**Entrada única:** cada punto debe tener exactamente una entrada.

$$\sum_{i \in N, i \ne j} x_{ij}=1 \quad \forall j \in N$$

**Eliminación de subciclos MTZ:** evita ciclos separados que no incluyan todos los puntos.

$$u_i-u_j+n x_{ij}\le n-1 \quad \forall i,j \ne 0, i\ne j$$
""")
    st.subheader("Modelo AMPL")
    st.code(make_mod(), language="ampl")

with tabs[3]:
    st.header("Implementación y resultados del TSP")
    result = solve_tsp(D)
    if not result["success"]:
        st.error("No se encontró solución: " + str(result["message"]))
    else:
        a, b, c, d = st.columns(4)
        a.metric("Distancia total", f"{result['objective']:.0f}")
        b.metric("Tiempo de solución", f"{result['time']:.4f} s")
        c.metric("Variables", result["variables"])
        d.metric("Restricciones", result["constraints"])
        st.subheader("Secuencia óptima de visita")
        st.markdown(f'<div class="route">{" → ".join(map(str, result["route"]))}</div>', unsafe_allow_html=True)
        st.subheader("Arcos seleccionados")
        st.dataframe(pd.DataFrame(result["selected_arcs"], columns=["origen", "destino", "distancia"]), use_container_width=True)
        st.success("Solución óptima encontrada con SciPy/HiGHS mediante formulación MTZ.")

with tabs[4]:
    st.header("Respuestas a las preguntas")
    result = solve_tsp(D)
    route_text = " → ".join(map(str, result["route"]))
    answers = [
        ("1. ¿Por qué no se resuelve el TSP directamente sobre toda la red vial original?", "Porque la red vial contiene muchas calles e intersecciones que no son puntos de entrega. Primero se calcula la distancia mínima entre los puntos relevantes y luego se trabaja con una matriz reducida 13×13, lo que conserva la información necesaria y hace el modelo más manejable."),
        ("2. ¿Qué representa una solución del TSP?", "Representa el orden en que el vehículo debe salir del depósito, visitar cada cliente exactamente una vez y regresar al depósito, minimizando la distancia total recorrida."),
        ("3. ¿Por qué esta instancia corre sin licencia comercial?", f"Porque solo tiene 13 puntos. El modelo genera {result['variables']} variables y {result['constraints']} restricciones, mucho menos que resolver sobre una red vial completa con cientos de nodos y miles de arcos."),
        ("4. Función objetivo del TSP", "La función objetivo minimiza la suma de las distancias de los arcos seleccionados: min Σ dᵢⱼxᵢⱼ. Es decir, busca la ruta total más corta."),
        ("5. Restricciones de salida y entrada", "Para cada punto se impone una salida única: Σ xᵢⱼ = 1, y una entrada única: Σ xᵢⱼ = 1. Con esto, cada cliente queda conectado una sola vez en la ruta."),
        ("6. Restricciones de eliminación de subciclos", "Las restricciones MTZ evitan que se formen ciclos pequeños separados del depósito. Son necesarias porque entrada y salida únicas por sí solas no garantizan una sola ruta completa."),
        ("7. Resultado del modelo", f"La distancia total óptima es {result['objective']:.0f}. La secuencia óptima es: {route_text}. Los arcos seleccionados aparecen en la pestaña Resolver TSP."),
        ("8. Si se agrega un cliente nuevo", "El tamaño del modelo aumenta rápido: con n puntos hay n(n−1) variables binarias xᵢⱼ y (n−1)(n−2) restricciones MTZ. Por eso un cliente adicional agrega muchas combinaciones nuevas."),
        ("9. Si una distancia aumenta", "La ruta óptima puede cambiar porque el solver intentará evitar arcos más costosos. Si ese arco era parte de la ruta óptima, posiblemente se sustituya por otra conexión con menor impacto en la distancia total."),
        ("10. Información adicional para reconstruir la ruta vial completa", "Además de la distancia mínima, se debe guardar la secuencia de nodos intermedios de cada camino más corto. Así se puede pasar de la ruta entre clientes a la ruta real sobre calles."),
    ]
    for q, ans in answers:
        st.markdown(f'<div class="answer"><b>{q}</b><br><br>{ans}</div>', unsafe_allow_html=True)

with tabs[5]:
    st.header("Archivos para entregar")
    result = solve_tsp(D)
    st.download_button("Descargar paquete .mod, .dat, matriz y resultados", package_zip(result), "caso2_tsp_13_entregables.zip", "application/zip")
    st.subheader("TSP_13.dat")
    st.code(make_dat(), language="ampl")
    st.info("Para evidencia formal en AMPL: model ampl/TSP_13.mod; data ampl/TSP_13.dat; option solver highs; solve; display Distancia_Total; display x;")
