import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP_AVAILABLE = True
except Exception:
    SCIPY_MILP_AVAILABLE = False

st.set_page_config(
    page_title="Camino más corto + TSP",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 35%);} 
    .stMetric {background-color: #ffffff; border: 1px solid #e7eaf0; padding: 14px; border-radius: 16px; box-shadow: 0 4px 14px rgba(30,41,59,.05);} 
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .hero {background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 28px; border-radius: 24px; margin-bottom: 20px;}
    .hero h1 {margin: 0; font-size: 2.15rem;}
    .hero p {margin-top: 10px; color: #dbeafe; font-size: 1rem;}
    .card {background: #ffffff; border: 1px solid #e7eaf0; border-radius: 18px; padding: 18px; margin: 10px 0; box-shadow: 0 4px 14px rgba(30,41,59,.04);} 
    .small-note {font-size: .92rem; color: #475569;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🚚 Optimización de rutas: Camino más corto + TSP</h1>
        <p>Aplicación para preparar la red vial, calcular la matriz Desde-Hasta y resolver/formular el Problema del Agente Viajero con eliminación de subciclos MTZ.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def read_network(file_bytes: bytes):
    text = file_bytes.decode("utf-8", errors="ignore").strip().splitlines()
    rows = []
    start_line = 0
    if not text:
        raise ValueError("El archivo de red está vacío.")
    first = text[0].split()
    # Si la primera línea trae número de nodos/arcos, se omite como encabezado.
    if len(first) <= 2 and len(text) > 1:
        try:
            int(first[0])
            start_line = 1
        except Exception:
            start_line = 0
    for line in text[start_line:]:
        parts = line.split()
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    arcs = pd.DataFrame(rows, columns=["origen", "destino", "distancia"])
    n_nodes = int(arcs[["origen", "destino"]].max().max()) + 1
    return n_nodes, arcs

@st.cache_data(show_spinner=False)
def read_clients(file_bytes: bytes):
    vals = []
    for line in file_bytes.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            vals.append(int(line.split()[0]))
    return vals

@st.cache_data(show_spinner=True)
def build_distance_matrix(n_nodes: int, arcs: pd.DataFrame, required_nodes_tuple):
    required_nodes = list(required_nodes_tuple)
    graph = csr_matrix((arcs["distancia"], (arcs["origen"], arcs["destino"])), shape=(n_nodes, n_nodes))
    dist = dijkstra(csgraph=graph, directed=True, indices=required_nodes, return_predecessors=False)
    matrix = pd.DataFrame(dist[:, required_nodes], index=required_nodes, columns=required_nodes)
    return matrix

def make_tsp_dat(matrix: pd.DataFrame):
    nodes = list(matrix.index)
    lines = [
        "set NODES := " + " ".join(map(str, nodes)) + ";",
        "param depot := 0;",
        "param d : " + " ".join(map(str, nodes)) + " :=",
    ]
    for i in nodes:
        vals = []
        for j in nodes:
            value = float(matrix.loc[i, j])
            if np.isinf(value):
                value = 999999999
            vals.append(str(int(round(value))))
        lines.append(str(i) + " " + " ".join(vals))
    lines.append(";")
    return "\n".join(lines)

def make_shortest_dat(n_nodes: int, arcs: pd.DataFrame, s: int, t: int):
    lines = ["set NODES := " + " ".join(map(str, range(n_nodes))) + ";", "set ARCS :="]
    for r in arcs.itertuples(index=False):
        lines.append(f"({int(r.origen)},{int(r.destino)})")
    lines.append(";")
    lines.append("param c :=")
    for r in arcs.itertuples(index=False):
        lines.append(f"{int(r.origen)} {int(r.destino)} {float(r.distancia):g}")
    lines.append(";")
    lines.append(f"param s := {s};")
    lines.append(f"param t := {t};")
    return "\n".join(lines)

def solve_tsp_mtz_scipy(matrix: pd.DataFrame, time_limit: int):
    if not SCIPY_MILP_AVAILABLE:
        raise RuntimeError("SciPy MILP no está disponible en este entorno.")
    nodes = list(matrix.index)
    depot = 0
    n = len(nodes)
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    arc_pos = {arc: k for k, arc in enumerate(arcs)}
    m_x = len(arcs)
    u_nodes = [node for node in nodes if node != depot]
    u_pos = {node: m_x + k for k, node in enumerate(u_nodes)}
    total_vars = m_x + len(u_nodes)

    c = np.zeros(total_vars)
    for k, (i, j) in enumerate(arcs):
        c[k] = matrix.loc[i, j]

    integrality = np.zeros(total_vars)
    integrality[:m_x] = 1
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, np.inf)
    ub[:m_x] = 1
    for node in u_nodes:
        lb[u_pos[node]] = 1
        ub[u_pos[node]] = n - 1

    n_constraints = 2 * n + (n - 1) * (n - 2)
    A = lil_matrix((n_constraints, total_vars))
    b_l = np.full(n_constraints, -np.inf)
    b_u = np.full(n_constraints, np.inf)
    row = 0

    for i in nodes:
        for j in nodes:
            if i != j:
                A[row, arc_pos[(i, j)]] = 1
        b_l[row] = b_u[row] = 1
        row += 1

    for j in nodes:
        for i in nodes:
            if i != j:
                A[row, arc_pos[(i, j)]] = 1
        b_l[row] = b_u[row] = 1
        row += 1

    for i in u_nodes:
        for j in u_nodes:
            if i != j:
                A[row, u_pos[i]] = 1
                A[row, u_pos[j]] = -1
                A[row, arc_pos[(i, j)]] = n
                b_u[row] = n - 1
                row += 1

    start = time.time()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A.tocsr(), b_l, b_u),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    elapsed = time.time() - start
    output = {"elapsed": elapsed, "num_vars": total_vars, "num_constraints": n_constraints, "message": result.message}
    if not result.success:
        output["success"] = False
        return output

    chosen = [arcs[k] for k, val in enumerate(result.x[:m_x]) if val > 0.5]
    next_node = {i: j for i, j in chosen}
    route = [depot]
    current = depot
    for _ in range(n + 1):
        current = next_node.get(current)
        if current is None:
            break
        route.append(current)
        if current == depot:
            break
    output.update({"success": True, "objective": result.fun, "route": route})
    return output

def zip_outputs(files: dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(name, content)
    buffer.seek(0)
    return buffer

def local_bytes(path: str):
    return Path(path).read_bytes()

with st.sidebar:
    st.header("📂 Datos")
    use_example = st.checkbox("Usar archivos incluidos", value=True)
    red_file = st.file_uploader("Red vial (.nf o .txt)", type=["nf", "txt"])
    clientes_file = st.file_uploader("Clientes (.txt)", type=["txt"])
    st.divider()
    st.header("⚙️ Solver")
    time_limit = st.slider("Tiempo límite en la app", 30, 600, 120, step=30)
    st.caption("Para la entrega formal, se recomienda correr también AMPL con TSP.mod y TSP.dat.")

if use_example and red_file is None and clientes_file is None:
    red_bytes = local_bytes("data/Red1.nf")
    clientes_bytes = local_bytes("data/Clientes1.txt")
elif red_file is not None and clientes_file is not None:
    red_bytes = red_file.getvalue()
    clientes_bytes = clientes_file.getvalue()
else:
    st.info("Subí ambos archivos o dejá marcada la opción de usar los archivos incluidos.")
    st.stop()

n_nodes, arcs = read_network(red_bytes)
clients = read_clients(clientes_bytes)
required_nodes = [0] + [c for c in clients if c != 0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodos red vial", f"{n_nodes:,}")
c2.metric("Arcos red vial", f"{len(arcs):,}")
c3.metric("Clientes", f"{len(clients):,}")
c4.metric("Nodos TSP", f"{len(required_nodes):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Datos", "🛣️ Caminos más cortos", "📐 Formulación", "🧩 TSP", "📦 Entregables"])

with tab1:
    st.markdown('<div class="card"><b>Depósito:</b> nodo 0. Los clientes se toman del archivo cargado.</div>', unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Red vial")
        st.dataframe(arcs.head(30), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Clientes")
        st.dataframe(pd.DataFrame({"cliente": clients}), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Parte 1: matriz Desde-Hasta")
    st.write("Se calcula el camino más corto entre el depósito y cada cliente, y también entre cada par de clientes.")
    if st.button("Calcular matriz Desde-Hasta", type="primary"):
        start = time.time()
        matrix = build_distance_matrix(n_nodes, arcs, tuple(required_nodes))
        st.session_state["matrix"] = matrix
        st.session_state["matrix_time"] = time.time() - start
    if "matrix" in st.session_state:
        matrix = st.session_state["matrix"]
        if np.isinf(matrix.values).any():
            st.warning("Hay pares de nodos sin camino alcanzable. Se reemplazarán por un costo grande en el .dat.")
        st.success(f"Matriz calculada en {st.session_state.get('matrix_time', 0):.2f} segundos.")
        st.dataframe(matrix, use_container_width=True)
        st.download_button("Descargar matriz CSV", matrix.to_csv(index=True).encode("utf-8"), "matriz_desde_hasta.csv", "text/csv")
    else:
        st.info("Presioná el botón para construir la matriz antes de generar el TSP.dat.")

with tab3:
    st.subheader("Parte 2: formulación matemática TSP")
    st.markdown(
        r"""
        **Variable:** $x_{ij}=1$ si se viaja del nodo $i$ al nodo $j$; 0 en caso contrario.  
        **Objetivo:** $\min \sum_i \sum_j d_{ij}x_{ij}$  
        **Salida única:** $\sum_{j \ne i}x_{ij}=1$  
        **Entrada única:** $\sum_{i \ne j}x_{ij}=1$  
        **MTZ:** $u_i-u_j+n x_{ij}\le n-1$, para eliminar subciclos.
        """
    )
    st.code(Path("ampl/TSP.mod").read_text(encoding="utf-8"), language="ampl")

with tab4:
    st.subheader("Parte 3 y 4: TSP y resultados")
    if "matrix" not in st.session_state:
        st.info("Primero calculá la matriz Desde-Hasta en la pestaña de caminos más cortos.")
    else:
        matrix = st.session_state["matrix"]
        tsp_dat = make_tsp_dat(matrix)
        shortest_dat = make_shortest_dat(n_nodes, arcs, required_nodes[0], required_nodes[1])
        files = {
            "ampl/TSP.mod": Path("ampl/TSP.mod").read_text(encoding="utf-8"),
            "ampl/TSP.dat": tsp_dat,
            "ampl/CaminoMasCorto.mod": Path("ampl/CaminoMasCorto.mod").read_text(encoding="utf-8"),
            "ampl/CaminoMasCorto.dat": shortest_dat,
            "outputs/matriz_desde_hasta.csv": matrix.to_csv(index=True),
        }
        st.download_button("Descargar .mod, .dat y matriz", zip_outputs(files), "resultados_ampl_tsp.zip", "application/zip")
        with st.expander("Ver TSP.dat generado"):
            st.code(tsp_dat[:12000] + ("\n..." if len(tsp_dat) > 12000 else ""), language="ampl")
        st.warning("La instancia tiene muchos nodos. La app puede intentar resolverla, pero AMPL con CPLEX/Gurobi/HiGHS es la evidencia formal más fuerte.")
        if st.button("Resolver TSP en la app", type="primary"):
            with st.spinner("Resolviendo formulación MTZ exacta..."):
                result = solve_tsp_mtz_scipy(matrix, time_limit)
            st.session_state["tsp_result"] = result
        if "tsp_result" in st.session_state:
            result = st.session_state["tsp_result"]
            r1, r2, r3, r4 = st.columns(4)
            r2.metric("Tiempo", f"{result['elapsed']:.2f} s")
            r3.metric("Variables", f"{result['num_vars']:,}")
            r4.metric("Restricciones", f"{result['num_constraints']:,}")
            if result["success"]:
                r1.metric("Distancia total", f"{result['objective']:.0f}")
                st.success("Solución óptima encontrada/certificada por el solver de la app.")
                st.code(" -> ".join(map(str, result["route"])))
            else:
                r1.metric("Distancia total", "No certificada")
                st.error("No se logró certificar optimalidad dentro del tiempo límite configurado.")
                st.write(result["message"])

with tab5:
    st.subheader("Checklist de entrega")
    st.markdown(
        """
        - Informe técnico en PDF.
        - `CaminoMasCorto.mod` y `CaminoMasCorto.dat`.
        - `TSP.mod` y `TSP.dat`.
        - Matriz Desde-Hasta como resultado intermedio.
        - Evidencia de ejecución de caminos más cortos y TSP.
        - Ruta óptima, distancia total, tiempo, variables y restricciones.
        - Análisis de limitaciones y mejoras para instancias grandes.
        """
    )
    st.subheader("Comandos sugeridos en AMPL")
    st.code(
        """reset;
model ampl/TSP.mod;
data ampl/TSP.dat;
option solver cplex;   # o gurobi/highs según disponibilidad
solve;
display Distancia_Total;
display x;
display u;""",
        language="ampl",
    )
