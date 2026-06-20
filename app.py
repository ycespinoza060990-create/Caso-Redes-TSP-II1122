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

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AMPL_DIR = BASE_DIR / "ampl"

st.set_page_config(
    page_title="Caso Redes | Camino más corto + TSP",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {--txt:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --card:#ffffff; --accent:#2563eb;}
    .stApp {background: linear-gradient(180deg, #f8fafc 0%, #ffffff 55%);} 
    .block-container {padding-top: 1.3rem; padding-bottom: 2.2rem; max-width: 1380px;}
    .hero {
        background: radial-gradient(circle at top left, #3b82f6 0%, #1e293b 42%, #020617 100%);
        color: white; padding: 30px 32px; border-radius: 28px; margin-bottom: 18px;
        box-shadow: 0 18px 40px rgba(15,23,42,.18);
    }
    .hero h1 {margin:0; font-size:2.15rem; line-height:1.15; font-weight:800;}
    .hero p {margin:12px 0 0 0; color:#dbeafe; font-size:1.02rem; max-width:960px;}
    .pill {display:inline-block; padding:7px 12px; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); margin-bottom:10px; font-size:.85rem;}
    .card {background:#fff; border:1px solid #e2e8f0; border-radius:20px; padding:18px; margin:10px 0; box-shadow:0 8px 22px rgba(15,23,42,.045);} 
    .soft {background:#f8fafc; border:1px solid #e2e8f0; border-radius:18px; padding:14px 16px;}
    .ok {background:#ecfdf5; border:1px solid #bbf7d0; color:#14532d; border-radius:16px; padding:12px 14px;}
    .warn {background:#fffbeb; border:1px solid #fde68a; color:#78350f; border-radius:16px; padding:12px 14px;}
    div[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:18px; padding:14px 16px; box-shadow:0 8px 22px rgba(15,23,42,.045);} 
    div[data-testid="stMetricValue"] {font-size:1.65rem; font-weight:800; color:#0f172a;}
    .small-note {font-size:.92rem; color:#475569;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <span class="pill">II-1122 Optimización Industrial · Caso Redes</span>
      <h1>Camino más corto + Problema del Agente Viajero</h1>
      <p>App en Streamlit para cargar la red vial y los clientes, calcular la matriz Desde-Hasta con caminos más cortos, generar archivos AMPL y resolver el TSP con formulación MTZ cuando el entorno lo permita.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def read_network(file_bytes: bytes):
    lines = file_bytes.decode("utf-8", errors="ignore").strip().splitlines()
    if not lines:
        raise ValueError("El archivo de red está vacío.")
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError("No se pudieron leer arcos con formato: NodoOrigen NodoDestino Distancia.")
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
    if not vals:
        raise ValueError("El archivo de clientes está vacío.")
    return vals

@st.cache_data(show_spinner=True)
def build_distance_matrix(n_nodes: int, arcs: pd.DataFrame, required_nodes_tuple):
    required_nodes = list(required_nodes_tuple)
    graph = csr_matrix((arcs["distancia"].to_numpy(), (arcs["origen"].to_numpy(), arcs["destino"].to_numpy())), shape=(n_nodes, n_nodes))
    dist = dijkstra(csgraph=graph, directed=True, indices=required_nodes, return_predecessors=False)
    return pd.DataFrame(dist[:, required_nodes], index=required_nodes, columns=required_nodes)

@st.cache_data(show_spinner=True)
def build_distance_and_paths(n_nodes: int, arcs: pd.DataFrame, required_nodes_tuple):
    required_nodes = list(required_nodes_tuple)
    graph = csr_matrix((arcs["distancia"].to_numpy(), (arcs["origen"].to_numpy(), arcs["destino"].to_numpy())), shape=(n_nodes, n_nodes))
    dist, pred = dijkstra(csgraph=graph, directed=True, indices=required_nodes, return_predecessors=True)
    matrix = pd.DataFrame(dist[:, required_nodes], index=required_nodes, columns=required_nodes)
    return matrix, pred

def reconstruct_path(pred_matrix, required_nodes, origin, dest):
    source_row = required_nodes.index(origin)
    if origin == dest:
        return [origin]
    path = [dest]
    current = dest
    seen = set()
    while current != origin:
        if current in seen:
            return []
        seen.add(current)
        current = int(pred_matrix[source_row, current])
        if current < 0:
            return []
        path.append(current)
    return list(reversed(path))

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

def route_table(route, matrix: pd.DataFrame):
    rows = []
    total = 0
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        d = float(matrix.loc[i, j])
        total += d
        rows.append({"orden": k + 1, "desde": i, "hasta": j, "distancia tramo": d, "acumulado": total})
    return pd.DataFrame(rows)

def solve_tsp_mtz_scipy(matrix: pd.DataFrame, time_limit: int):
    if not SCIPY_MILP_AVAILABLE:
        raise RuntimeError("SciPy MILP no está disponible en este entorno.")
    nodes = list(matrix.index)
    depot = 0
    n = len(nodes)
    if np.isinf(matrix.values).any():
        raise RuntimeError("Hay distancias infinitas en la matriz; revise conectividad de la red.")
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    arc_pos = {arc: k for k, arc in enumerate(arcs)}
    m_x = len(arcs)
    u_nodes = [node for node in nodes if node != depot]
    u_pos = {node: m_x + k for k, node in enumerate(u_nodes)}
    total_vars = m_x + len(u_nodes)

    c = np.zeros(total_vars)
    for k, (i, j) in enumerate(arcs):
        c[k] = float(matrix.loc[i, j])

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
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
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
    for _ in range(n + 2):
        current = next_node.get(current)
        if current is None:
            break
        route.append(current)
        if current == depot:
            break
    output.update({"success": True, "objective": float(result.fun), "route": route, "chosen_arcs": chosen})
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

def local_bytes(path: Path):
    return path.read_bytes()

with st.sidebar:
    st.header("📂 Datos")
    use_example = st.checkbox("Usar archivos incluidos", value=True)
    red_file = st.file_uploader("Red vial (.nf o .txt)", type=["nf", "txt"])
    clientes_file = st.file_uploader("Clientes (.txt)", type=["txt"])
    st.divider()
    st.header("⚙️ Configuración")
    time_limit = st.slider("Tiempo límite para TSP exacto", 30, 900, 180, step=30)
    st.caption("La app genera los archivos AMPL. El TSP exacto puede tardar bastante en instancias grandes.")

try:
    if use_example and red_file is None and clientes_file is None:
        red_bytes = local_bytes(DATA_DIR / "Red1.nf")
        clientes_bytes = local_bytes(DATA_DIR / "Clientes1.txt")
    elif red_file is not None and clientes_file is not None:
        red_bytes = red_file.getvalue()
        clientes_bytes = clientes_file.getvalue()
    else:
        st.info("Subí ambos archivos o dejá marcada la opción de usar los archivos incluidos.")
        st.stop()

    n_nodes, arcs = read_network(red_bytes)
    clients = read_clients(clientes_bytes)
except Exception as exc:
    st.error(f"No se pudieron cargar los datos: {exc}")
    st.stop()

required_nodes = [0] + [c for c in clients if c != 0]
n_tsp = len(required_nodes)
num_x = n_tsp * (n_tsp - 1)
num_u = n_tsp - 1
num_vars = num_x + num_u
num_constraints = 2 * n_tsp + (n_tsp - 1) * (n_tsp - 2)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nodos red vial", f"{n_nodes:,}")
m2.metric("Arcos red vial", f"{len(arcs):,}")
m3.metric("Clientes", f"{len(clients):,}")
m4.metric("Nodos TSP", f"{n_tsp:,}")
m5.metric("Variables MTZ", f"{num_vars:,}")

if n_tsp > 70:
    st.markdown('<div class="warn"><b>Nota:</b> esta instancia es grande para una formulación MTZ exacta en una app web. La app deja evidencia, matriz, archivos .mod/.dat y puede intentar resolver, pero para la entrega formal conviene guardar también una corrida en AMPL con un solver fuerte.</div>', unsafe_allow_html=True)

(tab_resumen, tab_datos, tab_camino, tab_formulacion, tab_tsp, tab_entrega) = st.tabs([
    "📌 Resumen", "📂 Datos", "🛣️ Camino más corto", "📐 Formulación", "🧩 TSP", "📦 Entregables"
])

with tab_resumen:
    a, b = st.columns([1.25, 1])
    with a:
        st.markdown("""
        <div class="card">
        <h3 style="margin-top:0">Flujo de solución</h3>
        <ol>
          <li>Leer la red vial con formato <b>NodoOrigen NodoDestino Distancia</b>.</li>
          <li>Leer los clientes y agregar el depósito como nodo 0.</li>
          <li>Calcular caminos más cortos entre todos los puntos requeridos.</li>
          <li>Construir la matriz Desde-Hasta.</li>
          <li>Formular y resolver el TSP sobre la matriz reducida.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown(f"""
        <div class="card">
        <h3 style="margin-top:0">Tamaño esperado del TSP</h3>
        <p><b>Nodos:</b> {n_tsp}</p>
        <p><b>Variables binarias xᵢⱼ:</b> {num_x:,}</p>
        <p><b>Variables MTZ uᵢ:</b> {num_u:,}</p>
        <p><b>Restricciones:</b> {num_constraints:,}</p>
        </div>
        """, unsafe_allow_html=True)

with tab_datos:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Red vial")
        st.dataframe(arcs.head(80), use_container_width=True, hide_index=True)
        st.caption("Se muestra una vista previa de los primeros arcos.")
    with right:
        st.subheader("Puntos de entrega")
        st.dataframe(pd.DataFrame({"cliente": clients}), use_container_width=True, hide_index=True)
        st.download_button("Descargar clientes usados", pd.DataFrame({"cliente": clients}).to_csv(index=False).encode("utf-8"), "clientes_usados.csv", "text/csv")

with tab_camino:
    st.subheader("Parte 1 · Matriz Desde-Hasta")
    st.write("Aquí se calcula el camino más corto entre el depósito y cada cliente, y entre cada par de clientes. Ese resultado es la matriz que alimenta el TSP.")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        calc_paths = st.button("Calcular matriz Desde-Hasta", type="primary", use_container_width=True)
    with col_b:
        st.markdown('<div class="soft">El cálculo usa Dijkstra sobre la red vial original.</div>', unsafe_allow_html=True)

    if calc_paths:
        start = time.time()
        with st.spinner("Calculando caminos más cortos..."):
            matrix, pred = build_distance_and_paths(n_nodes, arcs, tuple(required_nodes))
        st.session_state["matrix"] = matrix
        st.session_state["pred"] = pred
        st.session_state["matrix_time"] = time.time() - start

    if "matrix" in st.session_state:
        matrix = st.session_state["matrix"]
        st.success(f"Matriz Desde-Hasta calculada en {st.session_state.get('matrix_time', 0):.2f} segundos.")
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Filas", matrix.shape[0])
        c_b.metric("Columnas", matrix.shape[1])
        c_c.metric("Pares sin camino", int(np.isinf(matrix.values).sum()))
        st.dataframe(matrix, use_container_width=True)
        st.download_button("Descargar matriz Desde-Hasta CSV", matrix.to_csv(index=True).encode("utf-8"), "matriz_desde_hasta.csv", "text/csv", use_container_width=True)

        st.subheader("Ejemplo de camino más corto")
        e1, e2, e3 = st.columns([1, 1, 2])
        with e1:
            origen_ej = st.selectbox("Origen", required_nodes, index=0)
        with e2:
            destino_ej = st.selectbox("Destino", required_nodes, index=min(1, len(required_nodes)-1))
        if st.button("Mostrar camino elegido"):
            path = reconstruct_path(st.session_state["pred"], required_nodes, origen_ej, destino_ej)
            if path:
                st.code(" -> ".join(map(str, path)))
                st.write(f"Distancia mínima: **{float(matrix.loc[origen_ej, destino_ej]):.0f}**")
            else:
                st.warning("No se encontró camino entre esos nodos.")
    else:
        st.info("Presioná el botón para calcular la matriz.")

with tab_formulacion:
    st.subheader("Parte 2 · Formulación matemática")
    st.markdown(r"""
    **Conjuntos:**  
    $N$: depósito y clientes. El depósito es el nodo 0.  

    **Parámetro:**  
    $d_{ij}$: distancia mínima desde el nodo $i$ hasta el nodo $j$, calculada con caminos más cortos.  

    **Variables:**  
    $x_{ij}=1$ si se viaja directamente de $i$ a $j$ en la ruta TSP; $0$ en caso contrario.  
    $u_i$: variable de orden usada para eliminar subciclos.  

    **Función objetivo:**  
    $$\min \sum_{i\in N}\sum_{j\in N, j\ne i} d_{ij}x_{ij}$$

    **Restricciones principales:**  
    Salida única: $$\sum_{j\in N,j\ne i}x_{ij}=1 \quad \forall i\in N$$  
    Entrada única: $$\sum_{i\in N,i\ne j}x_{ij}=1 \quad \forall j\in N$$  
    Eliminación de subciclos MTZ: $$u_i-u_j+n x_{ij}\le n-1$$
    """)
    st.subheader("Modelo AMPL TSP.mod")
    st.code((AMPL_DIR / "TSP.mod").read_text(encoding="utf-8"), language="ampl")
    st.subheader("Modelo AMPL CaminoMasCorto.mod")
    st.code((AMPL_DIR / "CaminoMasCorto.mod").read_text(encoding="utf-8"), language="ampl")

with tab_tsp:
    st.subheader("Parte 3 y 4 · Implementación y resultados TSP")
    if "matrix" not in st.session_state:
        st.info("Primero calculá la matriz Desde-Hasta en la pestaña de Camino más corto.")
    else:
        matrix = st.session_state["matrix"]
        tsp_dat = make_tsp_dat(matrix)
        shortest_dat = make_shortest_dat(n_nodes, arcs, required_nodes[0], required_nodes[1])
        files = {
            "ampl/TSP.mod": (AMPL_DIR / "TSP.mod").read_text(encoding="utf-8"),
            "ampl/TSP.dat": tsp_dat,
            "ampl/CaminoMasCorto.mod": (AMPL_DIR / "CaminoMasCorto.mod").read_text(encoding="utf-8"),
            "ampl/CaminoMasCorto.dat": shortest_dat,
            "outputs/matriz_desde_hasta.csv": matrix.to_csv(index=True),
        }
        st.download_button("Descargar paquete .mod, .dat y matriz", zip_outputs(files), "resultados_ampl_tsp.zip", "application/zip", use_container_width=True)
        with st.expander("Vista previa del TSP.dat generado"):
            st.code(tsp_dat[:16000] + ("\n..." if len(tsp_dat) > 16000 else ""), language="ampl")

        r1, r2, r3 = st.columns(3)
        r1.metric("Variables del modelo", f"{num_vars:,}")
        r2.metric("Restricciones", f"{num_constraints:,}")
        r3.metric("Nodos visitados", f"{n_tsp}")

        if st.button("Resolver TSP exacto en Streamlit", type="primary", use_container_width=True):
            with st.spinner("Resolviendo TSP con MILP + MTZ. Puede tardar según la instancia..."):
                try:
                    result = solve_tsp_mtz_scipy(matrix, time_limit)
                except Exception as exc:
                    result = {"success": False, "elapsed": 0, "num_vars": num_vars, "num_constraints": num_constraints, "message": str(exc)}
            st.session_state["tsp_result"] = result

        if "tsp_result" in st.session_state:
            result = st.session_state["tsp_result"]
            a, b, c, d = st.columns(4)
            b.metric("Tiempo solución", f"{result.get('elapsed', 0):.2f} s")
            c.metric("Variables", f"{result.get('num_vars', num_vars):,}")
            d.metric("Restricciones", f"{result.get('num_constraints', num_constraints):,}")
            if result.get("success"):
                a.metric("Distancia total", f"{result['objective']:.0f}")
                st.markdown('<div class="ok"><b>Solución TSP encontrada.</b> La ruta inicia y finaliza en el depósito 0.</div>', unsafe_allow_html=True)
                st.code(" -> ".join(map(str, result["route"])))
                rt = route_table(result["route"], matrix)
                st.dataframe(rt, use_container_width=True, hide_index=True)
                st.download_button("Descargar ruta óptima CSV", rt.to_csv(index=False).encode("utf-8"), "ruta_tsp.csv", "text/csv", use_container_width=True)
            else:
                a.metric("Distancia total", "No certificada")
                st.error("No se logró certificar una solución óptima dentro del tiempo configurado.")
                st.write(result.get("message", "Sin mensaje del solver."))
                st.markdown('<div class="warn">Para la evidencia formal, descargá el TSP.dat y corré TSP.mod en AMPL con CPLEX, Gurobi o HiGHS.</div>', unsafe_allow_html=True)

with tab_entrega:
    st.subheader("Checklist alineado con la rúbrica")
    checklist = pd.DataFrame({
        "Punto solicitado": [
            "Formulación, implementación y resultados de Camino Más Corto",
            "Formulación matemática TSP",
            "Implementación en AMPL TSP",
            "Solución del modelo TSP",
            "Presentación e interpretación de resultados",
            "Organización y documentación del código",
        ],
        "Dónde está en la app": [
            "Pestaña Camino más corto + matriz Desde-Hasta + CSV",
            "Pestaña Formulación",
            "TSP.mod y TSP.dat descargables",
            "Pestaña TSP / corrida en AMPL",
            "Métricas, ruta, distancia, tiempo, variables y restricciones",
            "README + archivos separados por carpetas",
        ],
    })
    st.dataframe(checklist, use_container_width=True, hide_index=True)
    st.subheader("Preguntas de análisis que se responden con la corrida")
    st.markdown(f"""
    - **Secuencia óptima de visitas:** se muestra en la pestaña TSP cuando el solver encuentra solución.
    - **Distancia total recorrida:** se reporta como métrica principal del TSP.
    - **Tamaño del modelo:** para esta instancia son **{num_vars:,} variables** y **{num_constraints:,} restricciones**.
    - **Limitaciones:** la formulación MTZ crece rápido porque usa muchas variables binarias y restricciones de subciclos.
    - **Mejoras:** usar solvers más fuertes, cortes de subciclos dinámicos, formulaciones más ajustadas o descomposición para instancias mayores.
    """)
    st.subheader("Comandos sugeridos en AMPL")
    st.code(
        """reset;
model ampl/TSP.mod;
data ampl/TSP.dat;
option solver cplex;   # también puede usarse gurobi o highs si están disponibles
solve;
display Distancia_Total;
display x;
display u;""",
        language="ampl",
    )
