
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# BEAMLAB 1.0
# Beam Deflection and Stress Simulator
# ==========================================

st.set_page_config(
    page_title="BeamLab",
    page_icon="📐",
    layout="wide"
)

st.title("📐 BeamLab")
st.subheader("Beam Deflection and Stress Simulator")

st.write(
    "Analyze beam reactions, shear forces, bending moments, "
    "bending stress, and elastic deflection."
)

st.divider()

# ==========================================
# SIDEBAR: INPUT PARAMETERS
# ==========================================

st.sidebar.header("Beam Configuration")

beam_type = st.sidebar.selectbox(
    "Select beam type",
    ["Simply Supported", "Cantilever"]
)

L = st.sidebar.number_input(
    "Beam length (m)",
    min_value=0.1,
    max_value=100.0,
    value=4.0,
    step=0.5
)

P = st.sidebar.number_input(
    "Point load (N)",
    min_value=0.0,
    max_value=1_000_000.0,
    value=100.0,
    step=10.0
)

if beam_type == "Simply Supported":
    a = st.sidebar.number_input(
        "Load distance from left support (m)",
        min_value=0.0,
        max_value=float(L),
        value=float(L / 2),
        step=0.1
    )
else:
    a = st.sidebar.number_input(
        "Load distance from fixed support (m)",
        min_value=0.0,
        max_value=float(L),
        value=float(L),
        step=0.1
    )

st.sidebar.header("Material")

material = st.sidebar.selectbox(
    "Select material",
    ["Steel", "Aluminium"]
)

if material == "Steel":
    E = 200e9
else:
    E = 70e9

st.sidebar.header("Cross-Section")

b_mm = st.sidebar.number_input(
    "Width (mm)",
    min_value=1.0,
    max_value=1000.0,
    value=50.0,
    step=5.0
)

h_mm = st.sidebar.number_input(
    "Height (mm)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    step=5.0
)

# Convert dimensions to metres
b = b_mm / 1000
h = h_mm / 1000

# Second moment of area for a rectangle
I = b * h**3 / 12

# Distance from neutral axis to outermost fibre
c = h / 2

# ==========================================
# ENGINEERING CALCULATIONS
# ==========================================

x = np.linspace(0, L, 501)

# Initialize results
RA = 0.0
RB = 0.0
M_fixed = 0.0

V = np.zeros_like(x)
M = np.zeros_like(x)
y = np.zeros_like(x)

if beam_type == "Simply Supported":

    # Reactions
    RA = P * (L - a) / L
    RB = P * a / L

    # Shear force
    V = np.where(x < a, RA, RA - P)

    # Bending moment
    M = np.where(
        x <= a,
        RA * x,
        RA * x - P * (x - a)
    )

    # Deflection: piecewise Euler-Bernoulli solution
    left = x <= a
    right = ~left

    y[left] = (
        -P * (L - a) * x[left]
        / (6 * L * E * I)
        * (L**2 - (L - a)**2 - x[left]**2)
    )

    y[right] = (
        -P * a * (L - x[right])
        / (6 * L * E * I)
        * (L**2 - a**2 - (L - x[right])**2)
    )

    M_fixed = 0.0

else:

    # Cantilever: fixed support at x = 0
    # Point load is located at x = a.
    RA = P
    RB = 0.0

    M_fixed = P * a

    # Shear force and bending moment
    V = np.where(x < a, -P, 0.0)
    M = np.where(x <= a, -P * (a - x), 0.0)

    # Deflection for a point load at x = a
    left = x <= a
    right = ~left

    y[left] = (
        -P * x[left]**2 * (3 * a - x[left])
        / (6 * E * I)
    )

    y[right] = (
        -P * a**2 * (3 * x[right] - a)
        / (6 * E * I)
    )

# Maximum values
M_max = float(np.max(np.abs(M)))
V_max = float(np.max(np.abs(V)))
stress_max = M_max * c / I
deflection_max = float(np.max(np.abs(y)))

# ==========================================
# RESULTS DASHBOARD
# ==========================================

st.header("Analysis Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Maximum Moment",
    f"{M_max:.2f} N·m"
)

col2.metric(
    "Maximum Shear",
    f"{V_max:.2f} N"
)

col3.metric(
    "Maximum Stress",
    f"{stress_max / 1e6:.3f} MPa"
)

col4.metric(
    "Maximum Deflection",
    f"{deflection_max * 1000:.3f} mm"
)

st.subheader("Support Reactions")

if beam_type == "Simply Supported":
    col1, col2 = st.columns(2)

    col1.metric("Left Reaction", f"{RA:.2f} N")
    col2.metric("Right Reaction", f"{RB:.2f} N")

else:
    col1, col2 = st.columns(2)

    col1.metric("Vertical Reaction", f"{RA:.2f} N")
    col2.metric("Fixed-End Moment", f"{M_fixed:.2f} N·m")

st.caption(
    "Reactions and internal forces use the sign conventions "
    "implemented in the analysis. Magnitudes are shown above."
)

# ==========================================
# BEAM VISUALIZATION
# ==========================================

st.divider()
st.header("Beam Visualization")

fig, ax = plt.subplots(figsize=(10, 2.8))

ax.plot(
    [0, L],
    [0, 0],
    linewidth=4
)

# Draw supports
if beam_type == "Simply Supported":

    ax.scatter([0, L], [0, 0], marker="^", s=180)

else:

    ax.plot(
        [0, 0],
        [-0.25, 0.25],
        linewidth=5
    )

    for yy in np.linspace(-0.22, 0.22, 7):
        ax.plot(
            [-0.12, 0],
            [yy - 0.06, yy],
            linewidth=1
        )

# Draw applied load
if P > 0:

    ax.annotate(
        "",
        xy=(a, -0.05),
        xytext=(a, 0.65),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

    ax.text(
        a,
        0.72,
        f"{P:.1f} N",
        ha="center"
    )

ax.text(0, -0.32, "A", ha="center")
ax.text(L, -0.32, "B", ha="center")

ax.set_xlim(-0.25, L + 0.25)
ax.set_ylim(-0.55, 1.0)
ax.set_xlabel("Beam Length (m)")
ax.set_yticks([])
ax.set_title(f"{beam_type} Beam")
ax.grid(axis="x", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# SHEAR FORCE DIAGRAM
# ==========================================

st.divider()
st.header("Shear Force Diagram (SFD)")

fig, ax = plt.subplots(figsize=(10, 3.5))

ax.plot(x, V, linewidth=2)
ax.fill_between(x, V, 0, alpha=0.2)
ax.axhline(0, linewidth=1)

ax.set_xlabel("Distance Along Beam (m)")
ax.set_ylabel("Shear Force (N)")
ax.set_title("Shear Force Diagram")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# BENDING MOMENT DIAGRAM
# ==========================================

st.header("Bending Moment Diagram (BMD)")

fig, ax = plt.subplots(figsize=(10, 3.5))

ax.plot(x, M, linewidth=2)
ax.fill_between(x, M, 0, alpha=0.2)
ax.axhline(0, linewidth=1)

ax.set_xlabel("Distance Along Beam (m)")
ax.set_ylabel("Bending Moment (N·m)")
ax.set_title("Bending Moment Diagram")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# DEFLECTION CURVE
# ==========================================

st.header("Elastic Deflection Curve")

# Exaggerate the plotted shape for visibility only
visual_scale = st.slider(
    "Deflection visualization scale",
    min_value=1,
    max_value=1000,
    value=100,
    step=10
)

fig, ax = plt.subplots(figsize=(10, 3.5))

ax.plot(
    x,
    np.zeros_like(x),
    linestyle="--",
    linewidth=1.5,
    label="Original beam"
)

ax.plot(
    x,
    y * visual_scale,
    linewidth=2,
    label="Deflected shape"
)

ax.set_xlabel("Distance Along Beam (m)")
ax.set_ylabel("Illustrative Deflection")
ax.set_title("Beam Deflection")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "The deflection curve is exaggerated for visibility. "
    "The reported deflection remains the calculated physical value."
)

# ==========================================
# ENGINEERING SUMMARY
# ==========================================

st.divider()
st.header("Engineering Summary")

st.write(f"**Beam type:** {beam_type}")
st.write(f"**Material:** {material}")
st.write(f"**Young's modulus:** {E / 1e9:.1f} GPa")
st.write(f"**Second moment of area:** {I:.6e} m⁴")
st.write(f"**Maximum bending stress:** {stress_max / 1e6:.4f} MPa")
st.write(f"**Maximum deflection:** {deflection_max * 1000:.4f} mm")

st.info(
    "This educational simulator assumes linear-elastic Euler–Bernoulli "
    "beam behavior, small deflections, ideal supports, and a "
    "rectangular prismatic cross-section. It does not replace "
    "professional structural design or safety verification."
)
