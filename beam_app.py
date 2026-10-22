import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="BeamLab",
    page_icon="📐",
    layout="wide"
)

st.title("📐 BeamLab 2.0")
st.subheader("Beam Deflection and Stress Simulator")

st.write(
    "Analyze beam reactions, shear forces, bending moments, "
    "bending stress, and elastic deflection under point loads "
    "and uniformly distributed loads."
)

st.divider()


MATERIALS = {
    "Steel": 200e9,
    "Cast Iron": 100e9,
    "Aluminium": 70e9,
    "Copper": 120e9,
    "Brass": 100e9,
    "Titanium Alloy": 110e9,
}

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
    step=0.5,
    key="beam_length"
)


st.sidebar.header("Loading Configuration")

load_type = st.sidebar.selectbox(
    "Select load type",
    ["Point Load", "Uniformly Distributed Load (UDL)"]
)

P = 0.0
w = 0.0
a = 0.0
b_load = L

if load_type == "Point Load":

    P = st.sidebar.number_input(
        "Point load, P (N)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=100.0,
        step=10.0,
        key="point_load"
    )

    a = st.sidebar.number_input(
        (
            "Load position from left support (m)"
            if beam_type == "Simply Supported"
            else "Load position from fixed support (m)"
        ),
        min_value=0.0,
        max_value=float(L),
        value=float(L / 2 if beam_type == "Simply Supported" else L),
        step=0.1,
        key="point_position"
    )

else:

    w = st.sidebar.number_input(
        "UDL intensity, w (N/m)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=500.0,
        step=50.0,
        key="udl_intensity"
    )

    st.sidebar.caption(
        "The UDL acts downward between its start and end positions."
    )

    a = st.sidebar.number_input(
        "UDL start position (m)",
        min_value=0.0,
        max_value=float(L),
        value=0.0,
        step=0.1,
        key="udl_start"
    )

    b_load = st.sidebar.number_input(
        "UDL end position (m)",
        min_value=0.0,
        max_value=float(L),
        value=float(L),
        step=0.1,
        key="udl_end"
    )


st.sidebar.header("Material")

material = st.sidebar.selectbox(
    "Select material",
    list(MATERIALS.keys())
)

E = MATERIALS[material]

st.sidebar.info(
    f"**Material:** {material}\n\n"
    f"**Young's modulus:** {E / 1e9:.1f} GPa\n\n"
    f"**Young's modulus:** {E:.3e} Pa"
)

st.sidebar.header("Cross-Section")

b_mm = st.sidebar.number_input(
    "Width (mm)",
    min_value=1.0,
    max_value=1000.0,
    value=50.0,
    step=5.0,
    key="section_width"
)

h_mm = st.sidebar.number_input(
    "Height (mm)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    step=5.0,
    key="section_height"
)

# Convert millimetres to metres.
b = b_mm / 1000.0
h = h_mm / 1000.0

# Rectangular cross-section.
I = b * h**3 / 12.0

# Distance from neutral axis to outermost fibre.
c = h / 2.0


if load_type == "Uniformly Distributed Load (UDL)":

    if b_load <= a:
        st.error(
            "The UDL end position must be greater than "
            "its start position. Adjust the load positions."
        )
        st.stop()



special_positions = [0.0, float(L)]

if load_type == "Point Load":
    special_positions.append(float(a))
else:
    special_positions.extend([float(a), float(b_load)])

x = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, L, 2001),
            np.asarray(special_positions)
        ]
    )
)

# Initialize result arrays.
V = np.zeros_like(x)
M = np.zeros_like(x)
y = np.zeros_like(x)

RA = 0.0
RB = 0.0
M_fixed = 0.0


if beam_type == "Simply Supported":

    if load_type == "Point Load":

        # Support reactions.
        RA = P * (L - a) / L
        RB = P * a / L

        # Shear force.
        V = np.where(
            x < a,
            RA,
            RA - P
        )

        # Bending moment.
        M = (
            RA * x
            - P * np.maximum(x - a, 0.0)
        )

        # Deflection:
        #
        # y = -(RA*x^3/6
        #       - P*<x-a>^3/6
        #       + C1*x) / (E*I)
        #
        # <x-a> means max(x-a, 0).
        # C1 is obtained from y(L) = 0.

        G = (
            RA * x**3 / 6.0
            - P * np.maximum(x - a, 0.0)**3 / 6.0
        )

        G_L = (
            RA * L**3 / 6.0
            - P * max(L - a, 0.0)**3 / 6.0
        )

        C1 = -G_L / L

        y = -(G + C1 * x) / (E * I)

    else:

        # Partial or full-span UDL.
        udl_length = b_load - a
        total_load = w * udl_length

        # Equivalent resultant acts at the midpoint of the
        # loaded region.
        resultant_position = (a + b_load) / 2.0

        # Support reactions from equilibrium.
        RA = total_load * (L - resultant_position) / L
        RB = total_load * resultant_position / L

        # Length of the UDL to the left of each section.
        loaded_length = np.clip(
            x - a,
            0.0,
            udl_length
        )

        # Shear force.
        V = RA - w * loaded_length

        # Bending moment.
        M = (
            RA * x
            - w * np.maximum(x - a, 0.0)**2 / 2.0
            + w * np.maximum(x - b_load, 0.0)**2 / 2.0
        )

        # Deflection using the integrated bending-moment
        # equation and simply supported boundary conditions.
        #
        # M(x) = RA*x
        #        - w*<x-a>^2/2
        #        + w*<x-b>^2/2
        #
        # Downward-positive deflection:
        # y'' = -M/(E*I)
        #
        # y(0) = 0 and y(L) = 0.

        G = (
            RA * x**3 / 6.0
            - w * np.maximum(x - a, 0.0)**4 / 24.0
            + w * np.maximum(x - b_load, 0.0)**4 / 24.0
        )

        G_L = (
            RA * L**3 / 6.0
            - w * max(L - a, 0.0)**4 / 24.0
            + w * max(L - b_load, 0.0)**4 / 24.0
        )

        C1 = -G_L / L

        y = -(G + C1 * x) / (E * I)

    # Exact simply supported boundary conditions.
    y[0] = 0.0
    y[-1] = 0.0


else:

    if load_type == "Point Load":

        # Support reactions.
        RA = P
        RB = 0.0

        # Fixed-end reaction moment magnitude.
        M_fixed = P * a

        # Internal shear force.
        V = np.where(
            x < a,
            -P,
            0.0
        )

        # Internal bending moment.
        M = -P * np.maximum(a - x, 0.0)

        # Exact point-load deflection.
        #
        # For x <= a:
        # y = P*x^2*(3a-x)/(6EI)
        #
        # For x > a:
        # y = P*a^2*(3x-a)/(6EI)

        left = x <= a
        right = x > a

        y[left] = (
            P * x[left]**2
            * (3.0 * a - x[left])
            / (6.0 * E * I)
        )

        y[right] = (
            P * a**2
            * (3.0 * x[right] - a)
            / (6.0 * E * I)
        )

    else:

        # Partial/full UDL on a cantilever.
        udl_length = b_load - a
        total_load = w * udl_length

        # Vertical support reaction.
        RA = total_load
        RB = 0.0

        # Fixed-end reaction moment magnitude.
        M_fixed = total_load * (a + b_load) / 2.0

        # Internal shear force.
        V = -w * (
            np.maximum(b_load - x, 0.0)
            - np.maximum(a - x, 0.0)
        )

        # Internal bending moment.
        M = (
            -w * np.maximum(b_load - x, 0.0)**2 / 2.0
            + w * np.maximum(a - x, 0.0)**2 / 2.0
        )

      

        y = (
            w / (E * I)
            * (
                (
                    np.maximum(b_load - x, 0.0)**4
                    - np.maximum(a - x, 0.0)**4
                ) / 24.0
                + (b_load**3 - a**3) * x / 6.0
                - (b_load**4 - a**4) / 24.0
            )
        )

        # Exact fixed-end displacement.
        y[0] = 0.0

M_max = float(np.max(np.abs(M)))
V_max = float(np.max(np.abs(V)))
stress_max = M_max * c / I
deflection_max = float(np.max(np.abs(y)))

total_load = (
    P
    if load_type == "Point Load"
    else w * (b_load - a)
)

st.divider()
st.header("Analysis Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Maximum Moment",
    f"{M_max:.3f} N·m"
)

col2.metric(
    "Maximum Shear",
    f"{V_max:.3f} N"
)

col3.metric(
    "Maximum Bending Stress",
    f"{stress_max / 1e6:.4f} MPa"
)

col4.metric(
    "Maximum Deflection",
    f"{deflection_max * 1000:.4f} mm"
)

st.subheader("Support Reactions")

col1, col2 = st.columns(2)

if beam_type == "Simply Supported":

    col1.metric("Left Reaction, RA", f"{RA:.3f} N")
    col2.metric("Right Reaction, RB", f"{RB:.3f} N")

else:

    col1.metric("Vertical Reaction", f"{RA:.3f} N")
    col2.metric("Fixed-End Moment", f"{M_fixed:.3f} N·m")

st.subheader("Loading Information")

st.write(f"**Load type:** {load_type}")

if load_type == "Point Load":

    st.write(f"**Applied point load:** {P:.3f} N")
    st.write(f"**Load position:** {a:.3f} m")

else:

    st.write(f"**UDL intensity:** {w:.3f} N/m")
    st.write(f"**Loaded region:** {a:.3f} m to {b_load:.3f} m")
    st.write(f"**Total applied load:** {total_load:.3f} N")

st.caption(
    "Dashboard values show magnitudes. Shear-force and "
    "bending-moment diagrams use the sign conventions "
    "described in the calculation sections."
)

st.divider()
st.header("Beam Visualization")

fig, ax = plt.subplots(figsize=(11, 3))

# Beam.
ax.plot(
    [0, L],
    [0, 0],
    linewidth=4,
    color="black"
)

# Supports.
if beam_type == "Simply Supported":

    ax.scatter(
        [0, L],
        [0, 0],
        marker="^",
        s=180,
        color="black"
    )

else:

    ax.plot(
        [0, 0],
        [-0.25, 0.25],
        linewidth=5,
        color="black"
    )

    for yy in np.linspace(-0.22, 0.22, 7):

        ax.plot(
            [-0.12, 0],
            [yy - 0.06, yy],
            linewidth=1,
            color="black"
        )

# Draw point load.
if load_type == "Point Load":

    if P > 0:

        ax.annotate(
            "",
            xy=(a, -0.04),
            xytext=(a, 0.55),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="red"
            )
        )

        ax.text(
            a,
            0.62,
            f"P = {P:.1f} N",
            ha="center",
            color="red"
        )

# Draw UDL.
else:

    if w > 0:

        arrow_positions = np.linspace(a, b_load, 15)

        for pos in arrow_positions:

            ax.annotate(
                "",
                xy=(pos, -0.04),
                xytext=(pos, 0.36),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.2,
                    color="red"
                )
            )

        ax.plot(
            [a, b_load],
            [0.36, 0.36],
            color="red",
            linewidth=2
        )

        ax.text(
            (a + b_load) / 2.0,
            0.44,
            f"UDL = {w:.1f} N/m",
            ha="center",
            color="red"
        )

ax.text(0, -0.32, "A", ha="center")
ax.text(L, -0.32, "B", ha="center")

ax.set_xlim(-0.25, L + 0.25)
ax.set_ylim(-0.55, 0.85)
ax.set_xlabel("Beam Length (m)")
ax.set_yticks([])
ax.set_title(f"{beam_type} Beam — {load_type}")
ax.grid(axis="x", alpha=0.3)

st.pyplot(fig)
plt.close(fig)


st.divider()
st.header("Shear Force Diagram (SFD)")

fig, ax = plt.subplots(figsize=(11, 3.5))

if load_type == "Point Load":

    # A point load creates a shear-force jump.
    ax.step(
        x,
        V,
        where="post",
        linewidth=2,
        label="Shear force"
    )

else:

    ax.plot(
        x,
        V,
        linewidth=2,
        label="Shear force"
    )

ax.fill_between(x, V, 0, alpha=0.2)
ax.axhline(0, linewidth=1, color="black")

ax.set_xlabel("Distance Along Beam (m)")
ax.set_ylabel("Shear Force (N)")
ax.set_title("Shear Force Diagram")
ax.grid(True)
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.header("Bending Moment Diagram (BMD)")

fig, ax = plt.subplots(figsize=(11, 3.5))

ax.plot(
    x,
    M,
    linewidth=2,
    label="Bending moment"
)

ax.fill_between(x, M, 0, alpha=0.2)
ax.axhline(0, linewidth=1, color="black")

ax.set_xlabel("Distance Along Beam (m)")
ax.set_ylabel("Bending Moment (N·m)")
ax.set_title("Bending Moment Diagram")
ax.grid(True)
ax.legend()

st.pyplot(fig)
plt.close(fig)


st.header("Elastic Deflection Curve")

visual_scale = st.slider(
    "Deflection visualization scale",
    min_value=1,
    max_value=1000,
    value=100,
    step=10
)

fig, ax = plt.subplots(figsize=(11, 3.5))

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
ax.set_title("Elastic Deflection")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "The deflection shape is exaggerated for visibility. "
    "The reported deflection is the calculated physical value."
)

st.divider()
st.header("Engineering Summary")

st.write(f"**Beam type:** {beam_type}")
st.write(f"**Load type:** {load_type}")
st.write(f"**Material:** {material}")
st.write(f"**Young's modulus:** {E / 1e9:.1f} GPa")
st.write(f"**Cross-section:** {b_mm:.1f} mm × {h_mm:.1f} mm")
st.write(f"**Second moment of area:** {I:.6e} m⁴")
st.write(f"**Maximum bending stress:** {stress_max / 1e6:.4f} MPa")
st.write(f"**Maximum deflection:** {deflection_max * 1000:.4f} mm")

st.info(
    "This educational simulator assumes linear-elastic "
    "Euler-Bernoulli beam behavior, small deflections, "
    "ideal supports, and a constant rectangular cross-section. "
    "Material properties are approximate. This application "
    "is not a substitute for professional structural design "
    "or independent safety verification."
)
