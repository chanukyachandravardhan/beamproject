import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="StructLab",
    page_icon="📐",
    layout="wide"
)


def beam_page():
    st.title(" BeamLab_2 ")
    st.subheader("Beam Deflection and Stress Simulator")

    st.write(
        "Analyze beam reactions, shear forces, bending moments, "
        "bending stress, and elastic deflection under point loads, "
        "uniformly distributed loads, linearly varying loads, "
        "and parabolic loads."
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

    # Load type names (defined once so they are not mistyped later).
    POINT = "Point Load"
    UDL = "Uniformly Distributed Load (UDL)"
    LINEAR = "Linearly Varying Load (Triangular / Trapezoidal)"
    PARABOLIC = "Parabolic Load"

    VARYING_TYPES = [LINEAR, PARABOLIC]
    DISTRIBUTED_TYPES = [UDL, LINEAR, PARABOLIC]

    PARABOLA_SHAPES = {
        "Increasing  (zero at start, peak at end)": "increasing",
        "Decreasing  (peak at start, zero at end)": "decreasing",
        "Symmetric  (zero at both ends, peak at middle)": "symmetric",
    }


    def cumtrapz(f, xs):
        """Cumulative trapezoidal integral of f over xs, starting at 0."""
        return np.concatenate(
            [[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))]
        )


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
        [POINT, UDL, LINEAR, PARABOLIC]
    )

    P = 0.0
    w = 0.0
    w_start = 0.0
    w_end = 0.0
    w_peak = 0.0
    parabola_shape = "increasing"
    use_coeffs = False
    coef_A = 0.0
    coef_B = 0.0
    coef_C = 0.0
    a = 0.0
    b_load = L

    if load_type == POINT:

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

        if load_type == UDL:

            w = st.sidebar.number_input(
                "UDL intensity, w (N/m)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=500.0,
                step=50.0,
                key="udl_intensity"
            )

        elif load_type == LINEAR:

            input_mode = st.sidebar.radio(
                "Define linear load by",
                ["End intensities", "Equation coefficients"],
                key="linear_input_mode"
            )
            use_coeffs = input_mode == "Equation coefficients"

            if use_coeffs:

                st.sidebar.markdown("**q(x) = m·x + k**")

                coef_B = st.sidebar.number_input(
                    "Slope, m (N/m per m)",
                    min_value=-1_000_000.0,
                    max_value=1_000_000.0,
                    value=125.0,
                    step=10.0,
                    format="%.4f",
                    key="linear_coef_m"
                )

                coef_C = st.sidebar.number_input(
                    "Constant, k (N/m)",
                    min_value=-1_000_000.0,
                    max_value=1_000_000.0,
                    value=0.0,
                    step=10.0,
                    format="%.4f",
                    key="linear_coef_k"
                )

                st.sidebar.caption(
                    "x is measured in metres from the left end "
                    "(the fixed support for a cantilever). The equation "
                    "applies only between the load start and end positions. "
                    "Negative intensity means an upward load."
                )

            else:

                w_start = st.sidebar.number_input(
                    "Intensity at start, w₁ (N/m)",
                    min_value=0.0,
                    max_value=1_000_000.0,
                    value=0.0,
                    step=50.0,
                    key="linear_start_intensity"
                )

                w_end = st.sidebar.number_input(
                    "Intensity at end, w₂ (N/m)",
                    min_value=0.0,
                    max_value=1_000_000.0,
                    value=500.0,
                    step=50.0,
                    key="linear_end_intensity"
                )

                st.sidebar.caption(
                    "Intensity varies linearly from w₁ to w₂. "
                    "Set w₁ = 0 or w₂ = 0 for a triangular load."
                )

        else:

            input_mode = st.sidebar.radio(
                "Define parabolic load by",
                ["Peak intensity & shape", "Equation coefficients"],
                key="parabolic_input_mode"
            )
            use_coeffs = input_mode == "Equation coefficients"

            if use_coeffs:

                st.sidebar.markdown("**q(x) = A·x² + B·x + C**")

                coef_A = st.sidebar.number_input(
                    "A (N/m per m²)",
                    min_value=-1_000_000.0,
                    max_value=1_000_000.0,
                    value=31.25,
                    step=1.0,
                    format="%.4f",
                    key="parabolic_coef_A"
                )

                coef_B = st.sidebar.number_input(
                    "B (N/m per m)",
                    min_value=-1_000_000.0,
                    max_value=1_000_000.0,
                    value=0.0,
                    step=1.0,
                    format="%.4f",
                    key="parabolic_coef_B"
                )

                coef_C = st.sidebar.number_input(
                    "C (N/m)",
                    min_value=-1_000_000.0,
                    max_value=1_000_000.0,
                    value=0.0,
                    step=10.0,
                    format="%.4f",
                    key="parabolic_coef_C"
                )

                st.sidebar.caption(
                    "x is measured in metres from the left end "
                    "(the fixed support for a cantilever). The equation "
                    "applies only between the load start and end positions. "
                    "Negative intensity means an upward load."
                )

            else:

                w_peak = st.sidebar.number_input(
                    "Peak intensity, w_max (N/m)",
                    min_value=0.0,
                    max_value=1_000_000.0,
                    value=500.0,
                    step=50.0,
                    key="parabolic_peak_intensity"
                )

                shape_label = st.sidebar.selectbox(
                    "Parabola shape",
                    list(PARABOLA_SHAPES.keys())
                )
                parabola_shape = PARABOLA_SHAPES[shape_label]

                st.sidebar.caption(
                    "Increasing: w = w_max·s²  |  "
                    "Decreasing: w = w_max·(1−s)²  |  "
                    "Symmetric: w = 4·w_max·s·(1−s),  "
                    "where s is the fractional position along the loaded region."
                )

        st.sidebar.caption(
            "The distributed load acts downward between its start "
            "and end positions."
        )

        a = st.sidebar.number_input(
            "Load start position (m)",
            min_value=0.0,
            max_value=float(L),
            value=0.0,
            step=0.1,
            key="udl_start"
        )

        b_load = st.sidebar.number_input(
            "Load end position (m)",
            min_value=0.0,
            max_value=float(L),
            value=float(L),
            step=0.1,
            key="udl_end"
        )


    st.sidebar.header("Material")

    material_options = list(MATERIALS.keys()) + ["Custom"]

    material = st.sidebar.selectbox(
        "Select material",
        material_options
    )

    if material == "Custom":
        E_GPa = st.sidebar.number_input(
            "Enter Young's modulus (GPa)",
            min_value=0.001,
            max_value=1_000_000.0,
            value=200.0,
            step=1.0,
            format="%.3f",
            key="custom_youngs_modulus"
        )
        E = E_GPa * 1e9

    else:
        E = MATERIALS[material]
        E_GPa = E / 1e9

    st.sidebar.info(
        f"**Material:** {material}\n\n"
        f"**Young's modulus:** {E_GPa:.3f} GPa\n\n"
        f"**Young's modulus:** {E:.3e} Pa"
    )


    st.sidebar.header("Cross-Section")

    section_type = st.sidebar.selectbox(
        "Select cross-section shape",
        ["Rectangle", "Solid Circle", "Hollow Circle"]
    )

    if section_type == "Rectangle":
        b_mm = st.sidebar.number_input(
            "Width (mm)", min_value=1.0, max_value=1000.0,
            value=50.0, step=5.0, key="section_width"
        )
        h_mm = st.sidebar.number_input(
            "Height (mm)", min_value=1.0, max_value=1000.0,
            value=100.0, step=5.0, key="section_height"
        )

        # Rectangle: I = b*h^3/12; c = h/2.
        b = b_mm / 1000.0
        h = h_mm / 1000.0
        I = b * h**3 / 12.0
        c = h / 2.0
        section_description = f"Rectangle: {b_mm:.1f} mm × {h_mm:.1f} mm"

    elif section_type == "Solid Circle":
        diameter_mm = st.sidebar.number_input(
            "Diameter (mm)", min_value=1.0, max_value=1000.0,
            value=50.0, step=5.0, key="solid_circle_diameter"
        )

        # Solid circle: I = pi*d^4/64; c = d/2.
        diameter = diameter_mm / 1000.0
        I = np.pi * diameter**4 / 64.0
        c = diameter / 2.0
        section_description = f"Solid circle: diameter {diameter_mm:.1f} mm"

    else:
        outer_diameter_mm = st.sidebar.number_input(
            "Outer diameter (mm)", min_value=1.0, max_value=1000.0,
            value=60.0, step=5.0, key="hollow_outer_diameter"
        )
        inner_diameter_mm = st.sidebar.number_input(
            "Inner diameter (mm)", min_value=0.0, max_value=999.0,
            value=30.0, step=5.0, key="hollow_inner_diameter"
        )

        if inner_diameter_mm >= outer_diameter_mm:
            st.error("Inner diameter must be smaller than outer diameter.")
            st.stop()

        # Hollow circle: I = pi*(D^4-d^4)/64; c = D/2.
        outer_diameter = outer_diameter_mm / 1000.0
        inner_diameter = inner_diameter_mm / 1000.0
        I = np.pi * (outer_diameter**4 - inner_diameter**4) / 64.0
        c = outer_diameter / 2.0
        section_description = (
            f"Hollow circle: outer diameter {outer_diameter_mm:.1f} mm, "
            f"inner diameter {inner_diameter_mm:.1f} mm"
        )


    if load_type in DISTRIBUTED_TYPES:

        if b_load <= a:
            st.error(
                "The load end position must be greater than "
                "its start position. Adjust the load positions."
            )
            st.stop()


    def distributed_intensity(xx):
        """
        Downward load intensity q(x) in N/m for any distributed load type.
        Returns 0 outside the loaded region [a, b_load].
        """
        xx = np.asarray(xx, dtype=float)
        inside = (xx >= a) & (xx <= b_load)

        # s = fractional position along the loaded region (0 to 1).
        s = np.clip((xx - a) / (b_load - a), 0.0, 1.0)

        if load_type in VARYING_TYPES and use_coeffs:
            # Equation mode: q(x) = A*x^2 + B*x + C (A = 0 for linear),
            # with x measured from the left end of the beam.
            q = coef_A * xx**2 + coef_B * xx + coef_C

        elif load_type == LINEAR:
            q = w_start + (w_end - w_start) * s

        elif load_type == PARABOLIC:
            if parabola_shape == "increasing":
                q = w_peak * s**2
            elif parabola_shape == "decreasing":
                q = w_peak * (1.0 - s)**2
            else:
                q = 4.0 * w_peak * s * (1.0 - s)

        else:
            q = np.full_like(xx, w)

        return np.where(inside, q, 0.0)


    if load_type in VARYING_TYPES and use_coeffs:

        q_check = distributed_intensity(np.linspace(a, b_load, 400))

        if np.max(np.abs(q_check)) == 0:
            st.warning(
                "The load equation gives zero intensity over the "
                "loaded region, so there is no applied load."
            )
        elif np.min(q_check) < 0:
            st.warning(
                "The load equation gives negative intensity somewhere "
                "in the loaded region. Negative values act upward."
            )

    special_positions = [0.0, float(L)]

    if load_type == POINT:
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
    total_load_numeric = 0.0
    load_centroid = (a + b_load) / 2.0


    if load_type in VARYING_TYPES:

        # Linear and parabolic loads are solved by numerical integration
        # of q(x). Intensity is sampled at interval midpoints, so the
        # load start/end (which are grid nodes) never fall inside an
        # interval and the integration stays accurate.
        dx = np.diff(x)
        x_mid = 0.5 * (x[:-1] + x[1:])
        q_mid = distributed_intensity(x_mid)

        # Q(x): load applied to the left of each section.
        Q = np.concatenate([[0.0], np.cumsum(q_mid * dx)])
        total_load_numeric = float(Q[-1])

        # Centroid (line of action of the resultant) of the load.
        first_moment = float(np.sum(q_mid * x_mid * dx))

        if abs(total_load_numeric) > 1e-12:
            load_centroid = first_moment / total_load_numeric

        if beam_type == "Simply Supported":

            # Reactions from equilibrium.
            # (Moments about A give RB; this also works when the
            # net load is zero but its moment is not.)
            RB = first_moment / L
            RA = total_load_numeric - RB

            # Shear force: V = RA - Q(x).
            V = RA - Q

            # Bending moment: dM/dx = V, with M(0) = 0.
            M = cumtrapz(V, x)

            # Deflection (downward positive): EI*y'' = -M.
            # Integrate twice, then apply y(0) = 0 and y(L) = 0.
            y_raw = -cumtrapz(cumtrapz(M, x), x) / (E * I)
            y = y_raw - y_raw[-1] * x / L

            y[0] = 0.0
            y[-1] = 0.0

        else:

            # Cantilever fixed at x = 0, free at x = L.
            RA = total_load_numeric
            RB = 0.0

            # Shear force: V = -(load to the right of the section).
            V = -(total_load_numeric - Q)

            # Bending moment: dM/dx = V, with M(L) = 0 at the free end.
            M_raw = cumtrapz(V, x)
            M = M_raw - M_raw[-1]

            # Fixed-end reaction moment magnitude.
            M_fixed = float(abs(M[0]))

            # Deflection (downward positive): EI*y'' = -M,
            # with y(0) = 0 and y'(0) = 0 at the fixed support.
            y = cumtrapz(cumtrapz(-M / (E * I), x), x)

            y[0] = 0.0


    elif beam_type == "Simply Supported":

        if load_type == POINT:

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

        if load_type == POINT:

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

    if load_type == POINT:
        total_load = P
    elif load_type in VARYING_TYPES:
        total_load = total_load_numeric
    else:
        total_load = w * (b_load - a)

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

    if load_type == POINT:

        st.write(f"**Applied point load:** {P:.3f} N")
        st.write(f"**Load position:** {a:.3f} m")

    elif load_type == UDL:

        st.write(f"**UDL intensity:** {w:.3f} N/m")
        st.write(f"**Loaded region:** {a:.3f} m to {b_load:.3f} m")
        st.write(f"**Total applied load:** {total_load:.3f} N")

    elif load_type in VARYING_TYPES and use_coeffs:

        if load_type == LINEAR:
            st.write(
                f"**Load equation:** q(x) = {coef_B:.4g}·x + {coef_C:.4g}  (N/m)"
            )
        else:
            st.write(
                f"**Load equation:** q(x) = {coef_A:.4g}·x² + "
                f"{coef_B:.4g}·x + {coef_C:.4g}  (N/m)"
            )

        q_region = distributed_intensity(np.linspace(a, b_load, 400))

        st.write(
            f"**Intensity range over loaded region:** "
            f"{q_region.min():.3f} to {q_region.max():.3f} N/m"
        )
        st.write(f"**Loaded region:** {a:.3f} m to {b_load:.3f} m")
        st.write(f"**Total applied load:** {total_load:.3f} N")

        if abs(total_load) > 1e-12:
            st.write(
                f"**Resultant acts at:** {load_centroid:.3f} m from the left end"
            )

    elif load_type == LINEAR:

        st.write(f"**Intensity at start (w₁):** {w_start:.3f} N/m")
        st.write(f"**Intensity at end (w₂):** {w_end:.3f} N/m")
        st.write(f"**Loaded region:** {a:.3f} m to {b_load:.3f} m")
        st.write(f"**Total applied load:** {total_load:.3f} N")
        st.write(f"**Resultant acts at:** {load_centroid:.3f} m from the left end")

    else:

        st.write(f"**Peak intensity:** {w_peak:.3f} N/m")
        st.write(f"**Parabola shape:** {parabola_shape.capitalize()}")
        st.write(f"**Loaded region:** {a:.3f} m to {b_load:.3f} m")
        st.write(f"**Total applied load:** {total_load:.3f} N")
        st.write(f"**Resultant acts at:** {load_centroid:.3f} m from the left end")

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
    if load_type == POINT:

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

    # Draw distributed loads (UDL, linear, parabolic).
    else:

        envelope_x = np.linspace(a, b_load, 400)
        envelope_q = distributed_intensity(envelope_x)
        q_max = float(np.max(np.abs(envelope_q)))

        if q_max > 0:

            # Arrow heights are proportional to the local intensity.
            envelope_h = 0.36 * envelope_q / q_max

            arrow_positions = np.linspace(a, b_load, 15)
            arrow_heights = 0.36 * distributed_intensity(arrow_positions) / q_max

            for pos, arrow_h in zip(arrow_positions, arrow_heights):

                if abs(arrow_h) > 0.02:

                    ax.annotate(
                        "",
                        xy=(pos, 0.04 if arrow_h < 0 else -0.04),
                        xytext=(pos, arrow_h),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=1.2,
                            color="red"
                        )
                    )

            # Load envelope (flat for UDL, sloped for linear,
            # curved for parabolic).
            ax.plot(
                envelope_x,
                envelope_h,
                color="red",
                linewidth=2
            )

            if load_type in VARYING_TYPES and use_coeffs:
                if load_type == LINEAR:
                    load_label = f"q(x) = {coef_B:.3g}x + {coef_C:.3g}"
                else:
                    load_label = (
                        f"q(x) = {coef_A:.3g}x² + {coef_B:.3g}x + {coef_C:.3g}"
                    )
            elif load_type == UDL:
                load_label = f"UDL = {w:.1f} N/m"
            elif load_type == LINEAR:
                load_label = f"w = {w_start:.1f} → {w_end:.1f} N/m"
            else:
                load_label = f"Parabolic, peak = {w_peak:.1f} N/m"

            ax.text(
                (a + b_load) / 2.0,
                0.44,
                load_label,
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

    if load_type == POINT:

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

    # y is stored downward-positive, so it is negated for plotting
    # to make the beam visibly sag below the original position.
    ax.plot(
        x,
        -y * visual_scale,
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
    st.write(f"**Cross-section:** {section_description}")
    st.write(f"**Second moment of area:** {I:.6e} m⁴")
    st.write(f"**Maximum bending stress:** {stress_max / 1e6:.4f} MPa")
    st.write(f"**Maximum deflection:** {deflection_max * 1000:.4f} mm")

    st.info(
        "This educational simulator assumes linear-elastic "
        "Euler-Bernoulli beam behavior, small deflections, "
        "ideal supports, and a constant cross-section. "
        "Material properties are approximate. This application "
        "is not a substitute for professional structural design "
        "or independent safety verification."
    )


def truss_page():
    st.title("🏗️ 2D Truss / Frame Analysis")
    st.write(
        "Rigid-jointed 2D frame solved by the direct stiffness method. Nodal "
        "forces grow linearly with time, F(t) = rate × t. Each member's stress "
        "is compared with its material's stress-strain curve to find when it "
        "stops being elastic (yields) and when it breaks."
    )

    # E (Pa), yield (Pa), ultimate (Pa), strain at ultimate, strain at fracture.
    # Simplified, approximate values.
    MATS = {
        "Steel": (200e9, 250e6, 400e6, 0.15, 0.25),
        "Cast Iron": (100e9, 130e6, 150e6, 0.005, 0.006),
        "Aluminium": (70e9, 270e6, 310e6, 0.10, 0.12),
        "Copper": (120e9, 70e6, 220e6, 0.35, 0.45),
        "Brass": (100e9, 200e6, 350e6, 0.30, 0.40),
        "Titanium Alloy": (110e9, 880e6, 950e6, 0.10, 0.14),
    }
    SUPPORTS = {
        "Pin": [0, 1],
        "Roller (free in x)": [1],
        "Roller (free in y)": [0],
        "Fixed": [0, 1, 2],
    }

    # ---------------- Sidebar: material and section ----------------
    sb = st.sidebar
    sb.header("Material")
    mat = sb.selectbox("Material", list(MATS), key="t_mat")
    E, sy, su, eu, ef = MATS[mat]
    sb.info(f"E = {E/1e9:.0f} GPa\n\nYield = {sy/1e6:.0f} MPa\n\n"
            f"Ultimate = {su/1e6:.0f} MPa\n\nFracture strain = {ef:.3f}")

    sb.header("Cross-section (all members)")
    sec = sb.selectbox("Shape", ["Rectangle", "Solid Circle", "Hollow Circle"], key="t_sec")
    if sec == "Rectangle":
        bw = sb.number_input("Width (mm)", 1.0, 500.0, 30.0, 1.0, key="t_bw") / 1000
        hh = sb.number_input("Height (mm)", 1.0, 500.0, 50.0, 1.0, key="t_hh") / 1000
        A, I, c = bw * hh, bw * hh**3 / 12, hh / 2
    elif sec == "Solid Circle":
        d = sb.number_input("Diameter (mm)", 1.0, 500.0, 40.0, 1.0, key="t_d") / 1000
        A, I, c = np.pi * d**2 / 4, np.pi * d**4 / 64, d / 2
    else:
        D = sb.number_input("Outer dia. (mm)", 2.0, 500.0, 50.0, 1.0, key="t_D") / 1000
        di = sb.number_input("Inner dia. (mm)", 0.0, 499.0, 30.0, 1.0, key="t_di") / 1000
        if di >= D:
            st.error("Inner diameter must be smaller than outer diameter.")
            st.stop()
        A, I, c = np.pi * (D**2 - di**2) / 4, np.pi * (D**4 - di**4) / 64, D / 2

    # ---------------- Truss definition ----------------
    st.header("1. Define the truss")
    c1, c2, c3 = st.columns(3)
    preset = c1.selectbox("Starting layout", ["Simple triangle", "Warren", "Pratt"], key="t_preset")
    span = c2.number_input("Span (m)", 0.5, 100.0, 4.0, 0.5, key="t_span")
    height = c3.number_input("Height (m)", 0.2, 50.0, 1.5, 0.1, key="t_height")
    n = 2
    if preset != "Simple triangle":
        n = st.slider("Panels", 2, 8, 4, 2, key="t_panels")


    def make_preset(name, L, H, n):
        if name == "Simple triangle":
            nodes = [(0, 0), (L, 0), (L / 2, H)]
            mem = [(0, 1), (1, 2), (2, 0)]
            sup = [(0, "Pin"), (1, "Roller (free in x)")]
            loads = [(2, 0.0, -100.0)]
        elif name == "Warren":
            nodes = [(i * L / n, 0) for i in range(n + 1)]
            nodes += [((i + 0.5) * L / n, H) for i in range(n)]
            mem = [(i, i + 1) for i in range(n)]
            mem += [(n + 1 + i, n + 2 + i) for i in range(n - 1)]
            mem += [(i, n + 1 + i) for i in range(n)]
            mem += [(i + 1, n + 1 + i) for i in range(n)]
            sup = [(0, "Pin"), (n, "Roller (free in x)")]
            loads = [(n + 1 + i, 0.0, -100.0) for i in range(n)]
        else:  # Pratt
            nodes = [(i * L / n, 0) for i in range(n + 1)]
            nodes += [(i * L / n, H) for i in range(n + 1)]
            T = n + 1
            mem = [(i, i + 1) for i in range(n)] + [(T + i, T + i + 1) for i in range(n)]
            mem += [(i, T + i) for i in range(n + 1)]
            mem += [(T + i, i + 1) if i < n / 2 else (T + i + 1, i) for i in range(n)]
            sup = [(0, "Pin"), (n, "Roller (free in x)")]
            loads = [(T + i, 0.0, -100.0) for i in range(1, n)]
        return (
            pd.DataFrame(nodes, columns=["x", "y"]).astype(float),
            pd.DataFrame(mem, columns=["start", "end"]),
            pd.DataFrame(sup, columns=["node", "type"]),
            pd.DataFrame(loads, columns=["node", "Fx", "Fy"]),
        )


    nd0, mb0, sp0, ld0 = make_preset(preset, span, height, n)
    tag = f"{preset}_{span}_{height}_{n}"
    st.caption("Edit the tables to customise. Node numbers are row numbers starting at 0. "
               "Load values are forces per second (N/s); positive Fy is upward.")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Nodes (m)**")
        ned = st.data_editor(nd0, num_rows="dynamic", key="nd_" + tag)
        st.markdown("**Supports**")
        sed = st.data_editor(
            sp0, num_rows="dynamic", key="sp_" + tag,
            column_config={"type": st.column_config.SelectboxColumn(options=list(SUPPORTS))})
    with cb:
        st.markdown("**Members (node numbers)**")
        med = st.data_editor(mb0, num_rows="dynamic", key="mb_" + tag)
        st.markdown("**Nodal load rates (N/s)**")
        led = st.data_editor(ld0, num_rows="dynamic", key="ld_" + tag)

    nodes = ned.dropna().reset_index(drop=True).to_numpy(float)
    mem = med.dropna().astype(int).to_numpy()
    sup = sed.dropna().reset_index(drop=True)
    lds = led.dropna().astype(float).to_numpy()
    N_n, N_m = len(nodes), len(mem)

    if N_m == 0 or N_n < 2:
        st.warning("Add at least 2 nodes and 1 member.")
        st.stop()
    if mem.min() < 0 or mem.max() >= N_n or (mem[:, 0] == mem[:, 1]).any():
        st.error("Members must connect two different, existing nodes.")
        st.stop()
    if (sup["node"].astype(int) >= N_n).any() or (lds[:, 0] >= N_n).any() if len(lds) else False:
        st.error("A support or load refers to a node that does not exist.")
        st.stop()

    fixed = sorted({3 * int(r.node) + k for r in sup.itertuples() for k in SUPPORTS[r.type]})
    r_frame = len(fixed)
    r_pin = sum(min(len(SUPPORTS[t]), 2) for t in sup["type"])
    deg_pin = N_m + r_pin - 2 * N_n
    deg_frame = 3 * N_m + r_frame - 3 * N_n


    def state(v, kind):
        if v < 0:
            return "mechanism (unstable)"
        return "statically determinate" if v == 0 else f"statically indeterminate (degree {v})"


    k1, k2 = st.columns(2)
    k1.metric("Pin-jointed truss check (m + r − 2j)", deg_pin, state(deg_pin, "pin"), delta_color="off")
    k2.metric("Rigid-frame check (3m + r − 3j)", deg_frame, state(deg_frame, "frame"), delta_color="off")
    st.caption("Analysis below uses rigid joints, so a pin-jointed mechanism can still "
               "be solved here as a frame, but it will bend heavily.")

    # ---------------- Solver ----------------
    geo, K = [], np.zeros((3 * N_n, 3 * N_n))
    for i, j in mem:
        dx, dy = nodes[j] - nodes[i]
        Lm = float(np.hypot(dx, dy))
        if Lm < 1e-9:
            st.error("A member has zero length.")
            st.stop()
        cs, sn = dx / Lm, dy / Lm
        a, b = E * A / Lm, E * I
        kl = np.array([
            [a, 0, 0, -a, 0, 0],
            [0, 12 * b / Lm**3, 6 * b / Lm**2, 0, -12 * b / Lm**3, 6 * b / Lm**2],
            [0, 6 * b / Lm**2, 4 * b / Lm, 0, -6 * b / Lm**2, 2 * b / Lm],
            [-a, 0, 0, a, 0, 0],
            [0, -12 * b / Lm**3, -6 * b / Lm**2, 0, 12 * b / Lm**3, -6 * b / Lm**2],
            [0, 6 * b / Lm**2, 2 * b / Lm, 0, -6 * b / Lm**2, 4 * b / Lm]])
        R = np.array([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])
        T = np.zeros((6, 6))
        T[:3, :3], T[3:, 3:] = R, R
        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(dofs, dofs)] += T.T @ kl @ T
        geo.append((Lm, cs, sn, kl, T, dofs))

    F = np.zeros(3 * N_n)
    for nd, fx, fy in lds:
        F[3 * int(nd)] += fx
        F[3 * int(nd) + 1] += fy
    free = [k for k in range(3 * N_n) if k not in fixed]
    if not fixed or np.linalg.matrix_rank(K[np.ix_(free, free)]) < len(free):
        st.error("The structure is unstable (not enough or badly placed supports). "
                 "Add supports or members.")
        st.stop()
    if not np.any(F):
        st.warning("All load rates are zero, so nothing is loaded.")
    u = np.zeros(3 * N_n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    react = (K @ u - F)

    # Per-member results at unit load (1 second). Everything is linear in t.
    xs_list, res, sig_unit = [], [], np.zeros(N_m)
    for k, (Lm, cs, sn, kl, T, dofs) in enumerate(geo):
        dl = T @ u[dofs]
        f = kl @ dl
        xs = np.linspace(0, Lm, 101)
        xi = xs / Lm
        v = (dl[1] * (1 - 3 * xi**2 + 2 * xi**3) + dl[2] * Lm * (xi - 2 * xi**2 + xi**3)
             + dl[4] * (3 * xi**2 - 2 * xi**3) + dl[5] * Lm * (-xi**2 + xi**3))
        ax_u = dl[0] + (dl[3] - dl[0]) * xi
        M = -f[2] + f[1] * xs
        res.append(dict(xs=xs, N=f[3], V=f[1] * np.ones_like(xs), M=M, v=v, ux=ax_u))
        sig_unit[k] = abs(f[3]) / A + np.max(np.abs(M)) * c / I

    with np.errstate(divide="ignore"):
        t_y = np.where(sig_unit > 1e-9, sy / sig_unit, np.inf)
        t_u = np.where(sig_unit > 1e-9, su / sig_unit, np.inf)

    # ---------------- Failure times ----------------
    st.header("2. When does each member yield and break?")
    tab = pd.DataFrame({
        "Member": [f"{i}-{j}" for i, j in mem],
        "Stress per second (MPa/s)": sig_unit / 1e6,
        "Yield time (s)": t_y,
        "Break time (s)": t_u,
    })
    st.dataframe(tab.style.format({"Stress per second (MPa/s)": "{:.4f}",
                                   "Yield time (s)": "{:.2f}", "Break time (s)": "{:.2f}"}),
                 hide_index=True, width="stretch")
    if np.isfinite(t_y).any():
        ky, ku = int(np.argmin(t_y)), int(np.argmin(t_u))
        m1, m2 = st.columns(2)
        m1.metric("First member to yield", f"{mem[ky][0]}-{mem[ky][1]}", f"t = {t_y[ky]:.2f} s", delta_color="off")
        m2.metric("First member to break", f"{mem[ku][0]}-{mem[ku][1]}", f"t = {t_u[ku]:.2f} s", delta_color="off")
        tmax = float(np.max(t_u[np.isfinite(t_u)]) * 1.1)
    else:
        tmax = 1.0
    st.caption("Stress = axial stress + bending stress at the worst point of the member. "
               "Yield time: stress reaches the yield strength. Break time: stress reaches the "
               "ultimate strength (the member can carry no more load). Times come from a linear "
               "analysis of the intact structure; load redistribution after a member fails is "
               "not modelled, and buckling of slender members is not checked.")

    # ---------------- Time explorer ----------------
    st.header("3. Explore the structure at time t")
    t = st.slider("Time t (s)", 0.0, tmax, min(tmax / 2, float(np.min(t_y)) if np.isfinite(t_y).any() else 0.0),
                  tmax / 200, key="t_time")
    sig_t = sig_unit * t
    status = np.where(sig_t >= su, "Broken", np.where(sig_t >= sy, "Yielded", "Elastic"))
    colors = {"Elastic": "tab:green", "Yielded": "tab:orange", "Broken": "tab:red"}

    maxd = float(np.max(np.abs(u.reshape(-1, 3)[:, :2]))) * t
    auto = int(np.clip(0.08 * span / maxd, 1, 5000)) if maxd > 0 else 1
    scale = st.slider("Deformation exaggeration (×)", 1, 5000, auto, key="t_scale")
    if t > np.min(t_y) if np.isfinite(t_y).any() else False:
        st.info("Beyond the first yield time the diagrams are a linear extrapolation "
                "and become less accurate.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, (i, j) in enumerate(mem):
        Lm, cs, sn = geo[k][:3]
        P0 = nodes[i]
        p = P0[:, None] + np.outer([cs, sn], res[k]["xs"])
        ax.plot(*nodes[[i, j]].T, color=colors[status[k]], lw=3,
                ls="--" if status[k] == "Broken" else "-")
        ax.text(*nodes[[i, j]].mean(0), str(k), fontsize=8, ha="center",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
        d = p + scale * t * (np.outer([cs, sn], res[k]["ux"]) + np.outer([-sn, cs], res[k]["v"]))
        ax.plot(*d, color="gray", lw=1.2)
    ax.scatter(*nodes.T, color="black", zorder=5)
    for q, (px, py) in enumerate(nodes):
        ax.text(px, py + 0.04 * span, f"{q}", fontsize=9, color="navy", ha="center")
    for r_ in sup.itertuples():
        ax.scatter(*nodes[int(r_.node)], marker="^", s=160, color="navy", zorder=4)
    ax.set_aspect("equal")
    ax.set_title(f"t = {t:.2f} s   (colour = state, grey = deformed shape)")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for lbl, col in colors.items():
        ax.plot([], [], color=col, lw=3, label=lbl)
    ax.legend(loc="best")
    st.pyplot(fig); plt.close(fig)

    # ---------------- Member diagrams ----------------
    st.header("4. Member diagrams")
    sel = st.selectbox("Member", range(N_m), format_func=lambda k: f"Member {k} (nodes {mem[k][0]}-{mem[k][1]})",
                       key="t_member")
    r = res[sel]
    st.write(f"**State at t = {t:.2f} s:** {status[sel]}  |  stress = {sig_t[sel]/1e6:.2f} MPa  |  "
             f"axial force = {r['N']*t:.2f} N ({'tension' if r['N'] >= 0 else 'compression'})")
    d1, d2, d3 = st.columns(3)
    for col, key, ttl, yl, fac in [(d1, "V", "Shear force", "N", 1), (d2, "M", "Bending moment", "N·m", 1),
                                   (d3, "v", "Transverse deflection", "mm", 1000)]:
        fg, a_ = plt.subplots(figsize=(4, 3))
        yv = r[key] * t * fac
        a_.plot(r["xs"], yv, lw=2)
        a_.fill_between(r["xs"], yv, 0, alpha=0.2)
        a_.axhline(0, color="black", lw=0.8)
        a_.set_title(ttl); a_.set_xlabel("Along member (m)"); a_.set_ylabel(yl); a_.grid(alpha=0.3)
        col.pyplot(fg); plt.close(fg)

    # ---------------- Stress-strain and stress-time ----------------
    st.header("5. Material behaviour")
    g1, g2 = st.columns(2)
    cs_eps = np.array([0, sy / E, eu, ef]); cs_sig = np.array([0, sy, su, 0.85 * su])
    s_now = min(sig_t[sel], su)
    e_now = s_now / E if s_now <= sy else float(np.interp(s_now, [sy, su], [sy / E, eu]))
    fg, a_ = plt.subplots(figsize=(5, 3.5))
    a_.plot(cs_eps, cs_sig / 1e6, lw=2, label=mat)
    a_.scatter([sy / E, eu, ef], [sy / 1e6, su / 1e6, 0.85 * su / 1e6], color="k", zorder=4)
    a_.scatter(e_now, s_now / 1e6, color="red", s=70, zorder=5, label=f"Member {sel} now")
    a_.set_xlabel("Strain"); a_.set_ylabel("Stress (MPa)"); a_.grid(alpha=0.3); a_.legend()
    a_.set_title("Simplified stress-strain curve")
    g1.pyplot(fg); plt.close(fg)

    tt = np.linspace(0, tmax, 200)
    fg, a_ = plt.subplots(figsize=(5, 3.5))
    for k in range(N_m):
        a_.plot(tt, np.minimum(sig_unit[k] * tt, su * 1.05) / 1e6, lw=1, label=f"{k}")
    a_.axhline(sy / 1e6, color="orange", ls="--", label="Yield")
    a_.axhline(su / 1e6, color="red", ls="--", label="Ultimate")
    a_.axvline(t, color="gray", ls=":")
    a_.set_xlabel("Time (s)"); a_.set_ylabel("Max stress (MPa)"); a_.grid(alpha=0.3)
    a_.set_title("Stress vs time, all members")
    if N_m <= 12:
        a_.legend(fontsize=6, ncol=2)
    g2.pyplot(fg); plt.close(fg)

    st.subheader("Support reactions at time t")
    rr = pd.DataFrame([{"Node": int(r_.node), "Rx (N)": react[3 * int(r_.node)] * t,
                        "Ry (N)": react[3 * int(r_.node) + 1] * t,
                        "Mz (N·m)": react[3 * int(r_.node) + 2] * t} for r_ in sup.itertuples()])
    st.dataframe(rr, hide_index=True, width="stretch")
    st.info("Educational tool: linear-elastic Euler-Bernoulli frame analysis with a simplified "
            "material curve. Not a substitute for professional structural design.")


# ---------------- Page selector ----------------
st.sidebar.title("StructLab")
analysis = st.sidebar.radio(
    "Choose analysis",
    ["Beam Analysis", "2D Truss / Frame Analysis"],
    key="analysis_choice"
)
st.sidebar.divider()

if analysis == "Beam Analysis":
    beam_page()
else:
    truss_page()
