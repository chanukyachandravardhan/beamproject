"""
BeamLab 3D - Beam analysis with biaxial bending, axial force and torsion.

Run with:   streamlit run beamlab_3d.py

Coordinate system (right-handed)
    X : along the beam axis
    Y : vertical (up)
    Z : horizontal, points toward the viewer in the default 3D camera

Inputs
    Vertical load  : positive DOWNWARD (-Y)
    Lateral load   : positive toward +Z
    Axial load     : positive toward +X (tension at the free end)
    Torque         : positive by the right-hand rule about +X

Method
    Euler-Bernoulli beam finite elements (cubic Hermite) for each bending
    plane, 1-D bar elements for axial force and for torsion. Shear force and
    bending moment are then recovered exactly from statics, so the diagrams
    are exact for any combination of point loads and UDLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# name: (Young's modulus [GPa], Poisson's ratio [-])
MATERIALS = {
    "Steel": (200.0, 0.30),
    "Cast Iron": (100.0, 0.26),
    "Aluminium": (70.0, 0.33),
    "Copper": (120.0, 0.34),
    "Brass": (100.0, 0.34),
    "Titanium Alloy": (110.0, 0.32),
}

BEAM_TYPES = [
    "Simply Supported",
    "Cantilever",
    "Fixed-Fixed",
    "Propped Cantilever",
    "Overhanging",
]

BEAM_HELP = {
    "Simply Supported": "Pin at x = 0 and roller at x = L.",
    "Cantilever": "Fixed (built-in) at x = 0, free at x = L.",
    "Fixed-Fixed": "Built-in at both ends (statically indeterminate).",
    "Propped Cantilever": "Fixed at x = 0 and a roller prop at x = L.",
    "Overhanging": "Two supports at positions you choose; the ends can overhang.",
}

PT_COLS = [
    "Position x (m)",
    "Vertical ↓ (N)",
    "Lateral Z (N)",
    "Axial X (N)",
    "Torque (N·m)",
]
UDL_COLS = ["Start (m)", "End (m)", "Vertical ↓ (N/m)", "Lateral Z (N/m)"]

C_RED = "#e5484d"
C_BLUE = "#2f7fd1"
C_GREEN = "#2a9d8f"
C_ORANGE = "#f08c00"
C_PURPLE = "#8e5bd6"
C_GREY = "#7f8c8d"


# ----------------------------------------------------------------------------
# Cross-sections
# ----------------------------------------------------------------------------

@dataclass
class Section:
    kind: str
    A: float        # area [m^2]
    Izz: float      # second moment about Z, resists bending in the X-Y plane [m^4]
    Iyy: float      # second moment about Y, resists bending in the X-Z plane [m^4]
    J: float        # torsion constant [m^4]
    cy: float       # half-extent in Y [m]
    cz: float       # half-extent in Z [m]
    tau_factor: float   # tau_max = tau_factor * T  [1/m^3]
    label: str = ""

    def perimeter(self, per_edge: int = 8, n_circle: int = 56):
        """Points (y, z) around the outline of the section."""
        if self.kind == "Circle":
            ang = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
            return self.cy * np.cos(ang), self.cz * np.sin(ang)
        corners = np.array(
            [
                [self.cy, -self.cz],
                [self.cy, self.cz],
                [-self.cy, self.cz],
                [-self.cy, -self.cz],
            ]
        )
        ys, zs = [], []
        t = np.linspace(0.0, 1.0, per_edge, endpoint=False)
        for k in range(4):
            p0, p1 = corners[k], corners[(k + 1) % 4]
            ys.append(p0[0] + t * (p1[0] - p0[0]))
            zs.append(p0[1] + t * (p1[1] - p0[1]))
        return np.concatenate(ys), np.concatenate(zs)


def make_section(kind: str, b_mm: float, h_mm: float, d_mm: float) -> Section:
    if kind == "Rectangle":
        b, h = b_mm / 1000.0, h_mm / 1000.0          # b along Z, h along Y
        long_, short_ = max(b, h), min(b, h)
        J = long_ * short_**3 * (
            1.0 / 3.0 - 0.21 * (short_ / long_) * (1.0 - short_**4 / (12.0 * long_**4))
        )
        tau_factor = (3.0 * long_ + 1.8 * short_) / (long_**2 * short_**2)
        return Section(
            kind="Rectangle",
            A=b * h,
            Izz=b * h**3 / 12.0,
            Iyy=h * b**3 / 12.0,
            J=J,
            cy=h / 2.0,
            cz=b / 2.0,
            tau_factor=tau_factor,
            label=f"Rectangle {b_mm:g} × {h_mm:g} mm (width Z × height Y)",
        )
    d = d_mm / 1000.0
    I = np.pi * d**4 / 64.0
    return Section(
        kind="Circle",
        A=np.pi * d**2 / 4.0,
        Izz=I,
        Iyy=I,
        J=np.pi * d**4 / 32.0,
        cy=d / 2.0,
        cz=d / 2.0,
        tau_factor=16.0 / (np.pi * d**3),
        label=f"Circle Ø {d_mm:g} mm",
    )


# ----------------------------------------------------------------------------
# Model and solver
# ----------------------------------------------------------------------------

@dataclass
class Model:
    L: float
    pinned: list            # support positions with deflection restrained
    fixed: list             # support positions with deflection + rotation restrained
    axial_restraints: list  # positions where axial motion is restrained
    E: float
    G: float
    sec: Section
    points: list = field(default_factory=list)  # (x, Fv_down, Fz, Fx, T)
    udls: list = field(default_factory=list)    # (a, b, wv_down, wz)


def supports_for(beam_type: str, L: float, s1: float = 0.0, s2: float = 0.0):
    """Return (pinned, fixed, axial_restraints) for a beam type."""
    if beam_type == "Simply Supported":
        return [0.0, L], [], [0.0]
    if beam_type == "Cantilever":
        return [], [0.0], [0.0]
    if beam_type == "Fixed-Fixed":
        return [], [0.0, L], [0.0, L]
    if beam_type == "Propped Cantilever":
        return [L], [0.0], [0.0]
    return [s1, s2], [], [min(s1, s2)]  # Overhanging


def _node_index(nodes, x):
    return int(np.argmin(np.abs(nodes - x)))


def solve_transverse(nodes, EI, pinned, fixed, pts, udls):
    """
    Cubic-Hermite beam FE for one bending plane.
    pts : [(x, F)]   forces along the +direction of the plane
    udls: [(a, b, q)] distributed load along the +direction
    Returns nodal dofs [v, theta,...], reaction forces and reaction moments.
    """
    n = len(nodes)
    nd = 2 * n
    K = np.zeros((nd, nd))
    F = np.zeros(nd)
    for e in range(n - 1):
        l = nodes[e + 1] - nodes[e]
        k = (EI / l**3) * np.array(
            [
                [12.0, 6 * l, -12.0, 6 * l],
                [6 * l, 4 * l**2, -6 * l, 2 * l**2],
                [-12.0, -6 * l, 12.0, -6 * l],
                [6 * l, 2 * l**2, -6 * l, 4 * l**2],
            ]
        )
        s = slice(2 * e, 2 * e + 4)
        K[s, s] += k
    for x, f in pts:
        F[2 * _node_index(nodes, x)] += f
    for a, b, q in udls:
        for e in range(n - 1):
            mid = 0.5 * (nodes[e] + nodes[e + 1])
            if a < mid < b:
                l = nodes[e + 1] - nodes[e]
                F[2 * e: 2 * e + 4] += q * l / 2.0 * np.array([1.0, l / 6.0, 1.0, -l / 6.0])

    fixed_dofs = []
    for p in pinned:
        fixed_dofs.append(2 * _node_index(nodes, p))
    for p in fixed:
        i = _node_index(nodes, p)
        fixed_dofs += [2 * i, 2 * i + 1]
    fixed_dofs = np.unique(fixed_dofs)
    free = np.setdiff1d(np.arange(nd), fixed_dofs)

    u = np.zeros(nd)
    try:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    except np.linalg.LinAlgError as exc:
        raise ValueError("The beam is unstable with these supports.") from exc

    R = K @ u - F
    forces = [(nodes[d // 2], R[d]) for d in fixed_dofs if d % 2 == 0]
    moments = [(nodes[d // 2], R[d]) for d in fixed_dofs if d % 2 == 1]
    return u, forces, moments


def solve_bar(nodes, stiffness, restrained, pts):
    """1-D bar / shaft FE (axial force or torsion). Returns nodal values and reactions."""
    n = len(nodes)
    K = np.zeros((n, n))
    F = np.zeros(n)
    for e in range(n - 1):
        l = nodes[e + 1] - nodes[e]
        k = stiffness / l * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K[e: e + 2, e: e + 2] += k
    for x, f in pts:
        F[_node_index(nodes, x)] += f
    r_idx = np.unique([_node_index(nodes, p) for p in restrained])
    free = np.setdiff1d(np.arange(n), r_idx)
    u = np.zeros(n)
    if len(free):
        try:
            u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        except np.linalg.LinAlgError as exc:
            raise ValueError("Axial / torsional restraint is insufficient.") from exc
    R = K @ u - F
    reactions = [(nodes[i], R[i]) for i in r_idx]
    return u, reactions


def hermite(xq, nodes, v, th):
    e = np.clip(np.searchsorted(nodes, xq, side="right") - 1, 0, len(nodes) - 2)
    x0 = nodes[e]
    l = nodes[e + 1] - x0
    s = (xq - x0) / l
    h1 = 1 - 3 * s**2 + 2 * s**3
    h2 = l * (s - 2 * s**2 + s**3)
    h3 = 3 * s**2 - 2 * s**3
    h4 = l * (-(s**2) + s**3)
    return h1 * v[e] + h2 * th[e] + h3 * v[e + 1] + h4 * th[e + 1]


def transverse_diagrams(xq, pts, udls, rforces, rmoments, tol):
    """Shear and bending moment (sagging positive) from statics."""
    V = np.zeros_like(xq)
    M = np.zeros_like(xq)
    for xi, f in list(pts) + list(rforces):
        mask = (xi <= xq + tol).astype(float)
        V += f * mask
        M += f * (xq - xi) * mask
    for xi, c in rmoments:
        M -= c * (xi <= xq + tol)
    for a, b, q in udls:
        V += q * np.clip(xq - a, 0.0, b - a)
        M += q * (np.maximum(xq - a, 0.0) ** 2 - np.maximum(xq - b, 0.0) ** 2) / 2.0
    return V, M


def bar_diagram(xq, pts, reactions, tol):
    """Internal axial force or torque (tension / positive-outward-normal torque)."""
    out = np.zeros_like(xq)
    for xi, f in list(pts) + list(reactions):
        out -= f * (xi <= xq + tol)
    return out


class Solution:
    """Container with the solved model and an evaluator for any station x."""

    def __init__(self, model: Model):
        m = model
        self.model = m
        L = m.L
        self.eps = 1e-6 * L
        self.tol = 0.5 * self.eps

        rnd = lambda v: round(float(v), 9)  # noqa: E731
        m.points = [(rnd(p[0]), *p[1:]) for p in m.points]
        m.udls = [(rnd(u[0]), rnd(u[1]), *u[2:]) for u in m.udls]
        m.pinned = [rnd(p) for p in m.pinned]
        m.fixed = [rnd(p) for p in m.fixed]
        m.axial_restraints = [rnd(p) for p in m.axial_restraints]

        specials = [0.0, L, *m.pinned, *m.fixed]
        specials += [p[0] for p in m.points]
        specials += [u[0] for u in m.udls] + [u[1] for u in m.udls]
        specials = np.unique(np.round(specials, 9))
        grid = np.linspace(0.0, L, 301)
        keep = np.min(np.abs(grid[:, None] - specials[None, :]), axis=1) > 2e-3 * L
        self.nodes = np.sort(np.concatenate([specials, grid[keep]]))
        self.specials = specials

        # stations used for diagrams: nodes plus a point just before each special
        before = np.clip(specials - self.eps, 0.0, L)
        self.xg = np.unique(np.concatenate([self.nodes, before]))

        # load lists by plane
        self.pts_y = [(p[0], -p[1]) for p in m.points if p[1] != 0.0]
        self.pts_z = [(p[0], p[2]) for p in m.points if p[2] != 0.0]
        self.pts_x = [(p[0], p[3]) for p in m.points if p[3] != 0.0]
        self.pts_t = [(p[0], p[4]) for p in m.points if p[4] != 0.0]
        self.udl_y = [(u[0], u[1], -u[2]) for u in m.udls if u[2] != 0.0]
        self.udl_z = [(u[0], u[1], u[3]) for u in m.udls if u[3] != 0.0]

        # bending in X-Y (loads Fy, stiffness E*Izz) and in X-Z (loads Fz, E*Iyy)
        self.uy, self.ry, self.cy = solve_transverse(
            self.nodes, m.E * m.sec.Izz, m.pinned, m.fixed, self.pts_y, self.udl_y
        )
        self.uz, self.rz, self.cz = solve_transverse(
            self.nodes, m.E * m.sec.Iyy, m.pinned, m.fixed, self.pts_z, self.udl_z
        )
        # axial force and torsion
        self.ux, self.rx = solve_bar(
            self.nodes, m.E * m.sec.A, m.axial_restraints, self.pts_x
        )
        torsion_restraints = list(m.pinned) + list(m.fixed)
        self.phi, self.rt = solve_bar(
            self.nodes, m.G * m.sec.J, torsion_restraints, self.pts_t
        )

    def at(self, xq):
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        Vy, Mxy = transverse_diagrams(xq, self.pts_y, self.udl_y, self.ry, self.cy, self.tol)
        Vz, Mxz = transverse_diagrams(xq, self.pts_z, self.udl_z, self.rz, self.cz, self.tol)
        N = bar_diagram(xq, self.pts_x, self.rx, self.tol)
        T = bar_diagram(xq, self.pts_t, self.rt, self.tol)
        n = self.nodes
        return {
            "x": xq,
            "Vy": Vy,
            "Vz": Vz,
            "Mxy": Mxy,
            "Mxz": Mxz,
            "N": N,
            "T": T,
            "v": hermite(xq, n, self.uy[0::2], self.uy[1::2]),
            "w": hermite(xq, n, self.uz[0::2], self.uz[1::2]),
            "u": np.interp(xq, n, self.ux),
            "phi": np.interp(xq, n, self.phi),
        }

    def stress_field(self, ev):
        """Normal stress sigma_x at perimeter points for each station [Pa]."""
        s = self.model.sec
        yp, zp = s.perimeter()
        sig = (
            ev["N"][:, None] / s.A
            - ev["Mxy"][:, None] * yp[None, :] / s.Izz
            - ev["Mxz"][:, None] * zp[None, :] / s.Iyy
        )
        return sig, yp, zp

    def summary(self):
        ev = self.at(self.xg)
        sig, _, _ = self.stress_field(ev)
        sec = self.model.sec
        sig_abs = np.abs(sig)
        k = np.unravel_index(np.argmax(sig_abs), sig_abs.shape)
        sig_station = sig_abs.max(axis=1)
        tau = sec.tau_factor * np.abs(ev["T"])
        vm = np.sqrt(sig_station**2 + 3.0 * tau**2)
        defl = np.sqrt(ev["v"] ** 2 + ev["w"] ** 2)
        return {
            "ev": ev,
            "sigma_max": float(sig_abs[k]),
            "sigma_x_at": float(ev["x"][k[0]]),
            "vm_max": float(vm.max()),
            "vm_at": float(ev["x"][int(np.argmax(vm))]),
            "tau_max": float(tau.max()),
            "defl_max": float(defl.max()),
            "defl_at": float(ev["x"][int(np.argmax(defl))]),
            "v_max": float(np.abs(ev["v"]).max()),
            "w_max": float(np.abs(ev["w"]).max()),
            "u_max": float(np.abs(ev["u"]).max()),
            "phi_max": float(np.abs(ev["phi"]).max()),
            "Vy_max": float(np.abs(ev["Vy"]).max()),
            "Vz_max": float(np.abs(ev["Vz"]).max()),
            "Mxy_max": float(np.abs(ev["Mxy"]).max()),
            "Mxz_max": float(np.abs(ev["Mxz"]).max()),
            "N_max": float(np.abs(ev["N"]).max()),
            "T_max": float(np.abs(ev["T"]).max()),
        }

    def reaction_table(self):
        positions = sorted(set(self.model.pinned) | set(self.model.fixed))

        def val(lst, x):
            return sum(f for xi, f in lst if abs(xi - x) < 1e-7)

        rows = []
        for x in positions:
            rows.append(
                {
                    "Support": "Fixed" if x in self.model.fixed else "Pin / roller",
                    "x (m)": x,
                    "Ry (N)": val(self.ry, x),
                    "Rz (N)": val(self.rz, x),
                    "Rx (N)": val(self.rx, x),
                    "M about Z (N·m)": val(self.cy, x),
                    "M about Y (N·m)": -val(self.cz, x),
                    "Torque Tx (N·m)": val(self.rt, x),
                }
            )
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plotly figures
# ----------------------------------------------------------------------------

def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def line_fig(title, x, series, ytitle, supports, height=320, fill=True):
    fig = go.Figure()
    for name, y, color in series:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines",
                line=dict(width=2.6, color=color),
                fill="tozeroy" if fill else None,
                fillcolor=rgba(color, 0.14),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=supports,
            y=[0.0] * len(supports),
            mode="markers",
            name="Supports",
            marker=dict(symbol="triangle-up", size=11, color=C_GREY, line=dict(width=1, color="white")),
            hovertemplate="Support at x = %{x:.3f} m<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.14, x=1.0, xanchor="right"),
        xaxis_title="Position along beam, x (m)",
        yaxis_title=ytitle,
    )
    return fig


def P(v):
    """Physical (X, Y, Z) -> plot coordinates (right-handed, Y up)."""
    return (v[0], -v[2], v[1])


class Arrows:
    def __init__(self, name, color, head):
        self.name, self.color, self.head = name, color, head
        self.lx, self.ly, self.lz = [], [], []
        self.cx, self.cy, self.cz = [], [], []
        self.du, self.dv, self.dw = [], [], []
        self.tx, self.ty, self.tz, self.tt = [], [], [], []

    def _line(self, pts):
        for p in pts:
            q = P(p)
            self.lx.append(q[0])
            self.ly.append(q[1])
            self.lz.append(q[2])
        self.lx.append(None)
        self.ly.append(None)
        self.lz.append(None)

    def _head(self, tip, direction):
        d = np.asarray(direction, float)
        n = np.linalg.norm(d)
        if n == 0:
            return
        d = d / n
        tip = np.asarray(tip, float)
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        p1 = np.cross(d, ref)
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(d, p1)
        h = self.head
        for p in (p1, -p1, p2, -p2):
            self._line([tip, tip - d * h + p * 0.4 * h])

    def _label(self, pos, text):
        if text:
            q = P(pos)
            self.tx.append(q[0])
            self.ty.append(q[1])
            self.tz.append(q[2])
            self.tt.append(text)

    def arrow(self, tail, tip, label=None):
        tail, tip = np.asarray(tail, float), np.asarray(tip, float)
        self._line([tail, tip])
        self._head(tip, tip - tail)
        self._label(tail, label)

    def path(self, pts, label=None):
        pts = [np.asarray(p, float) for p in pts]
        self._line(pts)
        self._head(pts[-1], pts[-1] - pts[-2])
        self._label(pts[0], label)

    def traces(self):
        out = []
        if self.lx:
            out.append(
                go.Scatter3d(
                    x=self.lx, y=self.ly, z=self.lz, mode="lines",
                    line=dict(color=self.color, width=6),
                    name=self.name, legendgroup=self.name, hoverinfo="skip",
                )
            )
        if self.tx:
            out.append(
                go.Scatter3d(
                    x=self.tx, y=self.ty, z=self.tz, mode="text", text=self.tt,
                    textfont=dict(color=self.color, size=12),
                    legendgroup=self.name, showlegend=False, hoverinfo="skip",
                )
            )
        return out


def fmt(v, unit="", digits=4):
    return f"{v:.{digits}g}{(' ' + unit) if unit else ''}"


def build_3d(model: Model, sol: Solution, color_by, exag_pct, true_scale,
             show_loads, show_supports, show_axis):
    L, sec = model.L, model.sec
    # stop just short of x = L so an end reaction is not counted as "left of the cut"
    xs = np.linspace(0.0, L - sol.eps, 161)
    ev = sol.at(xs)
    disp =np.sqrt(ev["u"] ** 2 + ev["v"] ** 2 + ev["w"] ** 2)
    dmax = float(disp.max())
    phimax = float(np.abs(ev["phi"]).max())

    if true_scale:
        sd = st_ = 1.0
    else:
        f = exag_pct / 100.0
        sd = f * 0.25 * L / dmax if dmax > 1e-15 else 1.0
        st_ = f * np.radians(60.0) / phimax if phimax > 1e-15 else 1.0

    dim = 2.0 * max(sec.cy, sec.cz)
    sv = max(1.0, (L / 12.0) / dim)   # visual enlargement of the cross-section only

    yp, zp = sec.perimeter()
    npp, ns = len(yp), len(xs)

    cX = xs + ev["u"] * sd
    cY = ev["v"] * sd
    cZ = ev["w"] * sd
    th = ev["phi"] * st_
    cos_t, sin_t = np.cos(th)[:, None], np.sin(th)[:, None]
    Yv = cY[:, None] + (yp[None, :] * cos_t - zp[None, :] * sin_t) * sv
    Zv = cZ[:, None] + (yp[None, :] * sin_t + zp[None, :] * cos_t) * sv
    Xv = np.repeat(cX[:, None], npp, axis=1)

    if color_by == "Normal stress σx":
        sig, _, _ = sol.stress_field(ev)
        vals = sig / 1e6
        center_vals = ev["N"] / sec.A / 1e6
        lim = float(np.abs(vals).max())
        lim = lim if lim > 1e-12 else 1.0
        cmin, cmax, cscale = -lim, lim, "RdBu_r"
        cbar_title, hover = "σx (MPa)", "σx = %{intensity:.4g} MPa<extra></extra>"
    else:
        vals = np.repeat((disp * 1000.0)[:, None], npp, axis=1)
        center_vals = disp * 1000.0
        hi = float(vals.max())
        hi = hi if hi > 1e-12 else 1.0
        cmin, cmax, cscale = 0.0, hi, "Turbo"
        cbar_title, hover = "|displacement| (mm)", "disp = %{intensity:.4g} mm<extra></extra>"

    px = np.concatenate([Xv.ravel(), [cX[0], cX[-1]]])
    py = np.concatenate([(-Zv).ravel(), [-cZ[0], -cZ[-1]]])
    pz = np.concatenate([Yv.ravel(), [cY[0], cY[-1]]])
    inten = np.concatenate([vals.ravel(), [center_vals[0], center_vals[-1]]])

    i = np.arange(ns - 1)[:, None]
    j = np.arange(npp)[None, :]
    a = i * npp + j
    b = i * npp + (j + 1) % npp
    c = (i + 1) * npp + j
    d = (i + 1) * npp + (j + 1) % npp
    tri = np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)]
    )
    jj = np.arange(npp)
    cap0 = np.stack([np.full(npp, ns * npp), jj, (jj + 1) % npp], -1)
    cap1 = np.stack(
        [np.full(npp, ns * npp + 1), (ns - 1) * npp + jj, (ns - 1) * npp + (jj + 1) % npp], -1
    )
    tri = np.concatenate([tri, cap0, cap1])

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=px, y=py, z=pz,
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            intensity=inten, intensitymode="vertex",
            colorscale=cscale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=cbar_title, thickness=14, len=0.7),
            lighting=dict(ambient=0.62, diffuse=0.85, specular=0.25, roughness=0.55),
            flatshading=False, hovertemplate=hover, name="Beam", showlegend=False,
        )
    )

    if show_axis:
        fig.add_trace(
            go.Scatter3d(
                x=[0, L], y=[0, 0], z=[0, 0], mode="lines",
                line=dict(color=C_GREY, width=3, dash="dash"),
                name="Undeformed axis", hoverinfo="skip",
            )
        )

    def cpos(x):
        e = sol.at([x])
        return np.array([x + e["u"][0] * sd, e["v"][0] * sd, e["w"][0] * sd])

    arrow_len = L * 0.13
    head = arrow_len * 0.38
    top = sec.cy * sv
    side = sec.cz * sv

    if show_supports:
        wall_q = 1.6 * max(sec.cy, sec.cz) * sv
        fig_added_fixed = False
        for x in model.fixed:
            xw = x - 0.012 * L if x <= 0.5 * L else x + 0.012 * L
            verts = np.array(
                [[xw, -wall_q, -wall_q], [xw, wall_q, -wall_q], [xw, wall_q, wall_q], [xw, -wall_q, wall_q]]
            )
            pv = np.array([P(v) for v in verts])
            fig.add_trace(
                go.Mesh3d(
                    x=pv[:, 0], y=pv[:, 1], z=pv[:, 2],
                    i=[0, 0], j=[1, 2], k=[2, 3], color="#455a64", opacity=0.75,
                    name="Fixed support", legendgroup="Fixed support",
                    showlegend=not fig_added_fixed, hoverinfo="skip",
                )
            )
            fig_added_fixed = True
        if model.pinned:
            pp = [P(cpos(x) + np.array([0.0, -top * 1.25, 0.0])) for x in model.pinned]
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0] for p in pp], y=[p[1] for p in pp], z=[p[2] for p in pp],
                    mode="markers",
                    marker=dict(symbol="diamond", size=8, color=C_GREEN, line=dict(width=1, color="white")),
                    name="Pin / roller support",
                    hovertemplate="Support<extra></extra>",
                )
            )

    if show_loads:
        force = Arrows("Transverse loads", C_RED, head)
        axial = Arrows("Axial loads", C_ORANGE, head)
        torque = Arrows("Torque", C_PURPLE, head)

        def transverse_arrow(c, fv_axis, fz_axis, length, label_y, label_z):
            if fv_axis != 0:
                dvec = np.array([0.0, np.sign(fv_axis), 0.0])
                tip = c - dvec * top
                force.arrow(tip - dvec * length, tip, label_y)
            if fz_axis != 0:
                dvec = np.array([0.0, 0.0, np.sign(fz_axis)])
                tip = c - dvec * side
                force.arrow(tip - dvec * length, tip, label_z)

        for x, fv, fz, fx, t in model.points:
            c = cpos(x)
            transverse_arrow(
                c, -fv, fz, arrow_len,
                f"{abs(fv):,.4g} N" if fv else None,
                f"{abs(fz):,.4g} N" if fz else None,
            )
            if fx != 0:
                dvec = np.array([np.sign(fx), 0.0, 0.0])
                base = c + np.array([0.0, top * 1.7, 0.0])
                axial.arrow(base - dvec * arrow_len * 0.5, base + dvec * arrow_len * 0.5,
                            f"{abs(fx):,.4g} N")
            if t != 0:
                R = 1.5 * max(sec.cy, sec.cz) * sv
                ang = np.linspace(np.radians(30), np.radians(300), 36) * np.sign(t)
                arc = [c + np.array([0.0, R * np.cos(a_), R * np.sin(a_)]) for a_ in ang]
                torque.path(arc, f"{abs(t):,.4g} N·m")

        for a_, b_, wv, wz in model.udls:
            n_arr = int(np.clip(round((b_ - a_) / L * 18), 3, 24))
            xu = np.linspace(a_, b_, n_arr)
            small = arrow_len * 0.75
            tails_y, tails_z = [], []
            for k, xx in enumerate(xu):
                c = cpos(xx)
                mid = k == n_arr // 2
                transverse_arrow(
                    c, -wv, wz, small,
                    f"{abs(wv):,.4g} N/m" if (mid and wv) else None,
                    f"{abs(wz):,.4g} N/m" if (mid and wz) else None,
                )
                if wv:
                    tails_y.append(c - np.array([0.0, np.sign(-wv), 0.0]) * (top + small))
                if wz:
                    tails_z.append(c - np.array([0.0, 0.0, np.sign(wz)]) * (side + small))
            for tails in (tails_y, tails_z):
                if len(tails) > 1:
                    force._line(tails)
        for arr in (force, axial, torque):
            for tr in arr.traces():
                fig.add_trace(tr)

    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=0.02, x=0.01),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X  (beam axis, m)"),
            yaxis=dict(title="Z", showticklabels=False),
            zaxis=dict(title="Y  (up, m)"),
            camera=dict(eye=dict(x=1.15, y=-1.7, z=0.85)),
        ),
    )
    return fig, sd, st_, sv


# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------

def clean_points(df, L, errors):
    rows = []
    d = df.dropna(how="all")
    for n, (_, r) in enumerate(d.iterrows(), start=1):
        x = pd.to_numeric(r[PT_COLS[0]], errors="coerce")
        vals = [pd.to_numeric(r[c], errors="coerce") for c in PT_COLS[1:]]
        vals = [0.0 if pd.isna(v) else float(v) for v in vals]
        if pd.isna(x):
            errors.append(f"Point loads, row {n}: the position is missing.")
            continue
        if not 0.0 <= x <= L:
            errors.append(f"Point loads, row {n}: position {x:g} m is outside the beam (0 to {L:g} m).")
            continue
        if all(v == 0.0 for v in vals):
            continue
        rows.append((float(x), *vals))
    return rows


def clean_udls(df, L, errors):
    rows = []
    d = df.dropna(how="all")
    for n, (_, r) in enumerate(d.iterrows(), start=1):
        a = pd.to_numeric(r[UDL_COLS[0]], errors="coerce")
        b = pd.to_numeric(r[UDL_COLS[1]], errors="coerce")
        vals = [pd.to_numeric(r[c], errors="coerce") for c in UDL_COLS[2:]]
        vals = [0.0 if pd.isna(v) else float(v) for v in vals]
        if pd.isna(a) or pd.isna(b):
            errors.append(f"Distributed loads, row {n}: start and end positions are required.")
            continue
        if not (0.0 <= a < b <= L):
            errors.append(
                f"Distributed loads, row {n}: need 0 ≤ start < end ≤ {L:g} m (got {a:g} and {b:g})."
            )
            continue
        if all(v == 0.0 for v in vals):
            continue
        rows.append((float(a), float(b), *vals))
    return rows


CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.hero {
    background: linear-gradient(120deg, #0f2b46 0%, #1d5c8f 55%, #2a9d8f 100%);
    padding: 1.3rem 1.8rem; border-radius: 16px; margin-bottom: 1rem;
    box-shadow: 0 6px 22px rgba(15, 43, 70, 0.25);
}
.hero h1 {margin: 0; padding: 0; font-size: 2rem; color: #ffffff; letter-spacing: .3px;}
.hero p {margin: .35rem 0 0 0; color: rgba(255,255,255,.88); font-size: 1rem;}
.chips {margin-top: .7rem;}
.chip {
    display: inline-block; padding: .18rem .7rem; margin: 0 .4rem .3rem 0;
    border-radius: 999px; background: rgba(255,255,255,.16); color: #fff;
    font-size: .82rem; border: 1px solid rgba(255,255,255,.28);
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px; padding: .7rem .9rem;
}
div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: .5rem 1.1rem;}
.sectiontitle {font-size: 1.05rem; font-weight: 700; margin: 1rem 0 .3rem 0;}
</style>
"""


def sidebar_inputs():
    sb = st.sidebar
    sb.markdown("## 🏗️ Beam")
    beam_type = sb.selectbox("Support condition", BEAM_TYPES)
    sb.caption(BEAM_HELP[beam_type])
    L = sb.number_input("Beam length (m)", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
    s1 = s2 = 0.0
    if beam_type == "Overhanging":
        s1 = sb.number_input("Support 1 position (m)", min_value=0.0, max_value=float(L),
                             value=round(0.1 * L, 3), step=0.1)
        s2 = sb.number_input("Support 2 position (m)", min_value=0.0, max_value=float(L),
                             value=round(0.8 * L, 3), step=0.1)

    sb.markdown("## 🧱 Material")
    mats = list(MATERIALS) + ["Custom"]
    material = sb.selectbox("Material", mats)
    if material == "Custom":
        E_GPa = sb.number_input("Young's modulus E (GPa)", min_value=0.001, max_value=2000.0,
                                value=200.0, step=1.0, format="%.3f")
        nu = sb.number_input("Poisson's ratio ν", min_value=0.0, max_value=0.499, value=0.30, step=0.01)
    else:
        E_GPa, nu = MATERIALS[material]
    G_GPa = E_GPa / (2.0 * (1.0 + nu))
    sb.info(f"**E** = {E_GPa:g} GPa  ·  **ν** = {nu:g}  ·  **G** = {G_GPa:.1f} GPa")

    sb.markdown("## 📐 Cross-section")
    shape = sb.radio("Shape", ["Rectangle", "Circle"], horizontal=True)
    b_mm = h_mm = d_mm = 0.0
    if shape == "Rectangle":
        b_mm = sb.number_input("Width b, along Z (mm)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
        h_mm = sb.number_input("Height h, along Y (mm)", min_value=1.0, max_value=2000.0, value=100.0, step=5.0)
    else:
        d_mm = sb.number_input("Diameter d (mm)", min_value=1.0, max_value=2000.0, value=80.0, step=5.0)
    sec = make_section(shape, b_mm, h_mm, d_mm)
    sb.caption(
        f"A = {sec.A * 1e4:.3f} cm²  ·  I(zz) = {sec.Izz * 1e8:.3f} cm⁴  ·  "
        f"I(yy) = {sec.Iyy * 1e8:.3f} cm⁴  ·  J = {sec.J * 1e8:.3f} cm⁴"
    )
    return dict(beam_type=beam_type, L=float(L), s1=float(s1), s2=float(s2),
                material=material, E=E_GPa * 1e9, G=G_GPa * 1e9, nu=nu, sec=sec)


def default_points():
    return pd.DataFrame(
        {
            PT_COLS[0]: [2.0],
            PT_COLS[1]: [2000.0],
            PT_COLS[2]: [500.0],
            PT_COLS[3]: [1000.0],
            PT_COLS[4]: [200.0],
        }
    )


def default_udls():
    return pd.DataFrame(
        {UDL_COLS[0]: [0.0], UDL_COLS[1]: [4.0], UDL_COLS[2]: [500.0], UDL_COLS[3]: [0.0]}
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="BeamLab 3D", page_icon="📐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>📐 BeamLab 3D</h1>
          <p>Beam analysis under combined vertical, lateral, axial and torsional loading — with an interactive 3D view.</p>
          <div class="chips">
            <span class="chip">Biaxial bending</span><span class="chip">Axial force</span>
            <span class="chip">Torsion</span><span class="chip">Multiple loads</span>
            <span class="chip">5 support types</span><span class="chip">3D stress contour</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = sidebar_inputs()
    L = cfg["L"]
    error_box = st.container()

    tab_loads, tab_res, tab_3d, tab_diag, tab_info = st.tabs(
        ["① Loads", "② Results", "③ 3D Simulation", "④ Diagrams", "⑤ Summary & Notes"]
    )

    errors = []
    with tab_loads:
        st.markdown('<div class="sectiontitle">Point loads</div>', unsafe_allow_html=True)
        st.caption(
            "One row per load position. A row may combine vertical, lateral, axial force and torque. "
            "Use the **+** at the bottom of the table to add rows, or select a row and press Delete to remove it."
        )
        pts_df = st.data_editor(
            default_points(),
            num_rows="dynamic",
            key="pts_editor",
            column_config={
                PT_COLS[0]: st.column_config.NumberColumn(PT_COLS[0], format="%.3f", help="Distance from x = 0"),
                PT_COLS[1]: st.column_config.NumberColumn(PT_COLS[1], format="%.2f", help="Positive = downward (-Y)"),
                PT_COLS[2]: st.column_config.NumberColumn(PT_COLS[2], format="%.2f", help="Positive = +Z (toward viewer)"),
                PT_COLS[3]: st.column_config.NumberColumn(PT_COLS[3], format="%.2f", help="Positive = +X (tension)"),
                PT_COLS[4]: st.column_config.NumberColumn(PT_COLS[4], format="%.2f", help="Right-hand rule about +X"),
            },
        )
        st.markdown('<div class="sectiontitle">Uniformly distributed loads (UDL)</div>', unsafe_allow_html=True)
        st.caption("Each row is a UDL acting between its start and end positions.")
        udl_df = st.data_editor(
            default_udls(),
            num_rows="dynamic",
            key="udl_editor",
            column_config={
                UDL_COLS[0]: st.column_config.NumberColumn(UDL_COLS[0], format="%.3f"),
                UDL_COLS[1]: st.column_config.NumberColumn(UDL_COLS[1], format="%.3f"),
                UDL_COLS[2]: st.column_config.NumberColumn(UDL_COLS[2], format="%.2f", help="Positive = downward"),
                UDL_COLS[3]: st.column_config.NumberColumn(UDL_COLS[3], format="%.2f", help="Positive = +Z"),
            },
        )
        st.info(
            "**Sign conventions**  ·  Vertical load: positive **downward**  ·  Lateral load: positive toward "
            "**+Z** (toward the viewer in the 3D view)  ·  Axial load: positive toward **+X**  ·  "
            "Torque: positive by the **right-hand rule** about +X."
        )

    points = clean_points(pts_df, L, errors)
    udls = clean_udls(udl_df, L, errors)

    if cfg["beam_type"] == "Overhanging" and not cfg["s1"] < cfg["s2"]:
        errors.append("Overhanging beam: support 1 must be to the left of support 2.")

    if errors:
        with error_box:
            for e in errors:
                st.error(e)
        return

    pinned, fixed, axial_r = supports_for(cfg["beam_type"], L, cfg["s1"], cfg["s2"])
    model = Model(L=L, pinned=pinned, fixed=fixed, axial_restraints=axial_r,
                  E=cfg["E"], G=cfg["G"], sec=cfg["sec"], points=points, udls=udls)
    try:
        sol = Solution(model)
    except ValueError as exc:
        with error_box:
            st.error(str(exc))
        return

    with error_box:
        if not points and not udls:
            st.warning("No loads are applied yet. Add a point load or UDL in the Loads tab.")

    S = sol.summary()
    ev = S["ev"]
    sec = model.sec
    support_x = sorted(set(model.pinned) | set(model.fixed))

    # ---------------- Results ----------------
    with tab_res:
        st.markdown('<div class="sectiontitle">Governing values</div>', unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Max normal stress σx", fmt(S["sigma_max"] / 1e6, "MPa"),
                    help=f"Axial + biaxial bending, at x = {S['sigma_x_at']:.3f} m")
        c[1].metric("Max von Mises (conservative)", fmt(S["vm_max"] / 1e6, "MPa"),
                    help="Combines the peak σx with the peak torsional shear at the same section. "
                         "Conservative for rectangles.")
        c[2].metric("Max bending deflection", fmt(S["defl_max"] * 1000, "mm"),
                    help=f"Resultant √(v² + w²), at x = {S['defl_at']:.3f} m")
        c[3].metric("Max twist angle", fmt(np.degrees(S["phi_max"]), "°"))

        c = st.columns(4)
        c[0].metric("Max deflection, Y (vertical)", fmt(S["v_max"] * 1000, "mm"))
        c[1].metric("Max deflection, Z (lateral)", fmt(S["w_max"] * 1000, "mm"))
        c[2].metric("Max axial displacement", fmt(S["u_max"] * 1000, "mm"))
        c[3].metric("Max torsional shear τ", fmt(S["tau_max"] / 1e6, "MPa"))

        c = st.columns(4)
        c[0].metric("Max shear Vy", fmt(S["Vy_max"], "N"))
        c[1].metric("Max shear Vz", fmt(S["Vz_max"], "N"))
        c[2].metric("Max moment Mxy (about Z)", fmt(S["Mxy_max"], "N·m"))
        c[3].metric("Max moment Mxz (about Y)", fmt(S["Mxz_max"], "N·m"))

        c = st.columns(4)
        c[0].metric("Max axial force N", fmt(S["N_max"], "N"))
        c[1].metric("Max torque T", fmt(S["T_max"], "N·m"))

        st.markdown('<div class="sectiontitle">Support reactions</div>', unsafe_allow_html=True)
        rt = sol.reaction_table()
        st.dataframe(
            rt.style.format({k: "{:,.3f}" for k in rt.columns if k not in ("Support",)}),
            hide_index=True,
        )
        st.caption(
            "Ry, Rz, Rx are forces in +Y, +Z, +X. Reaction moments follow the right-hand rule about Z "
            "(bending in X-Y) and about Y (bending in X-Z). Tx is the reaction torque about X."
        )

    # ---------------- 3D ----------------
    with tab_3d:
        c1, c2, c3 = st.columns([1.2, 1.6, 2.2])
        color_by = c1.radio("Colour by", ["Normal stress σx", "Total displacement"])
        exag = c2.slider("Deformation exaggeration (%)", 0, 100, 35,
                         help="100% shows the maximum deflection as 25% of the beam length "
                              "and the maximum twist as 60°.")
        with c3:
            true_scale = st.checkbox("Show true scale (no exaggeration)", value=False)
            d1, d2, d3 = st.columns(3)
            show_loads = d1.checkbox("Loads", value=True)
            show_supports = d2.checkbox("Supports", value=True)
            show_axis = d3.checkbox("Axis", value=True)
        fig3d, sd, stw, sv = build_3d(model, sol, color_by, exag, true_scale,
                                      show_loads, show_supports, show_axis)
        st.plotly_chart(fig3d)
        st.caption(
            f"Displacements are drawn ×{sd:,.4g} and twist ×{stw:,.4g}. "
            + (f"The cross-section is drawn {sv:,.3g}× larger than real so it stays visible. " if sv > 1.0001 else "")
            + "Drag to rotate, scroll to zoom. Red = tension and blue = compression in the stress view."
        )

    # ---------------- Diagrams ----------------
    with tab_diag:
        x = ev["x"]
        st.plotly_chart(line_fig("Shear force", x,
                                 [("Vy (vertical plane)", ev["Vy"], C_BLUE),
                                  ("Vz (lateral plane)", ev["Vz"], C_ORANGE)],
                                 "Shear force (N)", support_x))
        st.plotly_chart(line_fig("Bending moment (sagging positive)", x,
                                 [("Mxy (about Z)", ev["Mxy"], C_BLUE),
                                  ("Mxz (about Y)", ev["Mxz"], C_ORANGE)],
                                 "Bending moment (N·m)", support_x))
        a1, a2 = st.columns(2)
        a1.plotly_chart(line_fig("Axial force (tension +)", x, [("N", ev["N"], C_GREEN)],
                                 "Axial force (N)", support_x, height=300))
        a2.plotly_chart(line_fig("Torque", x, [("T", ev["T"], C_PURPLE)],
                                 "Torque (N·m)", support_x, height=300))
        st.plotly_chart(line_fig("Deflection (Y up, Z toward viewer)", x,
                                 [("v  (Y)", ev["v"] * 1000, C_BLUE),
                                  ("w  (Z)", ev["w"] * 1000, C_ORANGE)],
                                 "Deflection (mm)", support_x, fill=False))
        b1, b2 = st.columns(2)
        b1.plotly_chart(line_fig("Axial displacement", x, [("u", ev["u"] * 1000, C_GREEN)],
                                 "Displacement (mm)", support_x, height=300, fill=False))
        b2.plotly_chart(line_fig("Twist angle", x, [("φ", np.degrees(ev["phi"]), C_PURPLE)],
                                 "Twist (°)", support_x, height=300, fill=False))

    # ---------------- Summary ----------------
    with tab_info:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Model")
            st.table(pd.DataFrame(
                {
                    "Item": ["Support condition", "Length", "Material", "Young's modulus E",
                             "Shear modulus G", "Cross-section", "Area A", "I about Z (X-Y bending)",
                             "I about Y (X-Z bending)", "Torsion constant J"],
                    "Value": [cfg["beam_type"], f"{L:g} m", cfg["material"], f"{model.E / 1e9:.1f} GPa",
                              f"{model.G / 1e9:.1f} GPa", sec.label, f"{sec.A:.6e} m²",
                              f"{sec.Izz:.6e} m⁴", f"{sec.Iyy:.6e} m⁴", f"{sec.J:.6e} m⁴"],
                }
            ).set_index("Item"))
        with c2:
            st.markdown("#### Applied loads")
            if points:
                st.dataframe(pd.DataFrame(points, columns=PT_COLS), hide_index=True)
            if udls:
                st.dataframe(pd.DataFrame(udls, columns=UDL_COLS), hide_index=True)
            if not points and not udls:
                st.write("None")

        with st.expander("Assumptions and limitations", expanded=True):
            st.markdown(
                """
- Linear-elastic, small-displacement **Euler-Bernoulli** beam theory; constant prismatic section.
- Vertical and lateral loads act through the section **shear centre**, so they cause no twist. Torque is
  applied separately. Axial loads act through the centroid (no eccentricity).
- Supports restrain deflection in both planes and **twist** (fork-type). Axial motion is restrained at the
  left-most support (at both ends for Fixed-Fixed).
- Bending, axial and torsional behaviour are **superposed** (no P-Δ, no buckling, no coupling).
- Torsion uses **St-Venant** theory: warping restraint and transverse-shear stress are neglected. The
  rectangular torsion constant and maximum shear use standard Roark approximations.
- The von Mises value pairs the peak normal stress with the peak torsional shear, which is conservative.
- Educational tool: not a substitute for code-compliant structural design or independent checking.
                """
            )

    return


if __name__ == "__main__":
    main()
