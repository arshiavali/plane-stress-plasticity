"""
Plane-stress elasto-plasticity with FEniCSx + JAX (small strains).

    python planestress.py example.json

Elasticity : isotropic or orthotropic (rotated by an orientation angle)
Yield      : von Mises or Hill48 (anisotropic), or none (pure elasticity)
Hardening  : linear / power law isotropic hardening + linear kinematic hardening

Voigt notation (in-plane only, sigma_zz = 0):
    strain = [eps_xx, eps_yy, gamma_xy]   (engineering shear)
    stress = [sig_xx, sig_yy, sig_xy]
    state  = [epsp_xx, epsp_yy, gammap_xy, p]   per integration point
"""

import csv
import json
import sys
import time
from pathlib import Path

import basix
import basix.ufl
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

LOGO = r"""
     _    __     __
    / \   \ \   / /
   / _ \   \ \ / /
  / ___ \   \ V /
 /_/   \_\   \_/    plane-stress plasticity with FEniCSx + JAX
"""


# =============================================================================
#   Material model (JAX)
# =============================================================================


def rotation_matrices(theta):
    """T: eps_material = T @ eps_global,  R: sig_material = R @ sig_global."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    T = jnp.array([[c * c, s * s, c * s], [s * s, c * c, -c * s], [-2 * c * s, 2 * c * s, c * c - s * s]])
    R = jnp.array([[c * c, s * s, 2 * c * s], [s * s, c * c, -2 * c * s], [-c * s, c * s, c * c - s * s]])
    return T, R


def stiffness(el, theta):
    """Reduced plane-stress stiffness in the global frame: C = T^T C_material T."""
    if el["model"] == "isotropic":
        E1 = E2 = el["E"]
        nu12 = el["nu"]
        G12 = el["E"] / (2 * (1 + el["nu"]))
    else:
        E1, E2, nu12, G12 = el["E1"], el["E2"], el["nu12"], el["G12"]
    d = 1 - nu12**2 * E2 / E1
    C_mat = jnp.array([[E1 / d, nu12 * E2 / d, 0], [nu12 * E2 / d, E2 / d, 0], [0, 0, G12]])
    T, _ = rotation_matrices(theta)
    return T.T @ C_mat @ T


def hardening(h):
    """Isotropic hardening R(p)."""
    if h["law"] == "linear":
        return lambda p: h["H"] * p
    if h["law"] == "power":  # R = c ((p + p0)^(1/d) - p0^(1/d))
        c, d, p0 = h["c"], h["d"], h.get("p0", 1e-8)
        return lambda p: c * ((jnp.maximum(p, 0) + p0) ** (1 / d) - p0 ** (1 / d))
    raise ValueError(f"unknown hardening law {h['law']}")


def equivalent_stress(pl, theta):
    """Returns f(xi) with xi = [xx, yy, zz, xy] (shifted stress, tensor components)."""
    if pl["yield"] == "von_mises":

        def f(xi):
            xx, yy, zz, xy = xi
            return jnp.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * xy**2 + 1e-30)

        return f

    # Hill48 from yield stress ratios R_ij = sigma_y,ij / sigma_y0 (all 1 -> von Mises)
    r = pl["hill_ratios"]
    F = 0.5 * (1 / r["R22"] ** 2 + 1 / r["R33"] ** 2 - 1 / r["R11"] ** 2)
    G = 0.5 * (1 / r["R33"] ** 2 + 1 / r["R11"] ** 2 - 1 / r["R22"] ** 2)
    H = 0.5 * (1 / r["R11"] ** 2 + 1 / r["R22"] ** 2 - 1 / r["R33"] ** 2)
    N = 1.5 / r["R12"] ** 2
    _, Rot = rotation_matrices(theta)

    def f(xi):
        s11, s22, s12 = Rot @ jnp.array([xi[0], xi[1], xi[3]])  # rotate into material axes
        s33 = xi[2]
        return jnp.sqrt(F * (s22 - s33) ** 2 + G * (s33 - s11) ** 2 + H * (s11 - s22) ** 2 + 2 * N * s12**2 + 1e-30)

    return f


def make_material_update(mat, tol=1e-10, max_iter=50):
    """Builds a jitted, vectorised update: (eps, state_n) -> (stress, tangent, state, local iterations)."""
    theta = jnp.deg2rad(mat.get("orientation_deg", 0.0))
    C = stiffness(mat["elasticity"], theta)
    pl = mat.get("plasticity")

    if pl is None:  # pure elasticity

        def update(eps, state_n):
            return C @ eps, C, state_n, jnp.int32(0)

        return jax.jit(jax.vmap(update))

    sy0 = pl["yield_stress"]
    R = hardening(pl["isotropic_hardening"])
    Hk = pl.get("kinematic_hardening", 0.0)
    eq = equivalent_stress(pl, theta)

    def yield_fn(sig, epsp, p):
        # Prager backstress alpha = 2/3 Hk epsp; plastic flow is isochoric -> epsp_zz = -(xx + yy)
        alpha = (2 / 3) * Hk * jnp.array([epsp[0], epsp[1], -(epsp[0] + epsp[1]), 0.5 * epsp[2]])
        xi = jnp.array([sig[0], sig[1], 0.0, sig[2]]) - alpha
        return eq(xi) - (sy0 + R(p))

    def residual(x, eps, state_n):
        # unknowns x = [epsp_xx, epsp_yy, gammap_xy, dlambda]  (backward Euler)
        epsp, dlam = x[:3], x[3]
        p = state_n[3] + dlam
        sig = C @ (eps - epsp)
        n = jax.grad(yield_fn)(sig, epsp, p)
        return jnp.concatenate([epsp - state_n[:3] - dlam * n, (yield_fn(sig, epsp, p) / C[0, 0])[None]])

    dr_dx = jax.jacfwd(residual, argnums=0)
    dr_deps = jax.jacfwd(residual, argnums=1)

    def update(eps, state_n):
        sig_trial = C @ (eps - state_n[:3])

        def elastic(_):
            return sig_trial, C, state_n, jnp.int32(0)

        def plastic(_):
            def cond(c):
                return (jnp.linalg.norm(c[1]) > tol) & (c[2] < max_iter)

            def body(c):
                x = c[0] - jnp.linalg.solve(dr_dx(c[0], eps, state_n), c[1])
                x = x.at[3].set(jnp.maximum(x[3], 0.0))
                return x, residual(x, eps, state_n), c[2] + 1

            x0 = jnp.concatenate([state_n[:3], jnp.zeros(1)])
            x, _, it = jax.lax.while_loop(cond, body, (x0, residual(x0, eps, state_n), jnp.int32(0)))
            # consistent tangent via implicit differentiation of r(x(eps), eps) = 0
            dx_deps = -jnp.linalg.solve(dr_dx(x, eps, state_n), dr_deps(x, eps, state_n))
            D = C - C @ dx_deps[:3]
            return C @ (eps - x[:3]), D, jnp.concatenate([x[:3], state_n[3:] + x[3]]), it

        f_trial = yield_fn(sig_trial, state_n[:3], state_n[3])
        return jax.lax.cond(f_trial <= 1e-10 * sy0, elastic, plastic, None)

    return jax.jit(jax.vmap(update))


# =============================================================================
#   Finite elements (FEniCSx)
# =============================================================================


def strain(u):
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])


# load case -> (loaded edge, direction, [(edge, component, value)]),  "U" = prescribed displacement
LOAD_CASES = {
    "tension_x": ("right", 0, [("left", 0, 0.0), ("bottom", 1, 0.0), ("right", 0, "U")]),
    "tension_y": ("top", 1, [("left", 0, 0.0), ("bottom", 1, 0.0), ("top", 1, "U")]),
    "clamped_y": ("top", 1, [("bottom", 0, 0.0), ("bottom", 1, 0.0), ("top", 0, 0.0), ("top", 1, "U")]),
    "shear": ("top", 0, [("bottom", 0, 0.0), ("bottom", 1, 0.0), ("top", 0, "U"), ("top", 1, 0.0)]),
}


def boundary_conditions(V, geo, load_type):
    domain = V.mesh
    W, H = geo["width"], geo["height"]
    markers = {
        "left": lambda x: np.isclose(x[0], 0.0),
        "right": lambda x: np.isclose(x[0], W),
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], H),
    }

    def dofs(edge, comp):
        facets = mesh.locate_entities_boundary(domain, 1, markers[edge])
        return fem.locate_dofs_topological(V.sub(comp), 1, facets)

    loaded_edge, direction, constraints = LOAD_CASES[load_type]
    U = fem.Constant(domain, 0.0)
    bcs = [fem.dirichletbc(U if v == "U" else fem.Constant(domain, v), dofs(e, c), V.sub(c)) for e, c, v in constraints]
    area = (W if loaded_edge in ("top", "bottom") else H) * geo["thickness"]
    gauge = W if load_type.endswith("_x") else H
    return bcs, U, dofs(loaded_edge, direction), area, gauge


def run(cfg, out_dir):
    geo, num, ld = cfg["geometry"], cfg.get("numerics", {}), cfg["loading"]
    deg, qdeg = num.get("element_degree", 1), num.get("quadrature_degree", 2)

    # --- mesh and spaces
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [[0, 0], [geo["width"], geo["height"]]], [geo["nx"], geo["ny"]], mesh.CellType.quadrilateral
    )
    n_cells = domain.topology.index_map(2).size_local
    V = fem.functionspace(domain, ("Lagrange", deg, (2,)))
    u = fem.Function(V, name="displacement")

    cell = domain.basix_cell()
    qp, qw = basix.make_quadrature(cell, qdeg)
    Q3 = fem.functionspace(domain, basix.ufl.quadrature_element(cell, value_shape=(3,), degree=qdeg))
    Q33 = fem.functionspace(domain, basix.ufl.quadrature_element(cell, value_shape=(3, 3), degree=qdeg))
    sig_q, tan_q = fem.Function(Q3), fem.Function(Q33)
    qdofs = Q3.dofmap.list[:n_cells].reshape(-1)  # (cell, point) -> quadrature dof
    state = np.zeros((n_cells * len(qw), 4))

    update = make_material_update(cfg["material"], num.get("local_tol", 1e-10), num.get("local_max_iter", 50))
    bcs, U, reaction_dofs, area, gauge = boundary_conditions(V, geo, ld["type"])

    # --- weak form: residual and consistent Jacobian
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    t = fem.Constant(domain, float(geo["thickness"]))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg, "quadrature_scheme": "default"})
    F_form = fem.form(t * ufl.inner(sig_q, strain(v)) * dx)
    J_form = fem.form(t * ufl.inner(ufl.dot(tan_q, strain(du)), strain(v)) * dx)
    eps_expr = fem.Expression(strain(u), qp)
    cells = np.arange(n_cells, dtype=np.int32)

    A = fem_petsc.create_matrix(J_form)
    b = fem_petsc.create_vector(V)
    delta = u.x.petsc_vec.duplicate()
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    def material_update():
        eps = eps_expr.eval(domain, cells).reshape(-1, 3)
        sig, tan, new_state, its = (np.asarray(a) for a in update(eps, state))
        if its.max() >= num.get("local_max_iter", 50):
            raise RuntimeError("local return mapping did not converge")
        sig_q.x.array.reshape(-1, 3)[qdofs] = sig
        tan_q.x.array.reshape(-1, 9)[qdofs] = tan.reshape(-1, 9)
        return new_state

    # --- output
    xdmf = io.XDMFFile(domain.comm, str(out_dir / "results.xdmf"), "w")
    xdmf.write_mesh(domain)
    V1 = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    DG0 = fem.functionspace(domain, ("DG", 0))
    u_out = fem.Function(V1, name="displacement")
    vm_out = fem.Function(DG0, name="von_mises_stress")
    p_out = fem.Function(DG0, name="equivalent_plastic_strain")
    w = qw / qw.sum()

    rows = []
    rtol, atol, max_it = num.get("newton_rtol", 1e-8), num.get("newton_atol", 1e-10), num.get("newton_max_iter", 25)

    for step, U_val in enumerate(np.linspace(0, ld["max_displacement"], ld["num_steps"] + 1)):
        U.value = U_val
        iters = 0
        if step > 0:
            r0 = None
            for it in range(max_it + 1):
                trial_state = material_update()
                # residual with Dirichlet lifting (Newton increment form)
                with b.localForm() as bl:
                    bl.set(0.0)
                fem_petsc.assemble_vector(b, F_form)
                fem_petsc.apply_lifting(b, [J_form], [bcs], x0=[u.x.petsc_vec], alpha=-1.0)
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem_petsc.set_bc(b, bcs, u.x.petsc_vec, -1.0)
                r = b.norm()
                r0 = r0 or r
                if it > 0 and (r <= rtol * r0 or r <= atol):
                    break
                A.zeroEntries()
                fem_petsc.assemble_matrix(A, J_form, bcs=bcs)
                A.assemble()
                ksp.solve(b, delta)
                u.x.petsc_vec.axpy(-1.0, delta)
                u.x.scatter_forward()
            else:
                raise RuntimeError(f"Newton did not converge at step {step} (|r| = {r:.2e})")
            state[:] = trial_state
            iters = it

        # reaction force = sum of internal nodal forces on the loaded edge
        f_int = fem_petsc.assemble_vector(F_form)
        f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        force = float(f_int.array[reaction_dofs].sum())
        f_int.destroy()
        rows.append([step, U_val, force, U_val / gauge, force / area, iters])
        print(f"step {step:3d}  U = {U_val:.4f}  F = {force:10.4f}  Newton its = {iters}")

        # cell-averaged fields
        sig = np.einsum("cqk,q->ck", sig_q.x.array.reshape(-1, 3)[qdofs].reshape(n_cells, -1, 3), w)
        vm_out.x.array[:n_cells] = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2 + 3 * sig[:, 2] ** 2)
        p_out.x.array[:n_cells] = state[:, 3].reshape(n_cells, -1) @ w
        u_out.interpolate(u)
        for f in (u_out, vm_out, p_out):
            xdmf.write_function(f, float(step))
    xdmf.close()
    return np.array(rows)


def save_curve(rows, out_dir, title):
    with open(out_dir / "curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "displacement", "force", "nominal_strain", "nominal_stress", "newton_iterations"])
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rows[:, 3] * 100, rows[:, 4], "-o", ms=3)
    ax.set_xlabel("nominal strain [%]")
    ax.set_ylabel("nominal stress [MPa]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stress_strain.png", dpi=150)
    plt.close(fig)


def main():
    print(LOGO)
    if len(sys.argv) != 2:
        sys.exit("usage: python planestress.py <input.json>")
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path("output") / f"{cfg.get('name', 'run')}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.json").write_text(json.dumps(cfg, indent=2))
    print(f"output: {out_dir}\n")

    t0 = time.perf_counter()
    rows = run(cfg, out_dir)
    save_curve(rows, out_dir, cfg.get("name", ""))
    print(f"\nfinished in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
