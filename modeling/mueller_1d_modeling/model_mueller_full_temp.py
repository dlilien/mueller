#!/usr/bin/env python
# coding: utf-8

from firedrake.__future__ import interpolate
import firedrake as fd
import numpy as np
import tqdm
from constants import density, diffusivity, capacity, basal_heat_flux
from copy import deepcopy


def model_temperature_variableH(times, H_np, acc_np, surf_T_np, dtc, nz=None):
    if nz is None:
        nz = np.max(H_np) // 10.0
    mesh = fd.IntervalMesh(nz, H_np[0])
    H = fd.Constant(H_np[0])
    acc = fd.Constant(acc_np[0])

    V = fd.FunctionSpace(mesh, "DG", 1)
    W = fd.FunctionSpace(mesh, "CG", 1)

    cdiffusivity = fd.Constant(diffusivity)

    (z,) = fd.SpatialCoordinate(mesh)

    def vert_vel(a, z):
        return -a * z / H

    def vert_vel_lliboutry(a, z, H, p=3.0):
        return -a * (1 - (p + 2.0) / (p + 1.0) * (1.0 - z / H) + 1.0 / (p + 1.0) * (1.0 - z / H) ** (p + 2))

    w = vert_vel_lliboutry(acc, z, H, p=3.0)
    # w = fd.Constant(0)

    surf_T_i = fd.Constant(surf_T_np[0])
    T = fd.Function(V).interpolate(((H - z) / H) * 23.0 * H_np[0] / 1200.0 + np.mean(surf_T_np[:100]))
    initial_T = fd.Function(V).assign(T)

    Tcg = fd.project(initial_T, W)
    out_Ts = [(deepcopy(Tcg.ufl_domain().coordinates.dat.data_ro[:]), Tcg.dat.data[:])]
    Ts = [initial_T]
    out_times = [times[0]]

    dq_trial = fd.TrialFunction(V)
    phi = fd.TestFunction(V)
    a = phi * dq_trial * fd.dx

    n = fd.FacetNormal(mesh)
    wn = 0.5 * (w * n[0] + abs(w * n[0]))
    h = fd.CellDiameter(mesh)

    L1_adv = dtc * (
        T * (phi * w).dx(0) * fd.dx
        - fd.conditional(w * n[0] < 0, phi * w * n[0] * surf_T_i, 0.0) * fd.ds
        - fd.conditional(w * n[0] >= 0, phi * w * n[0], 0.0) * fd.ds
        - (phi("+") - phi("-")) * (wn("+") * T("+") - wn("-") * T("-")) * fd.dS
    )

    alpha = fd.Constant(5.0)
    L1_diff = (
        dtc
        * cdiffusivity
        * (
            -T.dx(0) * phi.dx(0) * fd.dx
            - (alpha / h("+")) * fd.jump(phi, n)[0] * fd.jump(T, n)[0] * fd.dS
            + fd.avg(phi.dx(0)) * fd.jump(T, n)[0] * fd.dS
            + fd.jump(phi, n)[0] * fd.avg(T.dx(0)) * fd.dS
        )
    )
    # cdiffusivity * (phi.dx(0) * fd.jump(T, n) + fd.jump(phi, n) * T.dx(0)) * fd.dS)
    # L1_diff = dtc * (cdiffusivity * T.dx(0) * phi.dx(0) * fd.dx)

    L1_base = (
        dtc
        * (
            -T.dx(0) * phi.dx(0)
            + basal_heat_flux / capacity / density * phi * fd.conditional(z < H / 2, fd.Constant(1.0), fd.Constant(0.0))
        )
        * fd.ds
    )
    L1_surf = (surf_T_i - T) * phi * fd.conditional(z > H / 2, fd.Constant(1.0), fd.Constant(0.0)) * fd.ds

    # L1 = dtc*(-cdiffusivity * T.dx(0) * phi.dx(0) * fd.dx)
    # L1 = L1_adv + L1_diff
    L1 = L1_adv + L1_diff + L1_base + L1_surf

    T1 = fd.Function(V)
    T2 = fd.Function(V)
    L2 = fd.replace(L1, {T: T1})
    L3 = fd.replace(L1, {T: T2})
    dT = fd.Function(V)

    params = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
    prob1 = fd.LinearVariationalProblem(a, L1, dT)
    solv1 = fd.LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = fd.LinearVariationalProblem(a, L2, dT)
    solv2 = fd.LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = fd.LinearVariationalProblem(a, L3, dT)
    solv3 = fd.LinearVariationalSolver(prob3, solver_parameters=params)

    output_freq = len(times) // 10

    Tb = [T.at(0.0)]
    progress_bar = tqdm.tqdm(times[1:])
    for i, t in enumerate(progress_bar):
        acc.assign(acc_np[i + 1])
        surf_T_i.assign(surf_T_np[i + 1])
        dtc.assign(times[i + 1] - times[i])
        if H_np[i + 1] - H_np[i] != 0:
            new_coords = mesh.coordinates.dat.data[:] * (H_np[i + 1] / H_np[i])
            Tinter = fd.Function(W)
            vec = np.ones_like(Tinter.dat.data[:]) * surf_T_i
            vec[new_coords <= H_np[i]] = T.at(new_coords[new_coords <= H_np[i]])
            Tinter.dat.data[:] = vec
            T.dat.data[:] = fd.project(Tinter, V).dat.data[:]
            mesh.coordinates.dat.data[:] *= H_np[i + 1] / H_np[i]
            H.assign(H_np[i + 1])

        # surf_T_i.assign(surf_T[i])
        solv1.solve()
        T1.assign(T + dT)

        solv2.solve()
        T2.assign(0.75 * T + 0.25 * (T1 + dT))

        solv3.solve()
        T.assign((1.0 / 3.0) * T + (2.0 / 3.0) * (T2 + dT))

        Tb.append(T.at(0.0))
        progress_bar.set_description("{:f} ka, {:f}C".format(-t / 1000.0, Tb[-1]))

        if (i + 1) % output_freq == 0:
            Tcg = fd.assemble(interpolate(T, W))
            Ts.append(T.copy(deepcopy=True))
            out_times.append(t)
            out_Ts.append((deepcopy(Tcg.ufl_domain().coordinates.dat.data_ro[:]), Tcg.dat.data_ro[:]))

    print("Slope is ", fd.interpolate(Ts[-1].dx(0), W).at(560))
    return out_times, Ts, out_Ts, Tb
