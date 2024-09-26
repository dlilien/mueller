#!/usr/bin/env python
# coding: utf-8

import tqdm
import firedrake as fd
import numpy as np

from constants import acc_modern, Hmax, Hfinal

H = fd.Constant(Hfinal)


def vert_vel(a, z, W, H, Hmax):
    return fd.conditional(z > (Hmax - H), a * (Hmax - z) / H, a)


def vert_vel_lliboutry(a, z, H, Hmax, W, p=3.0):
    return (
        fd.conditional(
            z > (Hmax - H),
            a
            * (
                1 - (p + 2.0) / (p + 1.0) * (1.0 - (Hmax - z) / H) + 1.0 / (p + 1.0) * (1.0 - (Hmax - z) / H) ** (p + 2)
            ),
            a,
        )
        + 1.0e-12
    )


def dj_age(z, H, Hmax, a, alpha=0.4):
    exp_shallow = -(2.0 - alpha) * H / (2 * a) * fd.ln((2.0 * (Hmax - z) - H * alpha) / ((2.0 - alpha) * H))
    exp_deep = (2.0 - alpha) * H / a * ((alpha * H) / (Hmax + 1 - z) - 1) - (2.0 - alpha) * H / (2 * a) * fd.ln(
        (H * alpha) / ((2.0 - alpha) * H)
    )
    dj_model = fd.conditional(z < (Hmax - H * alpha), exp_shallow, exp_deep)
    return fd.conditional(z > (Hmax - H), dj_model, fd.Constant(0.0))


def run_age(times, H_np, acc_np, nz):
    mesh = fd.IntervalMesh(nz, Hmax)

    V = fd.FunctionSpace(mesh, "DG", 1)
    W = fd.FunctionSpace(mesh, "CG", 1)

    Acc = fd.Constant(acc_modern)
    Zs_full = np.linspace(0, Hmax, int(Hmax) + 1)

    dtc = fd.Constant(times[1] - times[0])
    (z,) = fd.SpatialCoordinate(mesh)
    w = vert_vel_lliboutry(Acc, z, H, Hmax, W, p=3.0)
    surf_age = fd.Constant(0.0)

    dq_trial = fd.TrialFunction(V)
    phi = fd.TestFunction(V)
    a = phi * dq_trial * fd.dx

    n = fd.FacetNormal(mesh)
    wn = 0.5 * (w * n[0] + abs(w * n[0]))

    initial_age = fd.Function(V).interpolate(dj_age(z, H, Hmax, Acc))
    age = fd.Function(V).assign(initial_age)
    n_outs = 10
    output_freq = len(times) // n_outs
    out_times = [times[0]]
    age_mat = np.empty((len(Zs_full), n_outs + 1))
    L1 = dtc * (
        age * (phi * w).dx(0) * fd.dx
        - fd.conditional(w * n[0] < 0, phi * w * n[0] * surf_age, 0.0) * fd.ds
        - fd.conditional(w * n[0] >= 0, phi * w * n[0], 0.0) * fd.ds
        - (phi("+") - phi("-")) * (wn("+") * age("+") - wn("-") * age("-")) * fd.dS
        + fd.conditional(z > (Hmax - H), fd.Constant(1.0), fd.Constant(0.0)) * phi * fd.dx
    )

    age1 = fd.Function(V)
    age2 = fd.Function(V)
    L2 = fd.replace(L1, {age: age1})
    L3 = fd.replace(L1, {age: age2})
    dage = fd.Function(V)

    params = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
    prob1 = fd.LinearVariationalProblem(a, L1, dage)
    solv1 = fd.LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = fd.LinearVariationalProblem(a, L2, dage)
    solv2 = fd.LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = fd.LinearVariationalProblem(a, L3, dage)
    solv3 = fd.LinearVariationalSolver(prob3, solver_parameters=params)

    H.assign(H_np[0])
    Acc.assign(acc_np[0])

    initial_age = fd.Function(V).interpolate(dj_age(z, H, Hmax, Acc))
    ages = [initial_age]
    age.assign(initial_age)
    age_mat[:, 0] = ages[-1].at(Zs_full)

    j = 1
    progress_bar = tqdm.tqdm(times[1:])
    for i, t in enumerate(progress_bar):
        H.assign(H_np[i])
        Acc.assign(acc_np[i])
        # surf_T_i.assign(surf_T[i])

        solv1.solve()
        age1.assign(age + dage)

        solv2.solve()
        age2.assign(0.75 * age + 0.25 * (age1 + dage))

        solv3.solve()
        # age.assign(fd.project(fd.conditional(z > (Hmax - H), (1.0/3.0)*age + (2.0/3.0)*(age2 + dage), 0.0), V))
        age.assign((1.0 / 3.0) * age + (2.0 / 3.0) * (age2 + dage))

        if (i + 1) % output_freq == 0:
            out_times.append(t)
            ages.append(age.copy(deepcopy=True))
            age_mat[:, j] = ages[-1].at(Zs_full)
            j += 1
        progress_bar.set_description("{:f} ka".format(-t / 1000.0))

    def age_at_dist2bed(dist):
        return ages[-1].at(Hmax - dist)

    def tenk_age(age):
        return lambda dist: abs(age - age_at_dist2bed(dist))

    return out_times, Zs_full, age_mat, ages[-1], age_at_dist2bed, tenk_age, fd.project(ages[-1].dx(0), V)
