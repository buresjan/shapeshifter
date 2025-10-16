#!/usr/bin/env python3
"""
Nelder–Mead optimalizace pro Cassiniho ovál.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from lb2dgeom import Grid, classify_cells, rasterize
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.io import save_txt
from lb2dgeom.shapes.cassini_oval import CassiniOval


# --------------------------------------------------------------------------- #
# Konfigurace (dle potřeby nutno upravit)
# --------------------------------------------------------------------------- #
TNL_LBM_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "tnl-lbm"  # root submodulu s LBMkem a pomocnými skripty
LBM_SOLVER_BINARY = Path("sim_2D/sim2d_3")  # cesta k binárce solveru (tady se používá sim2d_3, který není dobrý - nutno definovat vlastní)
GEOMETRY_WORKDIR = Path(__file__).resolve().parent / "lbm_geometry_work"  # lokální složka kam se ukládají generované geoemtrie

INITIAL_ANGLE_DEG = 0.0  # počáteční rotace (stupně); optimalizace prozkoumává rozsah 0–180°.
INITIAL_A = 66.0  # počáteční parametr parametru Cassiniho oválu
INITIAL_C = 62.0  # použito pouze jednorázově k určení plochy po rasterizaci - parametr c je dopočítán z požadavku na konstantnost plochy

LBM_RESOLUTION = 8 # POZOR!!! musí korespondovat s nastavením solveru i geometrie, momentálně se to musí na hard codit
LBM_TYPE1_BOUZIDI = "off"  # chci/nechci Bouzidiho - on/off; předává se runneru
LBM_RUNS_ROOT = "cassini_runs"  # adresář kam se ukládají informace o jednotlivých simulacích, žije v tnl-lbm
LBM_PARTITION = "gp" 
LBM_WALLTIME = "10:00:00"  # Limit walltime pro job s jedním LBMkem
LBM_GPUS = 1  # funguje pouze pro =1 zatím
LBM_CPUS = 4  # počet CPU, to dává smysl měnit hlavně pokud chci paralelní simulaci, tímhle efektivně stanovím, kolik jich může běžet naráz
LBM_MEM = "16G"  # víc než dost
LBM_POLL_INTERVAL = 30.0  # jak často [s] se koukám na stav jobu
LBM_JOB_TIMEOUT: float | None = None  # hard timeout na doběhnutá jobu (None = bez limitu).
LBM_RESULT_TIMEOUT: float | None = None  # timeout čekání na výsledek (None = bez limitu).

MAX_ITER = 25  # Maximální počet iterací Nelder–Mead
MAX_EVALS = 60  # Maximální počet vyhodnocení účelové funkce - zahraje si to z max iter a max evals, co nastane první
TOL = 1e-3  # Tolerance konvergence (viz optilb) - závisí na účelové funkci
SEED: int | None = None  # seed při pro reproducivilitu (pokud podporuje optimizer) - nutné u MADS, Nelder-Mead je vždy deterministický

AREA_TOLERANCE = 5.0  # přípustná odchylka plochy (v jednotkách mřížky) při vynucování konstantní plochy


# --------------------------------------------------------------------------- #
# Konstanty pro geometrii
# --------------------------------------------------------------------------- #
GRID_NX = LBM_RESOLUTION * 128  # Požadavek: resolution * 128, např. 1024 při resolution=8.
GRID_NY = LBM_RESOLUTION * 32   # Požadavek: resolution * 32, např. 256 při resolution=8.
GRID = Grid(nx=GRID_NX, ny=GRID_NY, dx=1.0, origin=(0.0, 0.0))  # Iniciace mřížky


def second_quarter_center(grid: Grid) -> Tuple[float, float]:
    """Vrátí souřadnice (x, y) středu druhé čtvrtiny mřížky.

    Umísťujeme ovál doprostřed druhé čtvrtky kanálu: svisle do poloviny výšky,
    vodorovně do středu druhé čtvrtiny délky.
    """
    quarter_width = grid.nx // 4
    x_index = quarter_width + quarter_width // 2  # Střed druhé čtvrtiny
    y_index = grid.ny // 2                        # Polovina výšky kanálu
    x_coord = grid.origin[0] + x_index * grid.dx
    y_coord = grid.origin[1] + y_index * grid.dx
    return float(x_coord), float(y_coord)


CASSINI_CENTER = second_quarter_center(GRID)


def rasterised_area(a_value: float, c_value: float, *, angle_rad: float = 0.0) -> float:
    """Helper pro odhad plochy (v jednotkách mřížky) Cassiniho oválu.

    Plocha se počítá po rasterizaci do mřížky (počtem pevných buněk × dx²),
    takže závisí i na natočení vůči mřížce. To je důvod, proč později při
    výpočtu parametru ``c`` bereme v potaz aktuální úhel.
    """
    shape = CassiniOval(
        x0=CASSINI_CENTER[0],
        y0=CASSINI_CENTER[1],
        a=float(a_value),
        c=float(c_value),
        theta=float(angle_rad),
    )
    _, solid = rasterize(GRID, shape)
    return float(np.count_nonzero(solid) * GRID.dx * GRID.dx)


TARGET_AREA = rasterised_area(INITIAL_A, INITIAL_C) # Jakou chceme konstantní plochu, spočte se z počátečních hodnot


# --------------------------------------------------------------------------- #
# Pomocné funkce
# --------------------------------------------------------------------------- #
def solve_for_c(a_value: float, target_area: float, *, angle_rad: float = 0.0) -> float:
    """
    Vrátí hodnotu ``c``, která udrží plochu Cassiniho oválu konstantní pro dané ``a``.
    """
    # S rostoucím ``c`` plocha klesá, počítám to bisekcí
    # Začneme nějakým rozumným okolím kolem ``INITIAL_C`` a dle potřeby interval rozšiřujeme
    c_low = 1.0
    c_high = max(INITIAL_C + 20.0, a_value + 40.0)

    def area_minus_target(cand: float) -> float:
        return rasterised_area(a_value, cand, angle_rad=angle_rad) - target_area

    low_value = area_minus_target(c_low)
    while low_value < 0.0:
        # Plocha je menší než chci -> ještě snížit c
        c_low *= 0.5
        if c_low < 1e-3:
            raise RuntimeError("Failed to bracket Cassini area constraint on the lower side.")
        low_value = area_minus_target(c_low)

    high_value = area_minus_target(c_high)
    while high_value > 0.0:
        # Plocha je větší než chci -> zvětšit c
        c_high *= 2.0
        if c_high > 10000.0:
            raise RuntimeError("Failed to bracket Cassini area constraint on the upper side.")
        high_value = area_minus_target(c_high)

    # Klasická bisekce: půlíme interval, dokud nedosáhneme tolerance
    for _ in range(60):
        c_mid = 0.5 * (c_low + c_high)
        mid_value = area_minus_target(c_mid)
        if abs(mid_value) <= AREA_TOLERANCE:
            return c_mid
        if mid_value > 0.0:
            c_low = c_mid
        else:
            c_high = c_mid

    return 0.5 * (c_low + c_high)


def make_geometry_builder(
    geometry_workdir: Path,
    tnllbm_root: Path,
) -> Callable[[float, float], tuple[str, float]]:
    """Vrátí funkci, která pro daný úhel a ``a`` vytvoří geometrii.

    Výstupem je dvojice (název_souboru, vypočtené_c), kde soubor s geometrií je
    uložen lokálně a zároveň zkopírován do kořene `tnl-lbm`, odkud si jej bere
    runner skript pro spuštění simulace.
    """
    geometry_workdir.mkdir(parents=True, exist_ok=True)
    if not tnllbm_root.is_dir():
        raise FileNotFoundError(f"tnl-lbm root '{tnllbm_root}' is not a directory.")

    eval_counter = 0

    def build(angle_deg: float, a_value: float) -> tuple[str, float]:
        nonlocal eval_counter
        eval_counter += 1

        angle_rad = math.radians(angle_deg)
        # Dopočítej c pro dané natočení – minimalizuje aliasing způsobený rasterizací.
        c_value = solve_for_c(a_value, TARGET_AREA, angle_rad=angle_rad)
        cassini = CassiniOval(
            x0=CASSINI_CENTER[0],
            y0=CASSINI_CENTER[1],
            a=float(a_value),
            c=float(c_value),
            theta=angle_rad,
        )

        phi, solid = rasterize(GRID, cassini)  # promítnu analytický objekt na mřížku
        bouzidi = compute_bouzidi(GRID, phi, solid)  # spočtu koeficienty pro Bouzidiho
        cell_types = classify_cells(solid)  # klasifikace uzlů (wall/fluid/near wall).

        measured_area = float(np.count_nonzero(solid) * GRID.dx * GRID.dx)
        if abs(measured_area - TARGET_AREA) > AREA_TOLERANCE:
            raise RuntimeError(
                f"Area drift detected: target={TARGET_AREA:.3f}, measured={measured_area:.3f}",
            )

        # Jednoznačný název souboru pro rekonstrukci běhů. Neberu ani hash ani process ID, takhle jen kóduju
        # hodnoty parametrů - šlo by udělat sofistikovaněji 
        basename = f"cassini_{eval_counter:04d}_{angle_deg:06.2f}_{a_value:06.2f}.txt"
        local_path = geometry_workdir / basename
        # Uložení geometrii pro LBMko
        save_txt(local_path, GRID, cell_types, bouzidi, selection="all", include_header=False)

        # Zkopírování geometrie do kořene `tnl-lbm`, odkud si ji načítá `run_lbm_simulation`
        staged_path = tnllbm_root / basename
        shutil.copy2(local_path, staged_path)
        return basename, c_value

    return build


def make_lbm_objective(
    *,
    geometry_builder: Callable[[float, float], tuple[str, float]],
    tnllbm_root: Path,
) -> Callable[[np.ndarray], float]:
    """Zabalí tvorbu geometrie a běh LBM do volatelné funkce pro ``optilb``.

    Účelová funkce očekává vektory ``[angle_deg, a]`` a vrací zápornou hodnotu
    výsledku simulace (protože Nelder–Mead v ``optilb`` defaultně minimalizuje).
    """
    runner = tnllbm_root / "run_lbm_simulation.py"
    if not runner.is_file():
        raise FileNotFoundError(
            f"LBM runner script not found at '{runner}'. Did you initialise submodules?",
        )

    def run_simulation(geometry_name: str) -> float:
        # Importujeme runner dynamicky z podadresáře `tnl-lbm` (byl přidán na sys.path výše).
        # Funkce `submit_and_collect` zařídí:
        # - přípravu a odeslání simulace,
        # - periodické dotazování stavu jobu,
        # - parsování výsledku do objektu s `numeric_value`.
        from run_lbm_simulation import submit_and_collect

        # Spuštění simulace s explicitním předáním všech parametrů.
        # - geometry: název TXT souboru s mřížkou (z `tnl-lbm` rootu), který jsme právě vygenerovali
        # - resolution: měřítko mřížky; musí souhlasit s generovanou geometrií i očekáváním uvnitř simulace
        # - partition/walltime/gpus/cpus/mem: požadavky na zdroje (např. pro SLURM)
        # - runs_root: adresář, kam runner ukládá běhy, logy a výsledky.
        # - type1_bouzidi: přepínač Bouzidiho
        # - poll_interval: perioda dotazování stavu jobu (v sekundách)
        # - timeout/result_timeout: limity na job a čekání na výsledek (None = bez limitu).
        # - solver_binary: cesta k binárce solveru, kterou má runner spouštět
        result = submit_and_collect(
            geometry=geometry_name,
            resolution=int(LBM_RESOLUTION),
            partition=LBM_PARTITION,
            walltime=LBM_WALLTIME,
            gpus=LBM_GPUS,
            cpus=LBM_CPUS,
            mem=LBM_MEM,
            runs_root=LBM_RUNS_ROOT,
            type1_bouzidi=LBM_TYPE1_BOUZIDI,
            poll_interval=LBM_POLL_INTERVAL,
            timeout=LBM_JOB_TIMEOUT,
            result_timeout=LBM_RESULT_TIMEOUT,
            solver_binary=LBM_SOLVER_BINARY,
        )
        # Runner musí vrátit objekt s `numeric_value`. Pokud chybí, jde o chybu.
        if result.numeric_value is None:
            raise RuntimeError("LBM run completed without a numeric objective value.")
        # Přetypovat na float, abychom měli jednoznačný typ pro optimalizátor.
        return float(result.numeric_value)

    def objective(x: np.ndarray) -> float:
        params = np.asarray(x, dtype=float)
        angle_deg = float(params[0])  # Úhel natočení ve stupních.
        a_value = float(params[1])    # Parametr ```a`` Cassiniho oválu

        geometry_name, c_value = geometry_builder(angle_deg, a_value)
        value = run_simulation(geometry_name)
        print(
            f"angle={angle_deg:7.3f} deg, a={a_value:7.3f}, "
            f"c={c_value:7.3f} -> objective={value:.6f}",
            flush=True,
        )
        return -value  # Nelder–Mead minimalizuje, proto minus

    return objective


def build_design_space() -> DesignSpace:
    """Definuje optimalizační prostor optimalizace: úhel v [0, 180], ``a`` v [60, 72].

    Jména proměnných slouží k čitelnému logování uvnitř ``optilb``.
    """
    lower = [0.0, 60.0]
    upper = [180.0, 72.0]
    names = ["angle_deg", "cassini_a"]
    return DesignSpace(lower=lower, upper=upper, names=names)


def main() -> int:
    """Sestaví optimalizační úlohu a spustí Nelder–Mead.

    Zajistí, aby byl `tnl-lbm` na import path (pro runner), vytvoří builder geometrie,
    zabalí objektivní funkci a spustí optimalizaci nad definovaným prostorem.
    """
    geometry_root = GEOMETRY_WORKDIR.resolve()
    tnllbm_root = TNL_LBM_ROOT.resolve()

    # Přidat ``tnl-lbm`` na import path, aby šlo importovat ``run_lbm_simulation``.
    if str(tnllbm_root) not in sys.path:
        sys.path.insert(0, str(tnllbm_root))

    geometry_builder = make_geometry_builder(geometry_root, tnllbm_root)
    objective = make_lbm_objective(geometry_builder=geometry_builder, tnllbm_root=tnllbm_root)
    space = build_design_space()

    start_point = [INITIAL_ANGLE_DEG, INITIAL_A]
    problem = OptimizationProblem(
        objective=objective,
        space=space,
        x0=start_point,
        optimizer="nelder-mead",
        normalize=True,  # normalizuje proměnné do [0, 1] pro lepší NM krok
        max_iter=MAX_ITER,
        max_evals=MAX_EVALS,
        tol=TOL,
        seed=SEED,
        parallel=False,  # vypínám paralelizaci
    )

    print(f"Target Cassini area A = {TARGET_AREA:.3f} grid units²")  # Log: plocha po rasterizaci.
    print("Starting optimisation (maximisation achieved via negative objective)...")

    result = problem.run()

    best_angle = float(result.best_x[0])
    best_a = float(result.best_x[1])
    best_c = solve_for_c(best_a, TARGET_AREA, angle_rad=math.radians(best_angle))  # Rekonstruovat c pro nejlepší řešení.
    best_value = -float(result.best_f)

    print(f"Best angle (deg): {best_angle:.6f}")
    print(f"Best Cassini a:   {best_a:.6f}")
    print(f"Derived Cassini c: {best_c:.6f}")
    print(f"Best objective value: {best_value:.6f}")
    print(f"Objective evaluations: {result.nfev}")
    if problem.log:
        print(
            f"Optimizer={problem.log.optimizer} iterations={problem.log.nfev} runtime={problem.log.runtime:.2f}s",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
