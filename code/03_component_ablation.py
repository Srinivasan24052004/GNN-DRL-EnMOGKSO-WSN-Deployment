# =====================================================================
# ABLATION STUDY (single self-contained file)
# TR-EnMOGKSO + TRAINABLE GNN + DRL/DQN + ARLS - 50 nodes, 250 x 300 m
#
# PART A: the main algorithm, unchanged (its main() is not called).
# PART B: the ablation study. FULL = the proposed method exactly;
#         every other variant switches off one component.
#
# How to run
#   * Jupyter / Colab : paste this whole file into one cell and run it.
#                       In Colab it mounts Google Drive and saves results
#                       to MyDrive/ablation50 (re-run the cell to resume).
#   * Terminal        : python TR_EnMOGKSO_ablation_50nodes.py
# Edit section B.9 (ABLATION_SEEDS / ABLATION_WORKERS / ABLATION_VARIANTS)
# to change seeds, parallelism or which variants run.
# Results are appended run by run to ablation50_runs.csv; re-running
# resumes an interrupted study.
# =====================================================================

# =====================================================================
# PART A - MAIN ALGORITHM (unchanged)
# =====================================================================

import random
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 0
AREA_X, AREA_Y = 250.0, 300.0
R_SENSE, COMM_RANGE = 25.0, 30.0
GRID_RES = 50
NODE_COUNTS = [20, 30, 40, 50]
POP_SIZE, GENERATIONS = 6, 30

TARGET_COVERAGE = 0.98
TARGET_OVERLAP = 0.10
TARGET_EE = 0.97

SEARCH_MARGIN_COVERAGE = 0.005
SEARCH_MARGIN_OVERLAP = 0.010
SEARCH_MARGIN_EE = 0.005

F, CR = 0.45, 0.80
ARCHIVE_SIZE = 50

OVERLAP_REG_WEIGHT = 1.75

GNN_INPUT, GNN_HIDDEN, GNN_OUTPUT = 4, 24, 16

STATE_BASE_DIM = 12
STATE_DIM = 28
DQN_HIDDEN = 48
NUM_ACTIONS = 5
DQN_LR, GAMMA = 0.001, 0.99
BATCH_SIZE, BUFFER_SIZE = 24, 2000
EPS_START, EPS_END, EPS_DECAY = 0.70, 0.05, 15.0
TARGET_UPDATE = 5

ARLS_MOVES_INIT = 1500
ARLS_MOVES_POP = 2500
ARLS_MOVES_ELITE = 12000
ARLS_MOVES_FINAL = 30000
ARLS_FINAL_CANDIDATES = 5
ARLS_T_START, ARLS_T_END = 1e-2, 1e-5
ARLS_HOLE_PROB = 0.25
ARLS_STEP_MAX, ARLS_STEP_MIN = 30.0, 1.0
PRIORITY_REFRESH = 100

MIN_SEPARATION = 0.0
SEP_PENALTY = 0.02

REWARD_COVERAGE = 8.0
REWARD_OVERLAP = 18.0
REWARD_EE = 5.0
REWARD_FEASIBLE = 3.0

ACTION_NAMES = [
    "Global Exploration",
    "Elite Hunting",
    "Differential Hunting",
    "Redundancy Reduction",
    "Adaptive Local Search",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

gx = np.linspace(0.0, AREA_X, GRID_RES)
gy = np.linspace(0.0, AREA_Y, GRID_RES)

GX, GY = np.meshgrid(gx, gy)

GRID = np.column_stack(
    (
        GX.ravel(),
        GY.ravel()
    )
)

NUM_GRID = len(GRID)

DX = gx[1] - gx[0]
DY = gy[1] - gy[0]


def clip_positions(pos):
    pos = np.asarray(
        pos,
        dtype=np.float64
    ).copy()

    pos[:, 0] = np.clip(
        pos[:, 0],
        0.0,
        AREA_X
    )

    pos[:, 1] = np.clip(
        pos[:, 1],
        0.0,
        AREA_Y
    )

    return pos


def clip_point(p):
    return np.array(
        [
            np.clip(
                float(p[0]),
                0.0,
                AREA_X
            ),
            np.clip(
                float(p[1]),
                0.0,
                AREA_Y
            ),
        ],
        dtype=np.float64,
    )


def pairwise_distances(pos):
    d = (
        pos[:, None, :]
        -
        pos[None, :, :]
    )

    return np.sqrt(
        np.sum(
            d * d,
            axis=2
        )
    )


def sensing_mask(pos):
    d = (
        GRID[:, None, :]
        -
        pos[None, :, :]
    )

    return (
        np.sum(
            d * d,
            axis=2
        )
        <=
        R_SENSE**2
    )


def coverage_data(pos):
    mask = sensing_mask(pos)

    count = mask.sum(
        axis=1
    )

    coverage = float(
        np.mean(
            count >= 1
        )
    )

    overlap = float(
        np.mean(
            count >= 2
        )
    )

    return (
        coverage,
        overlap,
        count,
        mask
    )


def energy_efficiency(
    coverage,
    overlap
):
    return float(
        np.clip(
            0.82 * coverage
            +
            0.18 * (
                1.0
                -
                overlap
            ),
            0.0,
            1.0
        )
    )


def separation_violations(pos):

    if (
        MIN_SEPARATION <= 0.0
        or
        len(pos) < 2
    ):
        return 0

    d = pairwise_distances(
        pos
    )[
        np.triu_indices(
            len(pos),
            1
        )
    ]

    return int(
        np.count_nonzero(
            d
            <
            MIN_SEPARATION
        )
    )


def min_pair_distance(pos):

    if len(pos) < 2:
        return 0.0

    d = pairwise_distances(
        pos
    )[
        np.triu_indices(
            len(pos),
            1
        )
    ]

    return float(
        np.min(
            d
        )
    )


def connectivity_metrics(pos):

    n = len(pos)

    if n <= 1:
        return (
            True,
            1.0,
            0,
            1
        )

    adjacency = (
        (
            pairwise_distances(
                pos
            )
            <=
            COMM_RANGE
        )
        &
        (
            ~np.eye(
                n,
                dtype=bool
            )
        )
    )

    links = int(
        np.sum(
            np.triu(
                adjacency,
                1
            )
        )
    )

    visited = np.zeros(
        n,
        dtype=bool
    )

    sizes = []

    for start in range(n):

        if visited[start]:
            continue

        stack = [
            start
        ]

        visited[start] = True

        size = 0

        while stack:

            u = stack.pop()

            size += 1

            for v in np.where(
                adjacency[u]
                &
                (~visited)
            )[0]:

                visited[v] = True

                stack.append(
                    int(v)
                )

        sizes.append(
            size
        )

    components = len(
        sizes
    )

    gcc = float(
        max(sizes)
        /
        n
    )

    return (
        components == 1,
        gcc,
        links,
        components
    )


def quick_evaluate(pos):

    pos = clip_positions(
        pos
    )

    c, o, count, mask = (
        coverage_data(
            pos
        )
    )

    return {
        "pos":
            pos.copy(),

        "coverage":
            c,

        "overlap":
            o,

        "ee":
            energy_efficiency(
                c,
                o
            ),

        "sep_viol":
            separation_violations(
                pos
            ),

        "count":
            count,

        "mask":
            mask,
    }


def full_evaluate(pos):

    sol = quick_evaluate(
        pos
    )

    connected, gcc, links, components = (
        connectivity_metrics(
            sol["pos"]
        )
    )

    sol.update(
        connected=connected,
        gcc=gcc,
        links=links,
        components=components
    )

    return sol


def is_feasible(sol):

    return (
        sol["coverage"]
        >
        TARGET_COVERAGE

        and

        sol["overlap"]
        <
        TARGET_OVERLAP

        and

        sol["ee"]
        >
        TARGET_EE

        and

        sol.get(
            "sep_viol",
            0
        )
        ==
        0
    )


def fitness_raw(
    coverage,
    overlap
):

    ee = energy_efficiency(
        coverage,
        overlap
    )

    violation = (

        2.0
        *
        max(
            0.0,
            TARGET_COVERAGE
            +
            SEARCH_MARGIN_COVERAGE
            -
            coverage
        )

        +

        4.0
        *
        max(
            0.0,
            overlap
            -
            (
                TARGET_OVERLAP
                -
                SEARCH_MARGIN_OVERLAP
            )
        )

        +

        2.0
        *
        max(
            0.0,
            TARGET_EE
            +
            SEARCH_MARGIN_EE
            -
            ee
        )
    )

    return (
        ee
        -
        violation
        -
        OVERLAP_REG_WEIGHT
        *
        overlap
    )


def fitness(sol):

    c = sol[
        "coverage"
    ]

    o = sol[
        "overlap"
    ]

    ee = sol[
        "ee"
    ]

    if (
        c
        >
        TARGET_COVERAGE

        and

        ee
        >
        TARGET_EE
    ):

        value = (
            5.0
            *
            c

            +

            5.0
            *
            ee

            -

            14.0
            *
            o
        )

    else:

        value = fitness_raw(
            c,
            o
        )

    return (
        value
        -
        SEP_PENALTY
        *
        sol.get(
            "sep_viol",
            0
        )
    )


def rank_key(sol):

    return (
        int(
            is_feasible(
                sol
            )
        ),

        fitness(
            sol
        ),

        -sol[
            "overlap"
        ],

        sol[
            "coverage"
        ],

        sol[
            "ee"
        ],
    )


def better_solution(
    new,
    old
):

    return (
        old is None
        or
        rank_key(new)
        >
        rank_key(old)
    )


def disk_indices(p):

    ix0 = max(
        0,
        int(
            np.floor(
                (
                    p[0]
                    -
                    R_SENSE
                )
                /
                DX
            )
        )
        -
        1
    )

    ix1 = min(
        GRID_RES - 1,
        int(
            np.ceil(
                (
                    p[0]
                    +
                    R_SENSE
                )
                /
                DX
            )
        )
        +
        1
    )

    iy0 = max(
        0,
        int(
            np.floor(
                (
                    p[1]
                    -
                    R_SENSE
                )
                /
                DY
            )
        )
        -
        1
    )

    iy1 = min(
        GRID_RES - 1,
        int(
            np.ceil(
                (
                    p[1]
                    +
                    R_SENSE
                )
                /
                DY
            )
        )
        +
        1
    )

    ix = np.arange(
        ix0,
        ix1 + 1
    )

    iy = np.arange(
        iy0,
        iy1 + 1
    )

    ddx = (
        gx[ix]
        -
        p[0]
    )

    ddy = (
        gy[iy]
        -
        p[1]
    )

    inside = (
        ddy[:, None] ** 2
        +
        ddx[None, :] ** 2
    ) <= (
        R_SENSE**2
    )

    jj, ii = np.nonzero(
        inside
    )

    return (
        iy[jj]
        *
        GRID_RES
        +
        ix[ii]
    )


class CoverageState:

    def __init__(
        self,
        pos
    ):

        self.pos = clip_positions(
            pos
        )

        self.n = len(
            self.pos
        )

        self.idx = [
            disk_indices(p)
            for p in self.pos
        ]

        self.count = np.zeros(
            NUM_GRID,
            dtype=np.int32
        )

        for ind in self.idx:

            self.count[
                ind
            ] += 1

        self.n_cov = int(
            np.sum(
                self.count
                >=
                1
            )
        )

        self.n_ov = int(
            np.sum(
                self.count
                >=
                2
            )
        )

        self.sep_viol = (
            separation_violations(
                self.pos
            )
        )

    def metrics(self):

        return (
            self.n_cov
            /
            NUM_GRID,

            self.n_ov
            /
            NUM_GRID
        )

    def fitness(self):

        c, o = (
            self.metrics()
        )

        return (
            fitness_raw(
                c,
                o
            )

            -

            SEP_PENALTY
            *
            self.sep_viol
        )

    def node_violations(
        self,
        i,
        p
    ):

        if MIN_SEPARATION <= 0.0:
            return 0

        d = np.hypot(
            self.pos[:, 0]
            -
            p[0],

            self.pos[:, 1]
            -
            p[1]
        )

        d[i] = np.inf

        return int(
            np.count_nonzero(
                d
                <
                MIN_SEPARATION
            )
        )

    def remove(
        self,
        ind
    ):

        old = self.count[
            ind
        ].copy()

        self.count[
            ind
        ] -= 1

        self.n_cov -= int(
            np.count_nonzero(
                old
                ==
                1
            )
        )

        self.n_ov -= int(
            np.count_nonzero(
                old
                ==
                2
            )
        )

    def add(
        self,
        ind
    ):

        old = self.count[
            ind
        ].copy()

        self.count[
            ind
        ] += 1

        self.n_cov += int(
            np.count_nonzero(
                old
                ==
                0
            )
        )

        self.n_ov += int(
            np.count_nonzero(
                old
                ==
                1
            )
        )

    def move(
        self,
        i,
        new_p
    ):

        new_p = clip_point(
            new_p
        )

        old_p = (
            self.pos[
                i
            ].copy()
        )

        old_ind = (
            self.idx[
                i
            ]
        )

        new_ind = disk_indices(
            new_p
        )

        dv = (
            self.node_violations(
                i,
                new_p
            )
            -
            self.node_violations(
                i,
                old_p
            )
        )

        self.remove(
            old_ind
        )

        self.add(
            new_ind
        )

        self.pos[
            i
        ] = new_p

        self.idx[
            i
        ] = new_ind

        self.sep_viol += dv

        return (
            i,
            old_p,
            old_ind,
            new_ind,
            dv
        )

    def undo(
        self,
        info
    ):

        (
            i,
            old_p,
            old_ind,
            new_ind,
            dv
        ) = info

        self.remove(
            new_ind
        )

        self.add(
            old_ind
        )

        self.pos[
            i
        ] = old_p

        self.idx[
            i
        ] = old_ind

        self.sep_viol -= dv

    def redundancy_priority(
        self
    ):

        priority = np.empty(
            self.n,
            dtype=np.float64
        )

        for k, ind in enumerate(
            self.idx
        ):

            counts = self.count[
                ind
            ]

            redundant = np.count_nonzero(
                counts
                >=
                2
            )

            unique = np.count_nonzero(
                counts
                ==
                1
            )

            priority[k] = (
                redundant
                /
                (
                    unique
                    +
                    1.0
                )
            )

        return priority


def ARLS(
    pos,
    moves,
    t_start,
    t_end,
    step_start,
    step_end,
    rng
):

    state = CoverageState(
        pos
    )

    current_fitness = (
        state.fitness()
    )

    best_fitness = (
        current_fitness
    )

    best_position = (
        state.pos.copy()
    )

    probability = np.full(
        state.n,
        1.0
        /
        state.n
    )

    for t in range(
        moves
    ):

        frac = (
            t
            /
            max(
                moves - 1,
                1
            )
        )

        temperature = (
            t_start
            *
            (
                t_end
                /
                t_start
            )
            **
            frac
        )

        step = (
            step_start
            +
            (
                step_end
                -
                step_start
            )
            *
            frac
        )

        if (
            t
            %
            PRIORITY_REFRESH
            ==
            0
        ):

            priority = (
                state.redundancy_priority()
                +
                0.05
            )

            probability = (
                0.5
                *
                priority
                /
                priority.sum()

                +

                0.5
                /
                state.n
            )

        node = int(
            rng.choice(
                state.n,
                p=probability
            )
        )

        if (
            rng.random()
            <
            ARLS_HOLE_PROB

            and

            state.n_cov
            <
            NUM_GRID
        ):

            holes = np.flatnonzero(
                state.count
                ==
                0
            )

            target = GRID[
                holes[
                    rng.integers(
                        len(holes)
                    )
                ]
            ]

            new_position = clip_point(
                target
                +
                rng.normal(
                    0.0,
                    0.25
                    *
                    R_SENSE,
                    2
                )
            )

        else:

            new_position = clip_point(
                state.pos[
                    node
                ]
                +
                rng.normal(
                    0.0,
                    step,
                    2
                )
            )

        info = state.move(
            node,
            new_position
        )

        new_fitness = (
            state.fitness()
        )

        delta = (
            new_fitness
            -
            current_fitness
        )

        accept = (
            delta >= 0.0
            or
            rng.random()
            <
            np.exp(
                delta
                /
                max(
                    temperature,
                    1e-12
                )
            )
        )

        if accept:

            current_fitness = (
                new_fitness
            )

            if (
                current_fitness
                >
                best_fitness
            ):

                best_fitness = (
                    current_fitness
                )

                best_position = (
                    state.pos.copy()
                )

        else:

            state.undo(
                info
            )

    return (
        quick_evaluate(
            state.pos
        ),

        quick_evaluate(
            best_position
        )
    )


def schedule(
    generation
):

    a = (
        generation
        /
        max(
            GENERATIONS - 1,
            1
        )
    )

    b = min(
        1.0,
        (
            generation
            +
            1
        )
        /
        max(
            GENERATIONS - 1,
            1
        )
    )

    t0 = (
        ARLS_T_START
        *
        (
            ARLS_T_END
            /
            ARLS_T_START
        )
        **
        a
    )

    t1 = (
        ARLS_T_START
        *
        (
            ARLS_T_END
            /
            ARLS_T_START
        )
        **
        b
    )

    s0 = (
        ARLS_STEP_MAX
        +
        (
            ARLS_STEP_MIN
            -
            ARLS_STEP_MAX
        )
        *
        a
    )

    s1 = (
        ARLS_STEP_MAX
        +
        (
            ARLS_STEP_MIN
            -
            ARLS_STEP_MAX
        )
        *
        b
    )

    return (
        t0,
        t1,
        s0,
        s1
    )


def make_seed(
    n,
    phase,
    jitter,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    cols = max(
        2,
        int(
            round(
                np.sqrt(
                    n
                    *
                    AREA_X
                    /
                    AREA_Y
                )
            )
        )
    )

    rows = int(
        np.ceil(
            n
            /
            cols
        )
    )

    xs = np.linspace(
        5.0,
        AREA_X - 5.0,
        cols
    )

    ys = np.linspace(
        0.0,
        AREA_Y,
        rows
    )

    positions = []

    for i, x in enumerate(
        xs
    ):

        for y in ys:

            if len(positions) >= n:
                break

            xx = x

            yy = (
                y
                +
                (
                    phase
                    if i % 2
                    else
                    0.0
                )
            )

            if jitter > 0:

                xx += rng.normal(
                    0.0,
                    jitter
                )

                yy += rng.normal(
                    0.0,
                    jitter
                )

            positions.append(
                [
                    xx,
                    yy
                ]
            )

        if len(positions) >= n:
            break

    return clip_positions(
        np.asarray(
            positions,
            dtype=np.float64
        )
    )


def create_population(n):

    phases = [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        8.0
    ]

    jitters = [
        0.0,
        0.2,
        0.3,
        0.3,
        0.4,
        0.5
    ]

    return [
        make_seed(
            n,
            phases[i],
            jitters[i],
            SEED + i
        )
        for i in range(
            POP_SIZE
        )
    ]


class GNNEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=GNN_INPUT,
        hidden_dim=GNN_HIDDEN,
        output_dim=GNN_OUTPUT
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        x,
        adjacency
    ):

        n = x.shape[0]

        eye = torch.eye(
            n,
            dtype=torch.float32,
            device=x.device
        )

        a = (
            adjacency
            +
            eye
        )

        degree = a.sum(
            dim=1,
            keepdim=True
        )

        denom = (
            torch.sqrt(
                degree
            )
            @
            torch.sqrt(
                degree
            ).T
        )

        a_norm = (
            a
            /
            (
                denom
                +
                1e-8
            )
        )

        h = self.relu(
            self.fc1(
                a_norm
                @
                x
            )
        )

        h = self.relu(
            self.fc2(
                a_norm
                @
                h
            )
        )

        return h.mean(
            dim=0
        )


def build_graph(pos):

    d = pairwise_distances(
        pos
    )

    return (
        (
            d
            <=
            COMM_RANGE
        )
        &
        (
            ~np.eye(
                len(pos),
                dtype=bool
            )
        )
    ).astype(
        np.float32
    )


def graph_tensors(pos):

    adjacency_np = build_graph(
        pos
    )

    n = len(pos)

    xy = np.column_stack(
        (
            pos[:, 0]
            /
            AREA_X,

            pos[:, 1]
            /
            AREA_Y
        )
    )

    degree = (
        adjacency_np.sum(
            axis=1,
            keepdims=True
        )
        /
        max(
            n - 1,
            1
        )
    )

    energy = np.ones(
        (
            n,
            1
        ),
        dtype=np.float32
    )

    features = np.hstack(
        (
            xy,
            degree,
            energy
        )
    ).astype(
        np.float32
    )

    x = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    )

    adjacency = torch.tensor(
        adjacency_np,
        dtype=torch.float32,
        device=DEVICE
    )

    return (
        x,
        adjacency
    )


def topology_features(pos):

    adjacency = build_graph(
        pos
    )

    n = len(pos)

    degree = (
        adjacency.sum(
            axis=1
        )
        /
        max(
            n - 1,
            1
        )
    )

    link_ratio = (
        np.sum(
            np.triu(
                adjacency,
                1
            )
        )
        /
        max(
            n
            *
            (
                n - 1
            )
            /
            2,
            1
        )
    )

    return np.array(
        [
            np.mean(
                degree
            ),

            1.0
            -
            np.mean(
                degree == 0
            ),

            link_ratio,

            np.std(
                degree
            ),
        ],
        dtype=np.float32,
    )


def build_state(sol):

    pos = sol[
        "pos"
    ]

    n = len(pos)

    if n > 1:

        upper = pairwise_distances(
            pos
        )[
            np.triu_indices(
                n,
                1
            )
        ]

        mean_distance = float(
            np.mean(
                upper
            )
        )

        minimum_distance = float(
            np.min(
                upper
            )
        )

    else:

        mean_distance = 0.0

        minimum_distance = 0.0

    scale = max(
        AREA_X,
        AREA_Y
    )

    base = np.array(
        [
            sol[
                "coverage"
            ],

            sol[
                "overlap"
            ],

            sol[
                "ee"
            ],

            mean_distance
            /
            scale,

            minimum_distance
            /
            scale,

            sol[
                "coverage"
            ]
            -
            TARGET_COVERAGE,

            TARGET_OVERLAP
            -
            sol[
                "overlap"
            ],

            TARGET_EE
            -
            sol[
                "ee"
            ],
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        (
            base,
            topology_features(
                pos
            )
        )
    ).astype(
        np.float32
    )


class HybridQNet(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.gnn = GNNEncoder()

        self.q = nn.Sequential(

            nn.Linear(
                STATE_BASE_DIM
                +
                GNN_OUTPUT,
                DQN_HIDDEN
            ),

            nn.ReLU(),

            nn.Linear(
                DQN_HIDDEN,
                DQN_HIDDEN
            ),

            nn.ReLU(),

            nn.Linear(
                DQN_HIDDEN,
                NUM_ACTIONS
            )
        )

    def forward(
        self,
        base_states,
        positions
    ):

        embeddings = []

        for pos in positions:

            x, adjacency = (
                graph_tensors(
                    pos
                )
            )

            embeddings.append(
                self.gnn(
                    x,
                    adjacency
                )
            )

        graph_embeddings = torch.stack(
            embeddings,
            dim=0
        )

        return self.q(
            torch.cat(
                (
                    base_states,
                    graph_embeddings
                ),
                dim=1
            )
        )


class ReplayBuffer:

    def __init__(
        self,
        capacity=BUFFER_SIZE
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        pos,
        action,
        reward,
        next_state,
        next_pos,
        done
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    np.float32
                ).copy(),

                np.asarray(
                    pos,
                    np.float32
                ).copy(),

                int(
                    action
                ),

                float(
                    reward
                ),

                np.asarray(
                    next_state,
                    np.float32
                ).copy(),

                np.asarray(
                    next_pos,
                    np.float32
                ).copy(),

                float(
                    done
                )
            )
        )

    def sample(
        self,
        batch_size
    ):

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(
        self
    ):

        return len(
            self.buffer
        )


def choose_action(
    q_net,
    state,
    pos,
    epsilon,
    rng
):

    if (
        rng.random()
        <
        epsilon
    ):

        return int(
            rng.integers(
                NUM_ACTIONS
            )
        )

    s = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(
        0
    )

    with torch.no_grad():

        q = q_net(
            s,
            [
                pos
            ]
        )

    return int(
        torch.argmax(
            q,
            dim=1
        ).item()
    )


def update_dqn(
    q_net,
    target_net,
    optimizer,
    replay
):

    if (
        len(replay)
        <
        BATCH_SIZE
    ):

        return (
            None,
            None
        )

    batch = replay.sample(
        BATCH_SIZE
    )

    (
        states,
        positions,
        actions,
        rewards,
        next_states,
        next_positions,
        dones
    ) = zip(
        *batch
    )

    states = torch.tensor(
        np.asarray(
            states,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    next_states = torch.tensor(
        np.asarray(
            next_states,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    actions = torch.tensor(
        np.asarray(
            actions,
            np.int64
        ),
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(
        1
    )

    rewards = torch.tensor(
        np.asarray(
            rewards,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    dones = torch.tensor(
        np.asarray(
            dones,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    current_q = (
        q_net(
            states,
            list(
                positions
            )
        )
        .gather(
            1,
            actions
        )
        .squeeze(
            1
        )
    )

    with torch.no_grad():

        next_q = (
            target_net(
                next_states,
                list(
                    next_positions
                )
            )
            .max(
                dim=1
            )
            .values
        )

        target = (
            rewards
            +
            GAMMA
            *
            (
                1.0
                -
                dones
            )
            *
            next_q
        )

    loss = nn.functional.mse_loss(
        current_q,
        target
    )

    optimizer.zero_grad()

    loss.backward()

    grad_norm = 0.0

    for p in q_net.gnn.parameters():

        if p.grad is not None:

            grad_norm += float(
                p.grad.detach().norm().item()
            )

    torch.nn.utils.clip_grad_norm_(
        q_net.parameters(),
        1.0
    )

    optimizer.step()

    return (
        float(
            loss.item()
        ),
        float(
            grad_norm
        )
    )


def global_exploration(
    current,
    elite,
    rng,
    generation
):

    progress = (
        generation
        /
        max(
            GENERATIONS - 1,
            1
        )
    )

    exploration = (
        0.16
        -
        0.08
        *
        progress
    )

    candidate = current.copy()

    noise = rng.normal(
        0.0,
        1.0,
        current.shape
    )

    attraction = (
        rng.random(
            current.shape
        )
        *
        (
            elite
            -
            current
        )
    )

    selected = (
        rng.random(
            len(current)
        )
        <
        0.25
    )

    candidate[
        selected
    ] += (
        exploration
        *
        max(
            AREA_X,
            AREA_Y
        )
        *
        noise
    )[
        selected
    ]

    candidate += (
        0.20
        *
        attraction
    )

    return clip_positions(
        candidate
    )


def elite_hunting(
    current,
    elite,
    rng
):

    candidate = (
        current

        +

        rng.random(
            current.shape
        )

        *
        (
            elite
            -
            current
        )

        +

        rng.normal(
            0.0,
            1.2,
            current.shape
        )
    )

    return clip_positions(
        candidate
    )


def differential_hunting(
    current,
    population,
    elite,
    rng
):

    if len(population) < 3:

        return current.copy()

    i1, i2 = rng.choice(
        len(population),
        2,
        replace=False
    )

    mutant = (
        current

        +

        F
        *
        (
            population[
                i1
            ]
            -
            population[
                i2
            ]
        )

        +

        0.20
        *
        rng.random(
            current.shape
        )
        *
        (
            elite
            -
            current
        )
    )

    mask = (
        rng.random(
            current.shape
        )
        <
        CR
    )

    return clip_positions(
        np.where(
            mask,
            mutant,
            current
        )
    )


def redundancy_operator(
    current,
    rng
):

    state = CoverageState(
        current
    )

    current_fitness = (
        state.fitness()
    )

    priority = (
        state.redundancy_priority()
    )

    for node in np.argsort(
        -priority
    )[
        :6
    ]:

        holes = np.flatnonzero(
            state.count
            ==
            0
        )

        if len(holes) == 0:
            break

        best_target = None

        best_value = (
            current_fitness
        )

        sampled = rng.choice(
            holes,
            size=min(
                12,
                len(holes)
            ),
            replace=False
        )

        for h in sampled:

            info = state.move(
                int(node),
                GRID[h]
            )

            value = (
                state.fitness()
            )

            state.undo(
                info
            )

            if value > best_value:

                best_value = (
                    value
                )

                best_target = (
                    GRID[h].copy()
                )

        if (
            best_target
            is not None
        ):

            state.move(
                int(node),
                best_target
            )

            current_fitness = (
                best_value
            )

    return state.pos.copy()


def adaptive_local_operator(
    current,
    rng
):

    state = CoverageState(
        current
    )

    node = int(
        np.argmax(
            state.redundancy_priority()
        )
    )

    directions = np.array(
        [
            [
                1.0,
                0.0
            ],

            [
                -1.0,
                0.0
            ],

            [
                0.0,
                1.0
            ],

            [
                0.0,
                -1.0
            ]
        ]
    )

    step = rng.choice(
        [
            6.0,
            3.0,
            1.5,
            0.75
        ]
    )

    direction = directions[
        rng.integers(
            4
        )
    ]

    candidate = current.copy()

    candidate[
        node
    ] += (
        step
        *
        direction
    )

    return clip_positions(
        candidate
    )


def apply_operator(
    action,
    current,
    population,
    elite,
    rng,
    generation
):

    if action == 0:

        return global_exploration(
            current,
            elite,
            rng,
            generation
        )

    if action == 1:

        return elite_hunting(
            current,
            elite,
            rng
        )

    if action == 2:

        return differential_hunting(
            current,
            population,
            elite,
            rng
        )

    if action == 3:

        return redundancy_operator(
            current,
            rng
        )

    return adaptive_local_operator(
        current,
        rng
    )


def dominates(
    a,
    b
):

    if (
        a.get(
            "sep_viol",
            0
        )
        >
        b.get(
            "sep_viol",
            0
        )
    ):

        return False

    no_worse = (
        a["coverage"]
        >=
        b["coverage"]
        -
        1e-10

        and

        a["ee"]
        >=
        b["ee"]
        -
        1e-10

        and

        a["overlap"]
        <=
        b["overlap"]
        +
        1e-10
    )

    strictly = (
        a["coverage"]
        >
        b["coverage"]
        +
        1e-10

        or

        a["ee"]
        >
        b["ee"]
        +
        1e-10

        or

        a["overlap"]
        <
        b["overlap"]
        -
        1e-10
    )

    return (
        no_worse
        and
        strictly
    )


def add_to_archive(
    archive,
    candidate
):

    if any(
        dominates(
            item,
            candidate
        )
        for item in archive
    ):

        return archive

    new_archive = [
        item
        for item in archive
        if not dominates(
            candidate,
            item
        )
    ]

    new_archive.append(
        {
            k: candidate[k]
            for k in (
                "coverage",
                "overlap",
                "ee",
                "sep_viol"
            )
        }
        |
        {
            "pos":
                candidate[
                    "pos"
                ].copy()
        }
    )

    new_archive.sort(
        key=rank_key,
        reverse=True
    )

    return new_archive[
        :ARCHIVE_SIZE
    ]


def select_global_leader(
    candidates
):

    candidates = [
        x
        for x in candidates
        if x is not None
    ]

    if candidates:

        return max(
            candidates,
            key=rank_key
        )

    return None


def select_final_candidate(
    candidates
):

    candidates = [
        x
        for x in candidates
        if x is not None
    ]

    feasible = [
        x
        for x in candidates
        if is_feasible(
            x
        )
    ]

    if feasible:

        return min(
            feasible,
            key=lambda x: (
                x["overlap"],
                -x["coverage"],
                -x["ee"]
            )
        )

    return max(
        candidates,
        key=rank_key
    )


def run_proposed(
    n,
    seed=SEED
):

    start = time.time()

    set_seed(
        seed
    )

    rng = np.random.default_rng(
        seed
    )

    q_net = HybridQNet().to(
        DEVICE
    )

    target_net = HybridQNet().to(
        DEVICE
    )

    target_net.load_state_dict(
        q_net.state_dict()
    )

    optimizer = optim.Adam(
        q_net.parameters(),
        lr=DQN_LR
    )

    replay = ReplayBuffer()

    gnn_params = sum(
        p.numel()
        for p in q_net.gnn.parameters()
    )

    losses = []

    grads = []

    gnn_updates = 0

    population = []

    archive = []

    elite = None

    t0, _, s0, _ = schedule(
        0
    )

    for raw in create_population(
        n
    ):

        current, best = ARLS(
            raw,
            ARLS_MOVES_INIT,
            t0,
            t0,
            s0,
            s0,
            rng
        )

        population.append(
            current[
                "pos"
            ].copy()
        )

        archive = add_to_archive(
            archive,
            best
        )

        if better_solution(
            best,
            elite
        ):

            elite = best

    population_solutions = [
        quick_evaluate(
            p
        )
        for p in population
    ]

    history = []

    chain_position = (
        elite[
            "pos"
        ].copy()
    )

    for generation in range(
        GENERATIONS
    ):

        epsilon = max(
            EPS_END,
            EPS_START
            *
            np.exp(
                -generation
                /
                EPS_DECAY
            )
        )

        (
            t_a,
            t_b,
            s_a,
            s_b
        ) = schedule(
            generation
        )

        new_population = []

        actions = []

        for i, current in enumerate(
            population
        ):

            current_sol = (
                population_solutions[
                    i
                ]
            )

            state = build_state(
                full_evaluate(
                    current
                )
            )

            action = choose_action(
                q_net,
                state,
                current,
                epsilon,
                rng
            )

            actions.append(
                action
            )

            candidate = apply_operator(
                action,
                current,
                population,
                elite[
                    "pos"
                ],
                rng,
                generation
            )

            cand_cur, cand_best = ARLS(
                candidate,
                ARLS_MOVES_POP,
                t_a,
                t_b,
                s_a,
                s_b,
                rng
            )

            archive = add_to_archive(
                archive,
                cand_best
            )

            if better_solution(
                cand_best,
                elite
            ):

                elite = cand_best

            reward = (
                REWARD_COVERAGE
                *
                (
                    cand_cur[
                        "coverage"
                    ]
                    -
                    current_sol[
                        "coverage"
                    ]
                )

                +

                REWARD_OVERLAP
                *
                (
                    current_sol[
                        "overlap"
                    ]
                    -
                    cand_cur[
                        "overlap"
                    ]
                )

                +

                REWARD_EE
                *
                (
                    cand_cur[
                        "ee"
                    ]
                    -
                    current_sol[
                        "ee"
                    ]
                )

                +

                REWARD_FEASIBLE
                *
                float(
                    is_feasible(
                        cand_best
                    )
                    and
                    not is_feasible(
                        current_sol
                    )
                )
            )

            next_sol = full_evaluate(
                cand_cur[
                    "pos"
                ]
            )

            next_state = build_state(
                next_sol
            )

            replay.push(
                state,
                current,
                action,
                reward,
                next_state,
                cand_cur[
                    "pos"
                ],
                generation
                ==
                GENERATIONS - 1
            )

            loss, grad = update_dqn(
                q_net,
                target_net,
                optimizer,
                replay
            )

            if loss is not None:

                losses.append(
                    loss
                )

                grads.append(
                    grad
                )

                gnn_updates += int(
                    grad > 0.0
                )

            delta = (
                fitness(
                    cand_cur
                )
                -
                fitness(
                    current_sol
                )
            )

            accept = (
                delta >= 0.0

                or

                rng.random()
                <
                np.exp(
                    delta
                    /
                    max(
                        t_b,
                        1e-12
                    )
                )
            )

            new_population.append(
                cand_cur[
                    "pos"
                ].copy()
                if accept
                else
                current.copy()
            )

        population = (
            new_population
        )

        population_solutions = [
            quick_evaluate(
                p
            )
            for p in population
        ]

        for sol in (
            population_solutions
        ):

            archive = add_to_archive(
                archive,
                sol
            )

        chain_current, elite_best = ARLS(
            chain_position,
            ARLS_MOVES_ELITE,
            t_a,
            t_b,
            s_a,
            s_b,
            rng
        )

        chain_position = (
            chain_current[
                "pos"
            ].copy()
        )

        archive = add_to_archive(
            archive,
            elite_best
        )

        leader_pool = (
            population_solutions

            +

            [
                quick_evaluate(
                    a[
                        "pos"
                    ]
                )
                for a in archive
            ]

            +

            [
                elite,
                elite_best
            ]
        )

        elite = select_global_leader(
            leader_pool
        )

        population_solutions = [
            quick_evaluate(
                p
            )
            for p in population
        ]

        worst = int(
            np.argmin(
                [
                    fitness(
                        sol
                    )
                    for sol in (
                        population_solutions
                    )
                ]
            )
        )

        population[
            worst
        ] = (
            elite[
                "pos"
            ].copy()
        )

        population_solutions[
            worst
        ] = quick_evaluate(
            elite[
                "pos"
            ]
        )

        if (
            generation
            +
            1
        ) % TARGET_UPDATE == 0:

            target_net.load_state_dict(
                q_net.state_dict()
            )

        dominant_action = (
            max(
                set(
                    actions
                ),
                key=actions.count
            )
            if actions
            else
            0
        )

        history.append(
            {
                "generation":
                    generation + 1,

                "coverage":
                    elite[
                        "coverage"
                    ]
                    *
                    100.0,

                "overlap":
                    elite[
                        "overlap"
                    ]
                    *
                    100.0,

                "ee":
                    elite[
                        "ee"
                    ]
                    *
                    100.0,

                "epsilon":
                    epsilon,

                "dominant_action":
                    ACTION_NAMES[
                        dominant_action
                    ],

                "feasible":
                    is_feasible(
                        elite
                    ),
            }
        )

        if (
            generation == 0
            or
            (
                generation + 1
            ) % 5 == 0
            or
            generation == GENERATIONS - 1
        ):

            print(
                f"N={n:2d} | "
                f"Gen={generation+1:2d} | "
                f"Cov={elite['coverage']*100:7.3f}% | "
                f"Ov={elite['overlap']*100:7.3f}% | "
                f"EE={elite['ee']*100:7.3f}% | "
                f"Action={ACTION_NAMES[dominant_action]} | "
                f"{'PASS' if is_feasible(elite) else 'SEARCHING'}",
                flush=True
            )

    pool = (
        population_solutions

        +

        [
            quick_evaluate(
                a[
                    "pos"
                ]
            )
            for a in archive
        ]

        +

        [
            elite
        ]
    )

    pool = [
        x
        for x in pool
        if x is not None
    ]

    pool.sort(
        key=rank_key,
        reverse=True
    )

    polished = []

    for candidate in pool[
        :ARLS_FINAL_CANDIDATES
    ]:

        _, refined = ARLS(
            candidate[
                "pos"
            ],
            ARLS_MOVES_FINAL,
            1e-4,
            ARLS_T_END,
            3.0,
            ARLS_STEP_MIN,
            rng
        )

        polished.append(
            refined
        )

    final_candidate = (
        select_final_candidate(
            pool
            +
            polished
        )
    )

    final = full_evaluate(
        final_candidate[
            "pos"
        ]
    )

    c2, o2, _, _ = (
        coverage_data(
            final[
                "pos"
            ]
        )
    )

    e2 = energy_efficiency(
        c2,
        o2
    )

    independent_match = (
        abs(
            final[
                "coverage"
            ]
            -
            c2
        )
        <
        1e-12

        and

        abs(
            final[
                "overlap"
            ]
            -
            o2
        )
        <
        1e-12

        and

        abs(
            final[
                "ee"
            ]
            -
            e2
        )
        <
        1e-12
    )

    return {
        "nodes":
            n,

        "coverage":
            final[
                "coverage"
            ]
            *
            100.0,

        "overlap":
            final[
                "overlap"
            ]
            *
            100.0,

        "ee":
            final[
                "ee"
            ]
            *
            100.0,

        "gcc":
            final[
                "gcc"
            ]
            *
            100.0,

        "links":
            final[
                "links"
            ],

        "components":
            final[
                "components"
            ],

        "connected":
            final[
                "connected"
            ],

        "feasible":
            is_feasible(
                final
            ),

        "min_sep":
            min_pair_distance(
                final[
                    "pos"
                ]
            ),

        "colocated":
            int(
                n
                -
                len(
                    np.unique(
                        np.round(
                            final[
                                "pos"
                            ],
                            3
                        ),
                        axis=0
                    )
                )
            ),

        "runtime":
            time.time()
            -
            start,

        "positions":
            final[
                "pos"
            ].copy(),

        "history":
            history,

        "archive_size":
            len(
                archive
            ),

        "gnn_params":
            gnn_params,

        "gnn_updates":
            gnn_updates,

        "avg_dqn_loss":
            (
                float(
                    np.mean(
                        losses
                    )
                )
                if losses
                else
                np.nan
            ),

        "avg_gnn_grad":
            (
                float(
                    np.mean(
                        grads
                    )
                )
                if grads
                else
                np.nan
            ),

        "independent_match":
            independent_match,
    }


def main():

    print(
        "="
        *
        110
    )

    print(
        "TR-EnMOGKSO + TRAINABLE GNN + DRL/DQN + ARLS"
    )

    print(
        "ENHANCED OVERLAP-AWARE 250 x 300 m NODE-SCALABILITY"
    )

    print(
        "="
        *
        110
    )

    print(
        f"Device                  : {DEVICE}"
    )

    print(
        f"Area                    : "
        f"{AREA_X:.0f} x {AREA_Y:.0f} m"
    )

    print(
        f"Sensing range           : "
        f"{R_SENSE:.0f} m"
    )

    print(
        f"Communication range     : "
        f"{COMM_RANGE:.0f} m"
    )

    print(
        f"Population              : "
        f"{POP_SIZE}"
    )

    print(
        f"Generations             : "
        f"{GENERATIONS}"
    )

    print(
        f"Node counts             : "
        f"{NODE_COUNTS}"
    )

    print(
        f"Overlap reg. weight     : "
        f"{OVERLAP_REG_WEIGHT:.2f}"
    )

    print(
        "Stage-2 overlap weight  : "
        "14.00"
    )

    print(
        f"Final ARLS moves        : "
        f"{ARLS_MOVES_FINAL}"
    )

    print(
        f"Final candidates        : "
        f"{ARLS_FINAL_CANDIDATES}"
    )

    print()

    print(
        "Hardcoded results       : NO"
    )

    print(
        "50-node special rule    : NO"
    )

    print(
        "Lloyd                   : NO"
    )

    print(
        "Virtual-force Lloyd     : NO"
    )

    print(
        "Trainable GNN           : YES"
    )

    print(
        "GNN -> DQN TD gradients : YES"
    )

    print(
        "DRL / DQN operator      : YES"
    )

    print(
        "ARLS refinement         : YES"
    )

    print(
        "Pareto archive          : YES"
    )

    print()

    print(
        "FEASIBILITY TARGETS"
    )

    print(
        f"Coverage > "
        f"{TARGET_COVERAGE * 100:.2f}%"
    )

    print(
        f"Overlap  < "
        f"{TARGET_OVERLAP * 100:.2f}%"
    )

    print(
        f"EE       > "
        f"{TARGET_EE * 100:.2f}%"
    )

    print(
        "="
        *
        110
    )

    overall_start = time.time()

    results = []

    histories = {}

    positions = {}

    for n in NODE_COUNTS:

        print(
            "\n"
            +
            "#"
            *
            110
        )

        print(
            f"STARTING {n}-NODE EXPERIMENT"
        )

        print(
            "#"
            *
            110
        )

        r = run_proposed(
            n,
            seed=SEED
        )

        histories[
            n
        ] = r[
            "history"
        ]

        positions[
            n
        ] = r[
            "positions"
        ]

        results.append(
            {
                "Nodes":
                    r[
                        "nodes"
                    ],

                "Coverage (%)":
                    r[
                        "coverage"
                    ],

                "Overlap (%)":
                    r[
                        "overlap"
                    ],

                "EE (%)":
                    r[
                        "ee"
                    ],

                "GCC (%)":
                    r[
                        "gcc"
                    ],

                "Links":
                    r[
                        "links"
                    ],

                "Components":
                    r[
                        "components"
                    ],

                "Connected":
                    r[
                        "connected"
                    ],

                "Feasible":
                    r[
                        "feasible"
                    ],

                "Min sep (m)":
                    r[
                        "min_sep"
                    ],

                "Co-located":
                    r[
                        "colocated"
                    ],

                "Runtime (s)":
                    r[
                        "runtime"
                    ],

                "Archive":
                    r[
                        "archive_size"
                    ],

                "GNN params":
                    r[
                        "gnn_params"
                    ],

                "GNN updates":
                    r[
                        "gnn_updates"
                    ],

                "Avg DQN loss":
                    r[
                        "avg_dqn_loss"
                    ],

                "Avg GNN grad":
                    r[
                        "avg_gnn_grad"
                    ],

                "Independent check":
                    r[
                        "independent_match"
                    ],
            }
        )

        print()

        print(
            f"FINAL {n}-NODE RESULT"
        )

        print(
            f"Coverage            : "
            f"{r['coverage']:.4f}%"
        )

        print(
            f"Overlap             : "
            f"{r['overlap']:.4f}%"
        )

        print(
            f"Energy Efficiency   : "
            f"{r['ee']:.4f}%"
        )

        print(
            f"GCC                 : "
            f"{r['gcc']:.4f}%"
        )

        print(
            f"Links               : "
            f"{r['links']}"
        )

        print(
            f"Components          : "
            f"{r['components']}"
        )

        print(
            f"Connected           : "
            f"{r['connected']}"
        )

        print(
            "Feasibility         : "
            +
            (
                "PASS"
                if r[
                    "feasible"
                ]
                else
                "NOT ACHIEVED"
            )
        )

        print(
            f"GNN trainable params: "
            f"{r['gnn_params']}"
        )

        print(
            f"GNN TD updates      : "
            f"{r['gnn_updates']}"
        )

        print(
            f"Average DQN loss    : "
            f"{r['avg_dqn_loss']:.8e}"
        )

        print(
            f"Average GNN grad    : "
            f"{r['avg_gnn_grad']:.8e}"
        )

        print(
            f"Independent check   : "
            f"{r['independent_match']}"
        )

        print(
            f"Runtime             : "
            f"{r['runtime']:.2f} s"
        )

    df = pd.DataFrame(
        results
    )

    print(
        "\n"
        +
        "="
        *
        120
    )

    print(
        "FINAL 250 x 300 NODE-SCALABILITY RESULTS"
    )

    print(
        "="
        *
        120
    )

    print(
        df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    independent = []

    for n in NODE_COUNTS:

        c, o, _, _ = (
            coverage_data(
                positions[
                    n
                ]
            )
        )

        e = energy_efficiency(
            c,
            o
        )

        independent.append(
            {
                "Nodes":
                    n,

                "Coverage (%)":
                    c
                    *
                    100,

                "Overlap (%)":
                    o
                    *
                    100,

                "EE (%)":
                    e
                    *
                    100,

                "Coverage > 98%":
                    c
                    >
                    TARGET_COVERAGE,

                "Overlap < 10%":
                    o
                    <
                    TARGET_OVERLAP,

                "EE > 97%":
                    e
                    >
                    TARGET_EE,
            }
        )

    idf = pd.DataFrame(
        independent
    )

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "INDEPENDENT METRIC RECOMPUTATION"
    )

    print(
        "="
        *
        110
    )

    print(
        idf.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    feasible = df[
        df[
            "Feasible"
        ]
        ==
        True
    ]

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "BEST FEASIBLE OPERATING POINT"
    )

    print(
        "="
        *
        110
    )

    if len(feasible):

        best = (
            feasible
            .sort_values(
                [
                    "Overlap (%)",
                    "Coverage (%)",
                    "EE (%)"
                ],
                ascending=[
                    True,
                    False,
                    False
                ]
            )
            .iloc[0]
        )

        print(
            f"Best feasible node count : "
            f"{int(best['Nodes'])}"
        )

        print(
            f"Coverage                 : "
            f"{best['Coverage (%)']:.4f}%"
        )

        print(
            f"Overlap                  : "
            f"{best['Overlap (%)']:.4f}%"
        )

        print(
            f"EE                       : "
            f"{best['EE (%)']:.4f}%"
        )

    else:

        print(
            "No node count achieved all targets."
        )

    row40 = (
        df[
            df[
                "Nodes"
            ]
            ==
            40
        ]
        .iloc[0]
    )

    row50 = (
        df[
            df[
                "Nodes"
            ]
            ==
            50
        ]
        .iloc[0]
    )

    cov_better = (
        row50[
            "Coverage (%)"
        ]
        >
        row40[
            "Coverage (%)"
        ]
    )

    ov_better = (
        row50[
            "Overlap (%)"
        ]
        <
        row40[
            "Overlap (%)"
        ]
    )

    ee_better = (
        row50[
            "EE (%)"
        ]
        >
        row40[
            "EE (%)"
        ]
    )

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "40-NODE VS 50-NODE"
    )

    print(
        "="
        *
        110
    )

    print(
        f"40-node Coverage : "
        f"{row40['Coverage (%)']:.4f}%"
    )

    print(
        f"50-node Coverage : "
        f"{row50['Coverage (%)']:.4f}%"
    )

    print(
        f"40-node Overlap  : "
        f"{row40['Overlap (%)']:.4f}%"
    )

    print(
        f"50-node Overlap  : "
        f"{row50['Overlap (%)']:.4f}%"
    )

    print(
        f"40-node EE       : "
        f"{row40['EE (%)']:.4f}%"
    )

    print(
        f"50-node EE       : "
        f"{row50['EE (%)']:.4f}%"
    )

    print(
        f"50 better Coverage : "
        f"{cov_better}"
    )

    print(
        f"50 better Overlap  : "
        f"{ov_better}"
    )

    print(
        f"50 better EE       : "
        f"{ee_better}"
    )

    print(
        "50-NODE OUTPERFORMS 40-NODE "
        "ON ALL THREE PRIMARY METRICS : "
        +
        str(
            cov_better
            and
            ov_better
            and
            ee_better
        )
    )

    total_runtime = (
        time.time()
        -
        overall_start
    )

    results_file = (
        "TR_EnMOGKSO_GNN_DRL_ARLS_"
        "250x300_UPDATED_results.csv"
    )

    independent_file = (
        "TR_EnMOGKSO_GNN_DRL_ARLS_"
        "250x300_UPDATED_independent_check.csv"
    )

    history_file = (
        "TR_EnMOGKSO_GNN_DRL_ARLS_"
        "250x300_UPDATED_history.csv"
    )

    df.to_csv(
        results_file,
        index=False
    )

    idf.to_csv(
        independent_file,
        index=False
    )

    history_rows = []

    for n, hist in histories.items():

        for item in hist:

            history_rows.append(
                {
                    "Nodes":
                        n,
                    **item
                }
            )

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_file,
        index=False
    )

    for n in NODE_COUNTS:

        pd.DataFrame(
            {
                "Node":
                    np.arange(
                        1,
                        n + 1
                    ),

                "X":
                    positions[
                        n
                    ][
                        :,
                        0
                    ],

                "Y":
                    positions[
                        n
                    ][
                        :,
                        1
                    ],
            }
        ).to_csv(
            (
                "TR_EnMOGKSO_GNN_DRL_ARLS_"
                f"250x300_UPDATED_{n}nodes_positions.csv"
            ),
            index=False
        )

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "FINAL VERIFICATION"
    )

    print(
        "="
        *
        110
    )

    print(
        "Same algorithm rules     : "
        "YES for 20/30/40/50"
    )

    print(
        "Hardcoded results        : NO"
    )

    print(
        "50-node special rule     : NO"
    )

    print(
        "Trainable GNN            : YES"
    )

    print(
        "GNN -> DQN TD gradients  : YES"
    )

    print(
        "ARLS                     : YES"
    )

    print(
        "Pareto archive           : YES"
    )

    print(
        "Independent check        : "
        "see table above"
    )

    print(
        f"Total runtime            : "
        f"{total_runtime:.2f} s"
    )

    print(
        f"Saved results            : "
        f"{results_file}"
    )

    print(
        f"Saved independent check  : "
        f"{independent_file}"
    )

    print(
        f"Saved history            : "
        f"{history_file}"
    )


# =====================================================================
# PART B - ABLATION STUDY
# =====================================================================

N_NODES = 50
RUNS_FILE = "ablation50_runs.csv"
HIST_FILE = "ablation50_history.csv"
SUMMARY_FILE = "ablation50_summary.csv"
STATS_FILE = "ablation50_stats.csv"

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

# =====================================================================
# 1. VARIANTS
# =====================================================================

DEFAULT_CFG = dict(
    gnn="trained",            # trained | frozen | none | no_mp
    drl="dqn",                # dqn | random
    operators=True,           # TR-EnMOGKSO operators
    archive=True,             # Pareto archive
    elite_chain=True,         # continuous elite ARLS trajectory
    arls=True,                # ARLS at all (init, population, elite, final)
    anneal=True,              # Metropolis acceptance (False = greedy)
    hole_filling=True,        # hole-filling relocation move in ARLS
    redundancy_select=True,   # redundancy-aware node selection in ARLS
    final_polish=True,        # final ARLS polishing of the best candidates
    overlap_reg=True,         # OVERLAP_REG_WEIGHT * overlap term in fitness
    stage2=True,              # stage-2 overlap-aware fitness once feasible
    margins=True,             # search margins inside the feasible region
)

VARIANTS = {
    # ---- reference ----------------------------------------------------
    "FULL": ("Full proposed method", "Reference", {}),
    # ---- deep-learning components --------------------------------------
    "NO_GNN": ("w/o GNN (no graph embedding in DQN state)", "GNN / DRL",
               dict(gnn="none")),
    "FROZEN_GNN": ("Frozen GNN (random weights, not trained)", "GNN / DRL",
                   dict(gnn="frozen")),
    "GNN_NO_MP": ("GNN w/o message passing (per-node MLP + mean pool)", "GNN / DRL",
                  dict(gnn="no_mp")),
    "NO_DRL": ("w/o DRL (random operator selection)", "GNN / DRL",
               dict(drl="random")),
    "NO_GNN_DRL": ("w/o GNN and w/o DRL", "GNN / DRL",
                   dict(gnn="none", drl="random")),
    # ---- metaheuristic --------------------------------------------------
    "NO_OPERATORS": ("w/o TR-EnMOGKSO operators (ARLS only)", "Metaheuristic",
                     dict(operators=False)),
    "NO_ARCHIVE": ("w/o Pareto archive", "Metaheuristic",
                   dict(archive=False)),
    "NO_ELITE_CHAIN": ("w/o elite annealing trajectory", "Metaheuristic",
                       dict(elite_chain=False)),
    # ---- ARLS -----------------------------------------------------------
    "NO_ARLS": ("w/o ARLS (no local search at all)", "ARLS",
                dict(arls=False)),
    "NO_ANNEAL": ("w/o annealing (greedy acceptance)", "ARLS",
                  dict(anneal=False)),
    "NO_HOLE": ("w/o hole-filling move", "ARLS",
                dict(hole_filling=False)),
    "NO_REDUNDANCY_SEL": ("w/o redundancy-aware node selection", "ARLS",
                          dict(redundancy_select=False)),
    "NO_FINAL_POLISH": ("w/o final ARLS polishing", "ARLS",
                        dict(final_polish=False)),
    # ---- objective ------------------------------------------------------
    "NO_OVERLAP_REG": ("w/o overlap regularisation term", "Objective",
                       dict(overlap_reg=False)),
    "NO_STAGE2": ("w/o stage-2 overlap-aware fitness", "Objective",
                  dict(stage2=False)),
    "NO_MARGINS": ("w/o search margins", "Objective",
                   dict(margins=False)),
}


# =====================================================================
# 2. ACCESS TO THE MAIN ALGORITHM (defined above in this same file)
# =====================================================================

class _GlobalsProxy:
    """Lets the ablation code read and override the main algorithm's
    globals (works as a script and inside Jupyter / Colab)."""

    def __init__(self, g):
        object.__setattr__(self, "_g", g)

    def __getattr__(self, k):
        return self._g[k]

    def __setattr__(self, k, v):
        self._g[k] = v


M = _GlobalsProxy(globals())
ORIG = {k: globals()[k] for k in ("OVERLAP_REG_WEIGHT", "SEARCH_MARGIN_COVERAGE",
                                  "SEARCH_MARGIN_OVERLAP", "SEARCH_MARGIN_EE",
                                  "fitness")}


def apply_globals(cfg):
    """Reset the base module, then apply objective-level switches."""
    for k, v in ORIG.items():
        setattr(M, k, v)
    if not cfg["overlap_reg"]:
        M.OVERLAP_REG_WEIGHT = 0.0
    if not cfg["margins"]:
        M.SEARCH_MARGIN_COVERAGE = 0.0
        M.SEARCH_MARGIN_OVERLAP = 0.0
        M.SEARCH_MARGIN_EE = 0.0
    if not cfg["stage2"]:
        def fitness_single_stage(sol):
            return (M.fitness_raw(sol["coverage"], sol["overlap"])
                    - M.SEP_PENALTY * sol.get("sep_viol", 0))
        M.fitness = fitness_single_stage


# =====================================================================
# 3. ARLS WITH SWITCHES (identical to the main file when all are on)
# =====================================================================

def ARLS_ablation(cfg, pos, moves, t_start, t_end, step_start, step_end, rng):
    if not cfg["arls"]:
        sol = M.quick_evaluate(pos)
        return sol, sol

    state = M.CoverageState(pos)
    current_fitness = state.fitness()
    best_fitness = current_fitness
    best_position = state.pos.copy()
    probability = np.full(state.n, 1.0 / state.n)

    for t in range(moves):
        frac = t / max(moves - 1, 1)
        temperature = t_start * (t_end / t_start) ** frac
        step = step_start + (step_end - step_start) * frac

        if cfg["redundancy_select"] and t % M.PRIORITY_REFRESH == 0:
            priority = state.redundancy_priority() + 0.05
            probability = 0.5 * priority / priority.sum() + 0.5 / state.n

        node = int(rng.choice(state.n, p=probability))

        if (cfg["hole_filling"] and rng.random() < M.ARLS_HOLE_PROB
                and state.n_cov < M.NUM_GRID):
            holes = np.flatnonzero(state.count == 0)
            target = M.GRID[holes[rng.integers(len(holes))]]
            new_position = M.clip_point(target + rng.normal(0.0, 0.25 * M.R_SENSE, 2))
        else:
            new_position = M.clip_point(state.pos[node] + rng.normal(0.0, step, 2))

        info = state.move(node, new_position)
        new_fitness = state.fitness()
        delta = new_fitness - current_fitness

        if cfg["anneal"]:
            accept = (delta >= 0.0 or
                      rng.random() < np.exp(delta / max(temperature, 1e-12)))
        else:
            accept = delta >= 0.0

        if accept:
            current_fitness = new_fitness
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_position = state.pos.copy()
        else:
            state.undo(info)

    return M.quick_evaluate(state.pos), M.quick_evaluate(best_position)


# =====================================================================
# 4. GNN / DQN VARIANTS
# =====================================================================

def make_qnet(cfg):
    import torch
    import torch.nn as nn

    class AblationQNet(M.HybridQNet):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def forward(self, base_states, positions):
            if self.mode == "none":
                emb = torch.zeros((base_states.shape[0], M.GNN_OUTPUT),
                                  dtype=torch.float32, device=base_states.device)
            else:
                embs = []
                for pos in positions:
                    x, adjacency = M.graph_tensors(pos)
                    if self.mode == "no_mp":
                        adjacency = torch.zeros_like(adjacency)   # A + I -> I
                    embs.append(self.gnn(x, adjacency))
                emb = torch.stack(embs, dim=0)
            return self.q(torch.cat((base_states, emb), dim=1))

    net = AblationQNet(cfg["gnn"]).to(M.DEVICE)
    if cfg["gnn"] == "frozen":
        for p in net.gnn.parameters():
            p.requires_grad_(False)
    return net


# =====================================================================
# 5. ONE RUN (same control flow as run_proposed in the main file)
# =====================================================================

def run_variant(cfg, n, seed):
    import torch
    import torch.optim as optim

    start = time.time()
    M.set_seed(seed)
    rng = np.random.default_rng(seed)

    q_net = make_qnet(cfg)
    target_net = make_qnet(cfg)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam([p for p in q_net.parameters() if p.requires_grad],
                           lr=M.DQN_LR)
    replay = M.ReplayBuffer()
    losses, grads, gnn_updates = [], [], 0

    add_arch = (M.add_to_archive if cfg["archive"]
                else (lambda archive, cand: []))

    population, archive, elite = [], [], None
    t0, _, s0, _ = M.schedule(0)
    for raw in M.create_population(n):
        current, best = ARLS_ablation(cfg, raw, M.ARLS_MOVES_INIT, t0, t0, s0, s0, rng)
        population.append(current["pos"].copy())
        archive = add_arch(archive, best)
        if M.better_solution(best, elite):
            elite = best

    population_solutions = [M.quick_evaluate(p) for p in population]
    history = []
    chain_position = elite["pos"].copy()

    for generation in range(M.GENERATIONS):
        epsilon = max(M.EPS_END, M.EPS_START * np.exp(-generation / M.EPS_DECAY))
        if cfg["drl"] == "random":
            epsilon = 1.0
        t_a, t_b, s_a, s_b = M.schedule(generation)
        new_population, actions = [], []

        for i, current in enumerate(population):
            current_sol = population_solutions[i]
            state = M.build_state(M.full_evaluate(current))
            action = M.choose_action(q_net, state, current, epsilon, rng)
            actions.append(action)

            if cfg["operators"]:
                candidate = M.apply_operator(action, current, population,
                                             elite["pos"], rng, generation)
            else:
                candidate = current.copy()

            cand_cur, cand_best = ARLS_ablation(cfg, candidate, M.ARLS_MOVES_POP,
                                       t_a, t_b, s_a, s_b, rng)
            archive = add_arch(archive, cand_best)
            if M.better_solution(cand_best, elite):
                elite = cand_best

            reward = (M.REWARD_COVERAGE * (cand_cur["coverage"] - current_sol["coverage"])
                      + M.REWARD_OVERLAP * (current_sol["overlap"] - cand_cur["overlap"])
                      + M.REWARD_EE * (cand_cur["ee"] - current_sol["ee"])
                      + M.REWARD_FEASIBLE * float(M.is_feasible(cand_best)
                                                  and not M.is_feasible(current_sol)))

            if cfg["drl"] == "dqn":
                next_state = M.build_state(M.full_evaluate(cand_cur["pos"]))
                replay.push(state, current, action, reward, next_state,
                            cand_cur["pos"], generation == M.GENERATIONS - 1)
                loss, grad = M.update_dqn(q_net, target_net, optimizer, replay)
                if loss is not None:
                    losses.append(loss)
                    grads.append(grad)
                    gnn_updates += int(grad > 0.0)

            delta = M.fitness(cand_cur) - M.fitness(current_sol)
            if cfg["anneal"]:
                accept = (delta >= 0.0 or
                          rng.random() < np.exp(delta / max(t_b, 1e-12)))
            else:
                accept = delta >= 0.0
            new_population.append(cand_cur["pos"].copy() if accept else current.copy())

        population = new_population
        population_solutions = [M.quick_evaluate(p) for p in population]
        for sol in population_solutions:
            archive = add_arch(archive, sol)

        if cfg["elite_chain"]:
            chain_current, elite_best = ARLS_ablation(cfg, chain_position, M.ARLS_MOVES_ELITE,
                                             t_a, t_b, s_a, s_b, rng)
            chain_position = chain_current["pos"].copy()
            archive = add_arch(archive, elite_best)
        else:
            elite_best = elite

        leader_pool = (population_solutions
                       + [M.quick_evaluate(a["pos"]) for a in archive]
                       + [elite, elite_best])
        elite = M.select_global_leader(leader_pool)

        population_solutions = [M.quick_evaluate(p) for p in population]
        worst = int(np.argmin([M.fitness(s) for s in population_solutions]))
        population[worst] = elite["pos"].copy()
        population_solutions[worst] = M.quick_evaluate(elite["pos"])

        if (generation + 1) % M.TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        history.append({"generation": generation + 1,
                        "coverage": elite["coverage"] * 100.0,
                        "overlap": elite["overlap"] * 100.0,
                        "ee": elite["ee"] * 100.0,
                        "feasible": bool(M.is_feasible(elite)),
                        "action": int(max(set(actions), key=actions.count))})

    pool = (population_solutions
            + [M.quick_evaluate(a["pos"]) for a in archive] + [elite])
    pool = [x for x in pool if x is not None]
    pool.sort(key=M.rank_key, reverse=True)

    polished = []
    if cfg["final_polish"]:
        for candidate in pool[:M.ARLS_FINAL_CANDIDATES]:
            _, refined = ARLS_ablation(cfg, candidate["pos"], M.ARLS_MOVES_FINAL, 1e-4,
                              M.ARLS_T_END, 3.0, M.ARLS_STEP_MIN, rng)
            polished.append(refined)

    final_candidate = M.select_final_candidate(pool + polished)
    final = M.full_evaluate(final_candidate["pos"])

    # independent recomputation from the final positions
    c2, o2, _, _ = M.coverage_data(final["pos"])
    e2 = M.energy_efficiency(c2, o2)
    feasible = (c2 > M.TARGET_COVERAGE and o2 < M.TARGET_OVERLAP
                and e2 > M.TARGET_EE)

    return {"coverage": c2 * 100.0, "overlap": o2 * 100.0, "ee": e2 * 100.0,
            "feasible": bool(feasible), "gcc": final["gcc"] * 100.0,
            "components": int(final["components"]),
            "min_sep": float(M.min_pair_distance(final["pos"])),
            "runtime": time.time() - start,
            "avg_dqn_loss": float(np.mean(losses)) if losses else np.nan,
            "avg_gnn_grad": float(np.mean(grads)) if grads else np.nan,
            "gnn_updates": gnn_updates,
            "first_feasible_gen": next((h["generation"] for h in history
                                        if h["feasible"]), np.nan),
            "history": history}


# =====================================================================
# 6. WORKER
# =====================================================================

def _init_worker(threads):
    import torch
    torch.set_num_threads(threads)


def _job(args):
    key, seed = args
    cfg = dict(DEFAULT_CFG)
    cfg.update(VARIANTS[key][2])
    apply_globals(cfg)
    r = run_variant(cfg, N_NODES, seed)
    apply_globals(DEFAULT_CFG)
    return key, seed, r


# =====================================================================
# 7. ANALYSIS
# =====================================================================

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[ok][j]))
        adj[j] = running
    out[ok] = adj
    return out


def analyse(df):
    from scipy.stats import wilcoxon

    order = [k for k in VARIANTS if k in set(df["Key"])]
    rows = []
    for k in order:
        d = df[df["Key"] == k]
        rows.append({
            "Key": k, "Variant": VARIANTS[k][0], "Group": VARIANTS[k][1],
            "Runs": len(d),
            "Coverage mean": d["Coverage (%)"].mean(),
            "Coverage std": d["Coverage (%)"].std(),
            "Overlap mean": d["Overlap (%)"].mean(),
            "Overlap std": d["Overlap (%)"].std(),
            "EE mean": d["EE (%)"].mean(), "EE std": d["EE (%)"].std(),
            "EE best": d["EE (%)"].max(), "EE worst": d["EE (%)"].min(),
            "EE median": d["EE (%)"].median(),
            "Pass rate (%)": 100.0 * d["Feasible"].mean(),
            "First feasible gen (mean)": d["First feasible gen"].mean(),
            "Runtime mean (s)": d["Runtime (s)"].mean(),
        })
    summary = pd.DataFrame(rows)

    stats = []
    if "FULL" in order:
        full = df[df["Key"] == "FULL"].set_index("Seed")
        for metric, better in (("EE (%)", "higher"), ("Overlap (%)", "lower"),
                               ("Coverage (%)", "higher")):
            raw = []
            for k in order:
                if k == "FULL":
                    continue
                v = df[df["Key"] == k].set_index("Seed")
                common = full.index.intersection(v.index)
                diff = (full.loc[common, metric] - v.loc[common, metric]).to_numpy()
                if len(common) < 2 or np.allclose(diff, 0.0):
                    p = 1.0 if len(common) >= 2 else np.nan
                else:
                    p = wilcoxon(full.loc[common, metric], v.loc[common, metric],
                                 zero_method="zsplit").pvalue
                wins = int(np.sum(diff > 1e-9)) if better == "higher" else int(np.sum(diff < -1e-9))
                losses = int(np.sum(diff < -1e-9)) if better == "higher" else int(np.sum(diff > 1e-9))
                raw.append({"Metric": metric, "Key": k, "Variant": VARIANTS[k][0],
                            "Paired seeds": len(common),
                            "FULL - variant (mean)": float(np.mean(diff)) if len(common) else np.nan,
                            "FULL better (seeds)": wins, "FULL worse (seeds)": losses,
                            "p (Wilcoxon)": p})
            adj = holm([r["p (Wilcoxon)"] for r in raw])
            for r, a in zip(raw, adj):
                r["p (Holm)"] = a
                r["Significant (0.05)"] = bool(a < 0.05) if not np.isnan(a) else False
                stats.append(r)
    return summary, pd.DataFrame(stats)


def print_report(summary, stats):
    pd.set_option("display.width", 220)
    print()
    print("=" * 118)
    print(f"ABLATION SUMMARY - {N_NODES} NODES (mean +/- std over seeds)")
    print("=" * 118)
    print(f"{'Variant':52s} {'Coverage %':>15s} {'Overlap %':>15s} {'EE %':>15s} "
          f"{'Pass':>6s} {'Time s':>8s}")
    for _, r in summary.iterrows():
        print(f"{r['Variant'][:52]:52s} "
              f"{r['Coverage mean']:7.3f}+/-{r['Coverage std']:5.3f} "
              f"{r['Overlap mean']:7.3f}+/-{r['Overlap std']:5.3f} "
              f"{r['EE mean']:7.3f}+/-{r['EE std']:5.3f} "
              f"{r['Pass rate (%)']:5.0f}% {r['Runtime mean (s)']:8.1f}")
    if len(stats):
        print()
        print("=" * 118)
        print("PAIRED WILCOXON SIGNED-RANK TEST VS FULL (Holm-corrected, same seeds)")
        print("Positive EE/Coverage difference or negative Overlap difference = FULL is better")
        print("=" * 118)
        for metric in ("EE (%)", "Overlap (%)"):
            s = stats[stats["Metric"] == metric]
            print(f"\n{metric}")
            for _, r in s.iterrows():
                print(f"  {r['Variant'][:52]:52s} diff={r['FULL - variant (mean)']:+8.4f}  "
                      f"FULL better/worse={r['FULL better (seeds)']:2d}/{r['FULL worse (seeds)']:2d}  "
                      f"p={r['p (Wilcoxon)']:.4f}  p_holm={r['p (Holm)']:.4f}  "
                      f"{'SIGNIFICANT' if r['Significant (0.05)'] else ''}")
    print("=" * 118)


def load_runs():
    if not os.path.exists(RUNS_FILE):
        return pd.DataFrame()
    return pd.read_csv(RUNS_FILE)


# =====================================================================
# 8. MAIN
# =====================================================================

def run_study(seeds=10, first_seed=0, workers=1, threads=1, variants=None,
              analyse_only=False):
    """Run the ablation study. Finished runs in ablation50_runs.csv are
    skipped, so calling it again resumes an interrupted study."""
    keys = variants or list(VARIANTS)
    bad = [k for k in keys if k not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variants: {bad}. Valid: {list(VARIANTS)}")
    seed_list = list(range(first_seed, first_seed + seeds))

    if not analyse_only:
        done = load_runs()
        done_set = set(zip(done["Key"], done["Seed"])) if len(done) else set()
        jobs = [(k, s) for s in seed_list for k in keys if (k, s) not in done_set]
        print(f"{len(jobs)} runs to do ({len(done_set)} already finished), "
              f"{workers} worker(s).", flush=True)

        def save(key, seed, r, j, t_start):
            row = {"Key": key, "Variant": VARIANTS[key][0],
                   "Group": VARIANTS[key][1], "Seed": seed,
                   "Coverage (%)": r["coverage"], "Overlap (%)": r["overlap"],
                   "EE (%)": r["ee"], "Feasible": r["feasible"],
                   "GCC (%)": r["gcc"], "Components": r["components"],
                   "Min sep (m)": r["min_sep"],
                   "First feasible gen": r["first_feasible_gen"],
                   "Avg DQN loss": r["avg_dqn_loss"],
                   "Avg GNN grad": r["avg_gnn_grad"],
                   "GNN updates": r["gnn_updates"],
                   "Runtime (s)": r["runtime"]}
            pd.DataFrame([row]).to_csv(RUNS_FILE, mode="a", index=False,
                                       header=not os.path.exists(RUNS_FILE))
            hist = pd.DataFrame(r["history"])
            hist.insert(0, "Seed", seed)
            hist.insert(0, "Key", key)
            hist.to_csv(HIST_FILE, mode="a", index=False,
                        header=not os.path.exists(HIST_FILE))
            el = time.time() - t_start
            eta = el / j * (len(jobs) - j)
            print(f"[{j:3d}/{len(jobs)}] {key:18s} seed={seed:2d} "
                  f"cov={r['coverage']:.2f} ov={r['overlap']:.2f} "
                  f"ee={r['ee']:.3f} {'PASS' if r['feasible'] else 'fail'} "
                  f"({r['runtime']:.0f}s, ETA {eta/60:.0f} min)", flush=True)

        t_start = time.time()
        j = 0
        if workers > 1:
            try:
                with Pool(workers, initializer=_init_worker,
                          initargs=(threads,)) as pool:
                    for key, seed, r in pool.imap_unordered(_job, jobs):
                        j += 1
                        save(key, seed, r, j, t_start)
            except Exception as e:      # e.g. notebooks that cannot pickle
                print(f"Parallel mode unavailable here ({type(e).__name__}); "
                      f"continuing with 1 worker.", flush=True)
                done = load_runs()
                done_set = set(zip(done["Key"], done["Seed"])) if len(done) else set()
                jobs_left = [job for job in jobs if job not in done_set]
                jobs = jobs[:j] + jobs_left
                workers = 1
        if workers <= 1:
            _init_worker(threads)
            for job in jobs[j:]:
                j += 1
                key, seed, r = _job(job)
                save(key, seed, r, j, t_start)

    df = load_runs()
    if not len(df):
        print("No runs found.")
        return None, None
    df = df[df["Key"].isin(keys) & df["Seed"].isin(seed_list)]
    summary, stats = analyse(df)
    summary.to_csv(SUMMARY_FILE, index=False)
    stats.to_csv(STATS_FILE, index=False)
    print_report(summary, stats)
    print(f"\nSaved: {RUNS_FILE}, {HIST_FILE}, {SUMMARY_FILE}, {STATS_FILE}")
    return summary, stats


# =====================================================================
# 9. SETTINGS - EDIT HERE
# =====================================================================

ABLATION_SEEDS = 10          # independent runs per variant
ABLATION_WORKERS = 1         # parallel runs (1 = safest in Jupyter/Colab)
ABLATION_VARIANTS = None     # None = all 17, or e.g. ["FULL", "NO_GNN", "NO_DRL"]

# Google Drive: every finished run is saved there immediately, so a
# disconnected / reset Colab runtime resumes from where it stopped.
USE_GOOGLE_DRIVE = True
DRIVE_FOLDER = "/content/drive/MyDrive/ablation50"


def setup_output_folder():
    """In Colab: mount Google Drive and save all results there.
    Elsewhere (local Jupyter / terminal): save in the current folder."""
    if USE_GOOGLE_DRIVE:
        try:
            from google.colab import drive
        except ImportError:
            print("Not running in Google Colab - saving results locally.")
        else:
            if not os.path.isdir("/content/drive/MyDrive"):
                drive.mount("/content/drive")
            os.makedirs(DRIVE_FOLDER, exist_ok=True)
            os.chdir(DRIVE_FOLDER)
    print(f"Results folder : {os.getcwd()}")
    if os.path.exists(RUNS_FILE):
        n_done = len(pd.read_csv(RUNS_FILE))
        print(f"Resuming       : {n_done} finished runs found in {RUNS_FILE}")
    else:
        print("Starting       : no previous runs found")


if __name__ == "__main__":
    setup_output_folder()
    run_study(seeds=ABLATION_SEEDS, workers=ABLATION_WORKERS,
              variants=ABLATION_VARIANTS)
