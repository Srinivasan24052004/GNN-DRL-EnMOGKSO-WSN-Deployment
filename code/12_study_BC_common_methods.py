# =====================================================================
# TR-EnMOGKSO + GNN + DRL + LEARNED ARLS - FINAL EXPERIMENT SUITE
# 250 x 300 m, sensing 25 m, population 6, 30 generations
#
# PART A : original algorithm (unchanged; used as the ORIGINAL baseline)
# PART B : learned ARLS. Proposed = "gnn_dqn": a GNN encodes the sensor
#          graph for a Move-DQN that chooses how ARLS moves sensors; the
#          GNN is trained end-to-end through the DQN's TD loss.
# PART C : EXP 1 final test (30 new seeds), EXP 2 scalability (20-50
#          nodes), EXP 3 PSO/GA/DE/GWO/WOA baselines (equal time),
#          statistics (Wilcoxon, Holm, A12, Friedman) and PNG figures.
# Kaggle: paste in ONE cell -> Save Version -> Save & Run All (Commit).
# Expected time ~6-7 h. Outputs: CSVs, final_results_tables.xlsx, fig*.png
# =====================================================================

# =====================================================================
# PART A - ORIGINAL ALGORITHM (unchanged)
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
GRID_RES = 110                       # OPTION 2: finer optimisation grid (was 50)
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
# PART B - LEARNED ARLS: the GNN and the DQN control the local search
# =====================================================================
#
# In Part A the GNN/DQN only choose 1 of 5 TR-EnMOGKSO operators
# (~180 decisions per run) while ARLS does almost all the work with
# hand-made rules. Here ARLS itself is driven by the learned components:
#
#   * NODE-GNN (trained online): graph convolution over the
#     disk-intersection graph of the current deployment. Per-node
#     features: position, unique / redundant coverage share, share of
#     the disk inside the field, boundary distance, nearby uncovered
#     points. It scores every node and ARLS samples WHICH node to move
#     (score re-weights the redundancy prior). It is trained on the
#     fitness gains its chosen moves produced.
#   * MOVE-DQN (trained online): before every block of ARLS moves it
#     chooses HOW to move (5 move operators) from a state that includes
#     the Node-GNN graph embedding. Reward = fitness gain of the block.
#
#   Both networks persist for the whole run (all individuals, all
#   generations), so they make thousands of decisions per run.
#
# The outer GNN + DQN of Part A (operator selection) are unchanged.
# =====================================================================

ARLS_MODE = "gnn_dqn"     # "gnn_dqn"  : GNN encodes the deployment graph for the
                          #              Move-DQN, trained end-to-end by TD loss
                          # "learned"  : GNN chooses node + DQN chooses move
                          # "no_gnn"   : DQN only, hand-made node choice
                          # "no_drl"   : GNN only, random move operator
                          # "original" : original Part-A ARLS (baseline)

ARLS_BLOCK = 100          # moves per DQN decision
NODE_FEAT = 7
NGNN_HID, NGNN_EMB = 32, 16
NGNN_LR = 1e-3
NGNN_BUFFER, NGNN_BATCH, NGNN_TRAIN_EVERY = 400, 8, 2
NGNN_TARGET_SCALE, NGNN_TARGET_CLIP = 1.0e4, 20.0
NODE_UNIFORM_MIX = 0.5

MOVE_OPS = ["Scheduled step", "Hole filling", "Small step",
            "Redundancy push", "Boundary push"]
N_MOVE_OPS = len(MOVE_OPS)
MDQN_STATE = 8 + N_MOVE_OPS + NGNN_EMB
MDQN_HIDDEN = 64
MDQN_LR, MDQN_GAMMA = 1e-3, 0.9
MDQN_BUFFER, MDQN_BATCH, MDQN_TARGET_EVERY = 5000, 32, 100
MDQN_EPS_START, MDQN_EPS_END, MDQN_EPS_DECAY = 0.5, 0.05, 800.0
MDQN_REWARD_SCALE = 1.0e3

ARLS_ORIGINAL = ARLS                 # keep the Part-A local search
RUN_PROPOSED_ORIGINAL = run_proposed
POLICY = None
MAX_DISK = len(disk_indices(np.array([AREA_X / 2.0, AREA_Y / 2.0])))


class NodeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(NODE_FEAT, NGNN_HID)
        self.g2 = nn.Linear(NGNN_HID, NGNN_EMB)
        self.head = nn.Sequential(nn.Linear(NGNN_EMB + NODE_FEAT, 32),
                                  nn.ReLU(), nn.Linear(32, 1))
        self.act = nn.ReLU()

    def forward(self, x, a_norm):
        h = self.act(self.g1(a_norm @ x))
        h = self.act(self.g2(a_norm @ h))
        return self.head(torch.cat([h, x], dim=-1)).squeeze(-1), h


class MoveDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(MDQN_STATE, MDQN_HIDDEN), nn.ReLU(),
                                 nn.Linear(MDQN_HIDDEN, MDQN_HIDDEN), nn.ReLU(),
                                 nn.Linear(MDQN_HIDDEN, N_MOVE_OPS))

    def forward(self, x):
        return self.net(x)


def disk_graph(pos):
    a = (pairwise_distances(pos) < 2.0 * R_SENSE).astype(np.float32)
    np.fill_diagonal(a, 1.0)
    d = 1.0 / np.sqrt(a.sum(axis=1))
    return (a * d[:, None] * d[None, :]).astype(np.float32)


def node_features(st):
    f = np.zeros((st.n, NODE_FEAT), dtype=np.float32)
    f[:, 0] = st.pos[:, 0] / AREA_X
    f[:, 1] = st.pos[:, 1] / AREA_Y
    for k, ind in enumerate(st.idx):
        ck = st.count[ind]
        sz = max(len(ind), 1)
        f[k, 2] = np.count_nonzero(ck == 1) / sz
        f[k, 3] = np.count_nonzero(ck >= 2) / sz
        f[k, 4] = len(ind) / MAX_DISK
    edge = np.minimum.reduce([st.pos[:, 0], AREA_X - st.pos[:, 0],
                              st.pos[:, 1], AREA_Y - st.pos[:, 1]])
    f[:, 5] = np.minimum(edge / (2.0 * R_SENSE), 1.0)
    holes = GRID[st.count == 0]
    if len(holes):
        d2 = ((st.pos[:, None, :] - holes[None, :, :]) ** 2).sum(axis=2)
        f[:, 6] = np.minimum(np.count_nonzero(d2 < (2 * R_SENSE) ** 2, axis=1)
                             / MAX_DISK, 1.0)
    return f


class GraphMoveQNet(nn.Module):
    """Move-DQN whose state includes a GNN embedding of the deployment
    graph. Gradients of the TD loss flow into the GNN (end-to-end)."""

    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(NODE_FEAT, NGNN_HID)
        self.g2 = nn.Linear(NGNN_HID, NGNN_EMB)
        self.q = nn.Sequential(nn.Linear(8 + N_MOVE_OPS + NGNN_EMB, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, N_MOVE_OPS))
        self.act = nn.ReLU()

    def embed(self, x, a):                       # x: [B,N,F], a: [B,N,N]
        h = self.act(self.g1(torch.bmm(a, x)))
        h = self.act(self.g2(torch.bmm(a, h)))
        return h.mean(dim=1)                     # [B, NGNN_EMB]

    def forward(self, base, x, a):
        return self.q(torch.cat([base, self.embed(x, a)], dim=1))


class LearnedSearchPolicy:
    """Node-GNN + Move-DQN shared by every ARLS call of one run."""

    def __init__(self, mode):
        self.mode = mode
        self.gnn = NodeGNN().to(DEVICE)
        self.gnn_opt = optim.Adam(self.gnn.parameters(), lr=NGNN_LR)
        self.gnn_buf = deque(maxlen=NGNN_BUFFER)
        self.gnn_losses = []
        qcls = GraphMoveQNet if mode == "gnn_dqn" else MoveDQN
        self.q = qcls().to(DEVICE)
        self.q_t = qcls().to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.q_opt = optim.Adam(self.q.parameters(), lr=MDQN_LR)
        self.q_buf = deque(maxlen=MDQN_BUFFER)
        self.q_losses = []
        self.decisions = 0
        self.blocks = 0
        self.op_counts = np.zeros((2, N_MOVE_OPS), dtype=np.int64)

    # ---------------- Node-GNN: which node to move ----------------------
    def node_probs(self, st):
        prior = st.redundancy_priority() + 0.05
        if self.mode == "gnn_dqn":
            p = (1.0 - NODE_UNIFORM_MIX) * prior / prior.sum() + NODE_UNIFORM_MIX / st.n
            return p, node_features(st), disk_graph(st.pos), np.zeros(NGNN_EMB, np.float32)
        if self.mode == "no_gnn":
            p = (1.0 - NODE_UNIFORM_MIX) * prior / prior.sum() + NODE_UNIFORM_MIX / st.n
            return p, None, None, np.zeros(NGNN_EMB, np.float32)
        x, a = node_features(st), disk_graph(st.pos)
        with torch.no_grad():
            s, h = self.gnn(torch.from_numpy(x).to(DEVICE),
                            torch.from_numpy(a).to(DEVICE))
        s = s.cpu().numpy().astype(np.float64)
        z = (s - s.max()) / (s.std() + 1e-6)
        p = prior * np.exp(z)
        p = (1.0 - NODE_UNIFORM_MIX) * p / p.sum() + NODE_UNIFORM_MIX / st.n
        return p, x, a, h.mean(dim=0).cpu().numpy().astype(np.float32)

    def learn_nodes(self, rng, x, a, nodes, targets):
        if self.mode in ("no_gnn", "gnn_dqn") or x is None or len(nodes) == 0:
            return
        self.gnn_buf.append((x, a, nodes, targets.astype(np.float32)))
        self.blocks += 1
        if self.blocks % NGNN_TRAIN_EVERY or len(self.gnn_buf) < NGNN_BATCH:
            return
        loss = 0.0
        for j in rng.choice(len(self.gnn_buf), NGNN_BATCH, replace=False):
            xb, ab, nb, tb = self.gnn_buf[j]
            s, _ = self.gnn(torch.from_numpy(xb).to(DEVICE),
                            torch.from_numpy(ab).to(DEVICE))
            loss = loss + nn.functional.mse_loss(
                s[torch.from_numpy(nb).to(DEVICE)], torch.from_numpy(tb).to(DEVICE))
        loss = loss / NGNN_BATCH
        self.gnn_opt.zero_grad()
        loss.backward()
        self.gnn_opt.step()
        self.gnn_losses.append(float(loss.item()))

    # ---------------- Move-DQN: how to move it --------------------------
    def state(self, st, emb, progress, T, acc_rate, last_gain, last_op):
        c, o = st.metrics()
        onehot = np.zeros(N_MOVE_OPS, np.float32)
        if last_op is not None:
            onehot[last_op] = 1.0
        base = np.array([c, o, energy_efficiency(c, o), st.fitness(), progress,
                         np.log10(max(T, 1e-12)) / 5.0, acc_rate,
                         np.clip(last_gain * MDQN_REWARD_SCALE, -5, 5)], np.float32)
        return np.concatenate([base, onehot, emb]).astype(np.float32)

    def choose_op(self, rng, s, progress, graph=None):
        if self.mode == "no_drl":
            op = int(rng.integers(N_MOVE_OPS))
        else:
            eps = max(MDQN_EPS_END,
                      MDQN_EPS_START * np.exp(-self.decisions / MDQN_EPS_DECAY))
            if rng.random() < eps:
                op = int(rng.integers(N_MOVE_OPS))
            else:
                with torch.no_grad():
                    if self.mode == "gnn_dqn":
                        qv = self.q(torch.from_numpy(s[:8 + N_MOVE_OPS]).unsqueeze(0).to(DEVICE),
                                    torch.from_numpy(graph[0]).unsqueeze(0).to(DEVICE),
                                    torch.from_numpy(graph[1]).unsqueeze(0).to(DEVICE))
                    else:
                        qv = self.q(torch.from_numpy(s).unsqueeze(0).to(DEVICE))
                op = int(torch.argmax(qv, dim=1).item())
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        if self.mode == "no_drl":
            return
        if self.mode == "gnn_dqn":
            return self._learn_op_graph(rng, s, op, r, s2, graph, graph2)
        self.q_buf.append((s, op, r, s2))
        self.decisions += 1
        if len(self.q_buf) >= MDQN_BATCH:
            idx = rng.choice(len(self.q_buf), MDQN_BATCH, replace=False)
            b = [self.q_buf[j] for j in idx]
            st = torch.from_numpy(np.stack([t[0] for t in b])).to(DEVICE)
            ac = torch.tensor([t[1] for t in b], device=DEVICE).unsqueeze(1)
            rw = torch.tensor([t[2] for t in b], dtype=torch.float32, device=DEVICE)
            ns = torch.from_numpy(np.stack([t[3] for t in b])).to(DEVICE)
            qv = self.q(st).gather(1, ac).squeeze(1)
            with torch.no_grad():
                tgt = rw + MDQN_GAMMA * self.q_t(ns).max(dim=1).values
            loss = nn.functional.smooth_l1_loss(qv, tgt)
            self.q_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q_opt.step()
            self.q_losses.append(float(loss.item()))
        if self.decisions % MDQN_TARGET_EVERY == 0:
            self.q_t.load_state_dict(self.q.state_dict())


def _learn_op_graph(self, rng, s, op, r, s2, graph, graph2):
    k = 8 + N_MOVE_OPS
    self.q_buf.append((s[:k], graph[0], graph[1], op, r, s2[:k], graph2[0], graph2[1]))
    self.decisions += 1
    if len(self.q_buf) >= MDQN_BATCH and self.decisions % GQ_TRAIN_EVERY == 0:
        idx = rng.choice(len(self.q_buf), GQ_BATCH, replace=False)
        b = [self.q_buf[j] for j in idx]
        T = lambda arr: torch.from_numpy(np.stack(arr)).to(DEVICE)
        bs, bx, ba = T([t[0] for t in b]), T([t[1] for t in b]), T([t[2] for t in b])
        ac = torch.tensor([t[3] for t in b], device=DEVICE).unsqueeze(1)
        rw = torch.tensor([t[4] for t in b], dtype=torch.float32, device=DEVICE)
        ns, nx, na = T([t[5] for t in b]), T([t[6] for t in b]), T([t[7] for t in b])
        qv = self.q(bs, bx, ba).gather(1, ac).squeeze(1)
        with torch.no_grad():
            tgt = rw + MDQN_GAMMA * self.q_t(ns, nx, na).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(qv, tgt)
        self.q_opt.zero_grad()
        loss.backward()                      # TD gradients reach the GNN layers
        gnorm = sum(float(p.grad.norm()) for p in (self.q.g1.weight, self.q.g2.weight)
                    if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.q_opt.step()
        self.q_losses.append(float(loss.item()))
        self.gnn_losses.append(gnorm)        # logged as GNN gradient norm
    if self.decisions % MDQN_TARGET_EVERY == 0:
        self.q_t.load_state_dict(self.q.state_dict())


LearnedSearchPolicy._learn_op_graph = _learn_op_graph
GQ_TRAIN_EVERY = 2          # graph-DQN update every 2 decisions
GQ_BATCH = 16


def propose_move(op, st, i, step, rng):
    p = st.pos[i]
    if op == 1 and st.n_cov < NUM_GRID:                       # hole filling
        holes = np.flatnonzero(st.count == 0)
        target = GRID[holes[rng.integers(len(holes))]]
        return clip_point(target + rng.normal(0.0, 0.25 * R_SENSE, 2))
    if op == 2:                                               # small step
        return clip_point(p + rng.normal(0.0, max(ARLS_STEP_MIN, 0.3 * step), 2))
    if op == 3:                                               # redundancy push
        d = p[None, :] - st.pos
        dist = np.hypot(d[:, 0], d[:, 1])
        w = np.clip(2.0 * R_SENSE - dist, 0.0, None)
        w[i] = 0.0
        v = (d * (w / np.maximum(dist, 1e-6))[:, None]).sum(axis=0)
        nv = np.hypot(v[0], v[1])
        if nv > 1e-9:
            return clip_point(p + max(step, 2.0) * v / nv + rng.normal(0.0, 1.0, 2))
    if op == 4:                                               # boundary push
        gaps = np.array([p[0], AREA_X - p[0], p[1], AREA_Y - p[1]])
        dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)
        return clip_point(p + dirs[int(np.argmin(gaps))] * rng.uniform(0.5, 1.0)
                          * max(step, 2.0) + rng.normal(0.0, 1.0, 2))
    return clip_point(p + rng.normal(0.0, step, 2))           # scheduled step


def ARLS(pos, moves, t_start, t_end, step_start, step_end, rng):
    """Policy-driven ARLS (same signature as Part A, so Part A uses it)."""
    if ARLS_MODE == "original" or POLICY is None:
        return ARLS_ORIGINAL(pos, moves, t_start, t_end, step_start, step_end, rng)

    pol = POLICY
    st = CoverageState(pos)
    f_cur = st.fitness()
    best_f, best_pos = f_cur, st.pos.copy()
    n_blocks = max(1, int(np.ceil(moves / ARLS_BLOCK)))
    acc_rate, last_gain, last_op = 0.0, 0.0, None

    for b in range(n_blocks):
        frac = b / max(n_blocks - 1, 1)
        T = t_start * (t_end / t_start) ** frac
        step = step_start + (step_end - step_start) * frac
        progress = float(np.clip(np.log(max(T, 1e-12) / ARLS_T_START)
                                 / np.log(ARLS_T_END / ARLS_T_START), 0.0, 1.0))

        probs, x, a, emb = pol.node_probs(st)
        s = pol.state(st, emb, progress, T, acc_rate, last_gain, last_op)
        op = pol.choose_op(rng, s, progress, graph=(x, a))

        f_start = f_cur
        gain = np.zeros(st.n)
        tried = np.zeros(st.n)
        accepted = 0
        for i in rng.choice(st.n, size=ARLS_BLOCK, p=probs):
            i = int(i)
            info = st.move(i, propose_move(op, st, i, step, rng))
            f_new = st.fitness()
            d = f_new - f_cur
            tried[i] += 1
            if d >= 0.0 or rng.random() < np.exp(d / max(T, 1e-12)):
                f_cur = f_new
                accepted += 1
                gain[i] += max(d, 0.0)
                if f_cur > best_f:
                    best_f, best_pos = f_cur, st.pos.copy()
            else:
                st.undo(info)

        acc_rate = accepted / ARLS_BLOCK
        last_gain = f_cur - f_start
        moved = np.flatnonzero(tried > 0)
        pol.learn_nodes(rng, x, a, moved.astype(np.int64),
                        np.minimum(NGNN_TARGET_SCALE * gain[moved] / tried[moved],
                                   NGNN_TARGET_CLIP))
        s2 = pol.state(st, emb, progress, T, acc_rate, last_gain, op)
        g2 = ((node_features(st), disk_graph(st.pos)) if pol.mode == "gnn_dqn" else None)
        pol.learn_op(rng, s, op, float(np.clip(MDQN_REWARD_SCALE * last_gain, -5, 5)), s2,
                     graph=(x, a), graph2=g2)
        last_op = op

    return quick_evaluate(st.pos), quick_evaluate(best_pos)


def run_proposed(n, seed=SEED):
    """Part-A run, with a fresh learned ARLS policy for this run."""
    global POLICY
    set_seed(seed)
    POLICY = LearnedSearchPolicy(ARLS_MODE) if ARLS_MODE != "original" else None
    r = RUN_PROPOSED_ORIGINAL(n, seed=seed)
    if POLICY is not None:
        sh = POLICY.op_counts / np.maximum(POLICY.op_counts.sum(axis=1, keepdims=True), 1)
        r["node_gnn_loss_start"] = (float(np.mean(POLICY.gnn_losses[:50]))
                                    if POLICY.gnn_losses else np.nan)
        r["node_gnn_loss_end"] = (float(np.mean(POLICY.gnn_losses[-50:]))
                                  if POLICY.gnn_losses else np.nan)
        r["move_dqn_loss_end"] = (float(np.mean(POLICY.q_losses[-100:]))
                                  if POLICY.q_losses else np.nan)
        r["move_dqn_decisions"] = int(POLICY.op_counts.sum())
        r["move_op_share"] = sh
    POLICY = None
    return r




# =====================================================================
# PART F - OPTION 2: FINER OPTIMISATION GRID + OUTSIDE-FIELD PENALTY
#          (one Kaggle commit, about 9-10 h)
# =====================================================================
#
#   The SAME proposed method, ablations and baselines, with two changes
#   to how a layout is scored DURING the optimisation:
#     (1) Optimisation grid 110 x 110 = 12,100 points (2.3 m x 2.75 m),
#         instead of 50 x 50 = 2,500 (5.1 m x 6.1 m). This is set by the
#         one line GRID_RES = 110 at the top of Part A.
#     (2) Sensing area outside the field is penalised exactly like
#         overlap: the outside area (as a fraction of the field area) is
#         subtracted with the overlap weight of each fitness stage
#         (1.93 in f1 = 0.18 + lambda, 14.9 in f2 = 5 x 0.18 + 14).
#   Everything else is unchanged: thresholds (C > 98 %, O < 10 %,
#   EE > 97 %), margins, lambda, population 6, 30 generations, ARLS
#   move budgets, five move types, GNN + Move-DQN settings.
#   Final layouts are REPORTED on a 1 m grid (75,000 cell centres).
#
#   METHODS (50 nodes)
#     PROPOSED     GNN-encoded Move-DQN controls ARLS
#     DQN_ONLY     same Move-DQN without the GNN encoder (effect of the GNN)
#     RANDOM_MOVE  same 5 moves and node rule, move type at random
#     BANDIT_UCB   same, discounted UCB bandit (c = 0.2, discount 0.995)
#     ORIGINAL     hand-crafted ARLS (2 move types)
#     SA_PLAIN     one annealing chain, hand-crafted ARLS, equal time
#     CVT, PSO, GA Lloyd-CVT and two metaheuristics, equal time
#       -> seeds 20-34 (15 paired runs)
#     DE, GWO, WOA, VFA  equal time -> seeds 20-25 (6 runs; these were
#                  far behind in every earlier test)
#   Equal time = mean runtime of the PROPOSED runs finished so far.
#
#   PRE-REGISTERED TEST OF THE DEEP-LEARNING CLAIM (fixed before running)
#     Primary metric: EE on the 1 m grid. Comparisons PROPOSED vs
#     RANDOM_MOVE and PROPOSED vs BANDIT_UCB, paired Wilcoxon (zsplit),
#     Holm over the two. The deep-learning claim is SUPPORTED only if
#     PROPOSED is better than BOTH with p_Holm < 0.05.
#
#   SAFETY: no new run is started after STOP_AFTER_H hours, so the
#   statistics, figures and files are always written before Kaggle's
#   12 h limit. Every finished run is saved at once; re-running resumes.
#
#   OUTPUTS: p3_runs.csv, p3_layout_metrics.csv, p3_results.xlsx,
#            positions_p3.zip, p3_fig_*.png (600 dpi) + .pdf
# =====================================================================

import os, io, shutil, contextlib
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.sparse.csgraph import connected_components, dijkstra

N_NODES = 50
CORE_SEEDS = list(range(20, 35))
WEAK_SEEDS = list(range(20, 26))
CORE_METHODS = ["PROPOSED", "DQN_ONLY", "RANDOM_MOVE", "BANDIT_UCB", "ORIGINAL", "SA_PLAIN", "CVT", "PSO", "GA"]
WEAK_METHODS = ["DE", "GWO", "WOA", "VFA"]
STOP_AFTER_H = 10.3
DEFAULT_BUDGET_S = 210.0           # used only until the first PROPOSED run has finished

UCB_C, UCB_DISCOUNT = 0.2, 0.995
SA_CHUNK = 2000
BASE_POP = 30

RUNS_FILE = "p3_runs.csv"
LAYOUT_FILE = "p3_layout_metrics.csv"
POS_DIR3 = "positions_p3"

E_ELEC, EPS_FS, EPS_MP = 50e-9, 10e-12, 0.0013e-12
D0 = np.sqrt(EPS_FS / EPS_MP)
K_BITS, E_INIT = 4000, 0.5

LABELS = {"PROPOSED": "Proposed (GNN-DQN)", "DQN_ONLY": "DQN only (no GNN)", "RANDOM_MOVE": "Random move type", "BANDIT_UCB": "UCB bandit",
          "ORIGINAL": "Hand-crafted ARLS", "SA_PLAIN": "SA (hand-crafted)", "CVT": "Lloyd-CVT",
          "PSO": "PSO", "GA": "GA", "DE": "DE", "GWO": "GWO", "WOA": "WOA", "VFA": "VFA"}
COLS3 = {"PROPOSED": "#2a78d6", "DQN_ONLY": "#0d4f99", "RANDOM_MOVE": "#8a8984", "BANDIT_UCB": "#1baf7a", "ORIGINAL": "#eb6834",
         "SA_PLAIN": "#eda100", "CVT": "#52514e", "PSO": "#4a3aa7", "GA": "#e87ba4", "DE": "#008300",
         "GWO": "#7a5c2e", "WOA": "#9b2d2d", "VFA": "#e34948"}

PRIMARY = [("PROPOSED", "RANDOM_MOVE"), ("PROPOSED", "BANDIT_UCB")]
SECONDARY = [("PROPOSED", "DQN_ONLY"), ("DQN_ONLY", "RANDOM_MOVE"), ("BANDIT_UCB", "RANDOM_MOVE"), ("RANDOM_MOVE", "ORIGINAL"), ("PROPOSED", "ORIGINAL"),
             ("PROPOSED", "SA_PLAIN"), ("PROPOSED", "CVT"), ("PROPOSED", "PSO"), ("PROPOSED", "GA")]
WEAK_PAIRS = [("PROPOSED", m) for m in WEAK_METHODS]
METRICS = [("EE fine (%)", True), ("Overlap fine (%)", False), ("Coverage fine (%)", True),
           ("Outside sensing (% of field)", False), ("Wasted sensing (% of field)", False),
           ("EE opt-grid (%)", True), ("Lifetime Rc50 (rounds)", True)]


# ---------------------------------------------------------------- outside-field penalty
_u = np.arange(-R_SENSE + 0.125, R_SENSE, 0.25)
_UX, _UY = np.meshgrid(_u, _u)
_DP = np.column_stack([_UX.ravel(), _UY.ravel()])
_DP = _DP[(_DP ** 2).sum(1) <= R_SENSE ** 2]
OUT_STEP = 0.25
_dgrid = np.arange(0.0, R_SENSE + 1e-9, OUT_STEP)
OUT_TABLE = np.zeros((len(_dgrid), len(_dgrid)))
for _i, _dx in enumerate(_dgrid):
    _mx = _DP[:, 0] < -_dx
    OUT_TABLE[_i] = ((_DP[None, :, 1] < -_dgrid[:, None]) | _mx[None, :]).mean(1)
OUT_K = np.pi * R_SENSE ** 2 / (AREA_X * AREA_Y)      # one disk as a fraction of the field
W_OUT1 = 0.18 + OVERLAP_REG_WEIGHT                    # weight of O in f1 (V = 0)
W_OUT2 = 5.0 * 0.18 + 14.0                            # weight of O in f2


def out_frac(p):
    """Fraction of the sensing disk of a sensor at p that lies outside the field."""
    dx = min(float(p[0]), AREA_X - float(p[0]))
    dy = min(float(p[1]), AREA_Y - float(p[1]))
    i = int(round(min(max(dx, 0.0), R_SENSE) / OUT_STEP))
    j = int(round(min(max(dy, 0.0), R_SENSE) / OUT_STEP))
    return float(OUT_TABLE[i, j])


def separation_violations(pos):                 # now: summed outside fraction
    return float(sum(out_frac(p) for p in np.asarray(pos)))


def _node_out(self, i, p):
    return out_frac(p)


def _cs_fitness(self):
    c, o = self.metrics()
    return fitness_raw(c, o) - W_OUT1 * OUT_K * self.sep_viol


CoverageState.node_violations = _node_out
CoverageState.fitness = _cs_fitness


def fitness(sol):
    c, o, ee = sol["coverage"], sol["overlap"], sol["ee"]
    sv = float(sol.get("sep_viol", 0.0))
    if c > TARGET_COVERAGE and ee > TARGET_EE:
        return 5.0 * c + 5.0 * ee - 14.0 * o - W_OUT2 * OUT_K * sv
    return fitness_raw(c, o) - W_OUT1 * OUT_K * sv


def is_feasible(sol):
    return sol["coverage"] > TARGET_COVERAGE and sol["overlap"] < TARGET_OVERLAP and sol["ee"] > TARGET_EE


# ---------------------------------------------------------------- helpers
def setup_output_folder_p3():
    if os.path.isdir("/kaggle/working"):
        os.chdir("/kaggle/working")
    os.makedirs(POS_DIR3, exist_ok=True)
    print(f"Results folder : {os.getcwd()}", flush=True)


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[cols].values.tolist()))


def budget_s():
    if os.path.exists(RUNS_FILE):
        d = pd.read_csv(RUNS_FILE)
        p = d[d.Method == "PROPOSED"]["Runtime (s)"]
        if len(p):
            return float(p.mean())
    return DEFAULT_BUDGET_S


# ---------------------------------------------------------------- simple move selectors
class RuleMovePolicy(LearnedSearchPolicy):
    def __init__(self, rule):
        super().__init__("no_gnn")
        self.rule = rule
        self.n_a = np.zeros(N_MOVE_OPS)
        self.s_a = np.zeros(N_MOVE_OPS)

    def choose_op(self, rng, s, progress, graph=None):
        if self.rule == "random":
            op = int(rng.integers(N_MOVE_OPS))
        elif np.any(self.n_a < 1e-9):
            op = int(np.flatnonzero(self.n_a < 1e-9)[0])
        else:
            mean = self.s_a / self.n_a
            bonus = UCB_C * np.sqrt(2.0 * np.log(max(self.n_a.sum(), 2.0)) / self.n_a)
            op = int(np.argmax(mean + bonus))
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        if self.rule == "ucb":
            self.n_a *= UCB_DISCOUNT
            self.s_a *= UCB_DISCOUNT
            self.n_a[op] += 1.0
            self.s_a[op] += r / 5.0
        self.decisions += 1


def share_from_counts(c):
    c = np.asarray(c, float)
    out = {}
    for h, name in ((0, "first half"), (1, "second half")):
        tot = max(c[h].sum(), 1.0)
        for k, op in enumerate(MOVE_OPS):
            out[f"Share {op} ({name}) (%)"] = 100.0 * c[h, k] / tot
    return out


# ---------------------------------------------------------------- runners
def run_framework(seed, mode, policy=None):
    global ARLS_MODE, POLICY
    t0 = time.time()
    ARLS_MODE = mode
    extra = {}
    if policy is None:
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_proposed(N_NODES, seed=seed)
        if r.get("move_op_share") is not None:
            sh = r["move_op_share"]
            for h, name in ((0, "first half"), (1, "second half")):
                for k, op in enumerate(MOVE_OPS):
                    extra[f"Share {op} ({name}) (%)"] = 100.0 * sh[h, k]
    else:
        set_seed(seed)
        POLICY = policy
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = RUN_PROPOSED_ORIGINAL(N_NODES, seed=seed)
        finally:
            POLICY = None
        extra = share_from_counts(policy.op_counts)
    ARLS_MODE = "gnn_dqn"
    return np.asarray(r["positions"], float), time.time() - t0, np.nan, extra


def run_sa(seed, budget):
    global ARLS_MODE, POLICY
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ARLS_MODE, POLICY = "original", None
    t0 = time.time()
    pos = make_seed(N_NODES, 0.0, 0.0, SEED)
    best, moves, last_dt = None, 0, 0.0
    while time.time() - t0 < budget:
        el = time.time() - t0
        f0, f1 = el / budget, min(1.0, (el + max(last_dt, 0.05)) / budget)
        Ta = ARLS_T_START * (ARLS_T_END / ARLS_T_START) ** f0
        Tb = ARLS_T_START * (ARLS_T_END / ARLS_T_START) ** f1
        sa = ARLS_STEP_MAX + (ARLS_STEP_MIN - ARLS_STEP_MAX) * f0
        sb = ARLS_STEP_MAX + (ARLS_STEP_MIN - ARLS_STEP_MAX) * f1
        ts = time.time()
        cur, b = ARLS(pos, SA_CHUNK, Ta, Tb, sa, sb, rng)
        last_dt = time.time() - ts
        moves += SA_CHUNK
        pos = cur["pos"]
        if best is None or rank_key(b) > rank_key(best):
            best = b
    ARLS_MODE = "gnn_dqn"
    return best["pos"].copy(), time.time() - t0, moves, {}


class _Track:
    def __init__(self, budget):
        self.budget, self.t0 = budget, time.time()
        self.best_k, self.best_x, self.evals = None, None, 0

    def frac(self):
        return (time.time() - self.t0) / self.budget

    def alive(self):
        return self.frac() < 1.0

    def update(self, x):
        sol = quick_evaluate(x.reshape(N_NODES, 2))
        k = rank_key(sol)
        self.evals += 1
        if self.best_k is None or k > self.best_k:
            self.best_k, self.best_x = k, x.copy()
        return fitness(sol)


def _bounds():
    return np.zeros(2 * N_NODES), np.tile([AREA_X, AREA_Y], N_NODES)


def run_pso(T, rng):
    lo, hi = _bounds(); D = 2 * N_NODES
    x = rng.uniform(lo, hi, (BASE_POP, D)); v = np.zeros_like(x); vmax = 0.2 * (hi - lo)
    pf = np.array([T.update(xi) for xi in x]); pb, gb = x.copy(), x[np.argmax(pf)].copy()
    while T.alive():
        r1, r2 = rng.random((BASE_POP, D)), rng.random((BASE_POP, D))
        v = np.clip(0.729 * v + 1.49445 * r1 * (pb - x) + 1.49445 * r2 * (gb - x), -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        for i in range(BASE_POP):
            f = T.update(x[i])
            if f > pf[i]:
                pf[i], pb[i] = f, x[i].copy()
        gb = pb[np.argmax(pf)].copy()


def run_ga(T, rng):
    lo, hi = _bounds(); D = 2 * N_NODES
    x = rng.uniform(lo, hi, (BASE_POP, D)); f = np.array([T.update(xi) for xi in x]); sigma = 0.05 * (hi - lo)
    while T.alive():
        el = np.argsort(-f)[:2]; new = [x[el[0]].copy(), x[el[1]].copy()]
        while len(new) < BASE_POP:
            p = [x[max(rng.choice(BASE_POP, 3, replace=False), key=lambda k: f[k])] for _ in range(2)]
            child = p[0] + rng.uniform(-0.5, 1.5, D) * (p[1] - p[0]) if rng.random() < 0.9 else p[0].copy()
            mut = rng.random(D) < 4.0 / D
            child[mut] += rng.normal(0, sigma[mut])
            new.append(np.clip(child, lo, hi))
        x = np.array(new); f = np.array([T.update(xi) for xi in x])


def run_de(T, rng):
    lo, hi = _bounds(); D = 2 * N_NODES
    x = rng.uniform(lo, hi, (BASE_POP, D)); f = np.array([T.update(xi) for xi in x])
    while T.alive():
        for i in range(BASE_POP):
            a, b, c = x[rng.choice([k for k in range(BASE_POP) if k != i], 3, replace=False)]
            cross = rng.random(D) < 0.9; cross[rng.integers(D)] = True
            trial = np.where(cross, np.clip(a + 0.5 * (b - c), lo, hi), x[i])
            ft = T.update(trial)
            if ft >= f[i]:
                x[i], f[i] = trial, ft
            if not T.alive():
                break


def run_gwo(T, rng):
    lo, hi = _bounds(); D = 2 * N_NODES
    x = rng.uniform(lo, hi, (BASE_POP, D)); f = np.array([T.update(xi) for xi in x])
    while T.alive():
        leaders = [x[k].copy() for k in np.argsort(-f)[:3]]
        a = 2.0 * (1.0 - min(T.frac(), 1.0))
        for i in range(BASE_POP):
            xs = []
            for L in leaders:
                A = 2 * a * rng.random(D) - a; C = 2 * rng.random(D)
                xs.append(L - A * np.abs(C * L - x[i]))
            x[i] = np.clip(np.mean(xs, axis=0), lo, hi); f[i] = T.update(x[i])


def run_woa(T, rng):
    lo, hi = _bounds()
    x = rng.uniform(lo, hi, (BASE_POP, 2 * N_NODES)); f = np.array([T.update(xi) for xi in x])
    while T.alive():
        best = x[np.argmax(f)].copy(); a = 2.0 * (1.0 - min(T.frac(), 1.0))
        for i in range(BASE_POP):
            A = 2 * a * rng.random() - a; C = 2 * rng.random()
            if rng.random() < 0.5:
                ref = best if abs(A) < 1 else x[rng.integers(BASE_POP)]
                x[i] = ref - A * np.abs(C * ref - x[i])
            else:
                l = rng.uniform(-1, 1)
                x[i] = np.abs(best - x[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
            x[i] = np.clip(x[i], lo, hi); f[i] = T.update(x[i])


def run_vfa(T, rng):
    pos = rng.uniform([0, 0], [AREA_X, AREA_Y], (N_NODES, 2))
    dth, step_max, m = np.sqrt(3.0) * R_SENSE, 0.2 * R_SENSE, R_SENSE / 2.0
    while T.alive():
        T.update(pos.ravel())
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + np.eye(N_NODES) * 1e9
        u = d / dist[..., None]
        F = ((((dth - dist)[..., None] * u) * (dist < dth)[..., None]).sum(1)
             - 0.5 * (((dist - dth)[..., None] * u) * ((dist > dth) & (dist < 2 * dth))[..., None]).sum(1))
        F[:, 0] += np.clip(m - pos[:, 0], 0, None) - np.clip(pos[:, 0] - (AREA_X - m), 0, None)
        F[:, 1] += np.clip(m - pos[:, 1], 0, None) - np.clip(pos[:, 1] - (AREA_Y - m), 0, None)
        nrm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
        pos = np.clip(pos + np.minimum(nrm, step_max) * F / nrm + rng.normal(0, 0.3, (N_NODES, 2)),
                      [0, 0], [AREA_X, AREA_Y])


def run_cvt(T, rng):
    pos = rng.uniform([0, 0], [AREA_X, AREA_Y], (N_NODES, 2))
    while T.alive():
        T.update(pos.ravel())
        a = ((GRID[:, None, :] - pos[None, :, :]) ** 2).sum(2).argmin(1)
        new = np.array([GRID[a == i].mean(0) if np.any(a == i) else GRID[rng.integers(NUM_GRID)]
                        for i in range(N_NODES)])
        if np.abs(new - pos).max() < 1e-2:
            new = np.clip(new + rng.normal(0, 0.5 * R_SENSE, (N_NODES, 2)), [0, 0], [AREA_X, AREA_Y])
        pos = new


TIMED = {"CVT": run_cvt, "PSO": run_pso, "GA": run_ga, "DE": run_de, "GWO": run_gwo, "WOA": run_woa, "VFA": run_vfa}


def run_method(key, seed):
    if key == "PROPOSED":
        return run_framework(seed, "gnn_dqn")
    if key == "DQN_ONLY":
        return run_framework(seed, "no_gnn")
    if key == "ORIGINAL":
        return run_framework(seed, "original")
    if key == "RANDOM_MOVE":
        return run_framework(seed, "no_gnn", RuleMovePolicy("random"))
    if key == "BANDIT_UCB":
        return run_framework(seed, "no_gnn", RuleMovePolicy("ucb"))
    B = budget_s()
    if key == "SA_PLAIN":
        return run_sa(seed, B)
    T = _Track(B)
    TIMED[key](T, np.random.default_rng(seed))
    return T.best_x.reshape(N_NODES, 2), time.time() - T.t0, T.evals, {}


# ---------------------------------------------------------------- layout metrics
FINE = np.column_stack([g.ravel() for g in np.meshgrid(np.arange(0.5, AREA_X, 1.0), np.arange(0.5, AREA_Y, 1.0))])


def fine_metrics(pos):
    cnt = np.zeros(len(FINE), np.int16)
    for p in pos:
        cnt += (((FINE[:, 0] - p[0]) ** 2 + (FINE[:, 1] - p[1]) ** 2) <= R_SENSE ** 2)
    c, o = float(np.mean(cnt >= 1)), float(np.mean(cnt >= 2))
    e = energy_efficiency(c, o)
    return 100 * c, 100 * o, 100 * e, bool(c > TARGET_COVERAGE and o < TARGET_OVERLAP and e > TARGET_EE)


def sink_pos():
    return np.array([AREA_X / 2.0, AREA_Y / 2.0])


def lifetime(pos, rc):
    n = len(pos)
    P = np.vstack([pos, sink_pos()])
    d = pairwise_distances(P)
    amp = np.where(d < D0, EPS_FS * d ** 2, EPS_MP * d ** 4)
    cost = np.where((d <= rc) & ~np.eye(n + 1, dtype=bool), 2 * E_ELEC * K_BITS + amp * K_BITS + 1e-15, 0.0)
    dist, pred = dijkstra(cost, directed=False, indices=n, return_predecessors=True)
    if np.any(~np.isfinite(dist[:n])):
        return np.nan
    traffic = np.ones(n)
    for i in np.argsort(-dist[:n]):
        p = pred[i]
        if 0 <= p < n:
            traffic[p] += traffic[i]
    dl = d[np.arange(n), pred[:n]]
    e_tx = E_ELEC * K_BITS + np.where(dl < D0, EPS_FS * dl ** 2, EPS_MP * dl ** 4) * K_BITS
    return float(int(E_INIT / (traffic * e_tx + (traffic - 1) * E_ELEC * K_BITS).max()))


def reach(pos, rc):
    P = np.vstack([pos, sink_pos()])
    _, lab = connected_components(pairwise_distances(P) <= rc, directed=False)
    return 100.0 * float(np.mean(lab[:-1] == lab[-1]))


def layout_row(key, seed, pos):
    c, o, _, _ = coverage_data(pos)
    fc, fo, fe, ff = fine_metrics(pos)
    fr = np.array([out_frac(p) for p in pos])
    out_field = 100.0 * fr.sum() * OUT_K                     # outside area, % of field area
    edge = np.minimum.reduce([pos[:, 0], AREA_X - pos[:, 0], pos[:, 1], AREA_Y - pos[:, 1]])
    return {"Method": key, "Seed": seed,
            "Coverage opt-grid (%)": 100 * c, "Overlap opt-grid (%)": 100 * o,
            "EE opt-grid (%)": 100 * energy_efficiency(c, o),
            "Feasible opt-grid": bool(c > TARGET_COVERAGE and o < TARGET_OVERLAP and energy_efficiency(c, o) > TARGET_EE),
            "Coverage fine (%)": fc, "Overlap fine (%)": fo, "EE fine (%)": fe, "Feasible fine": ff,
            "Outside sensing (% of sensing area)": 100.0 * fr.mean(),
            "Outside sensing (% of field)": out_field, "Wasted sensing (% of field)": fo + out_field,
            "Sensors on boundary (<=1 m)": int(np.sum(edge <= 1.0)), "Sensors with disk outside": int(np.sum(fr > 0)),
            "Reach sink Rc30 (%)": reach(pos, 30.0), "Reach sink Rc50 (%)": reach(pos, 2 * R_SENSE),
            "Lifetime Rc50 (rounds)": lifetime(pos, 2 * R_SENSE)}


# ---------------------------------------------------------------- experiment loop
def jobs_todo():
    done = done_keys(RUNS_FILE, ["Method", "Seed"])
    jobs = []
    for s in CORE_SEEDS:
        ms = CORE_METHODS + (WEAK_METHODS if s in WEAK_SEEDS else [])
        jobs += [(k, s) for k in ms if (k, s) not in done]
    return jobs


def run_all(t_start):
    jobs = jobs_todo()
    print(f"\nRUNS: {len(jobs)} to do", flush=True)
    t0 = time.time()
    for j, (key, seed) in enumerate(jobs, 1):
        if (time.time() - t_start) / 3600.0 > STOP_AFTER_H:
            print(f"\nSTOP: {STOP_AFTER_H} h reached; {len(jobs) - j + 1} runs left for a later commit.", flush=True)
            break
        ts = time.time()
        pos, rt, evals, extra = run_method(key, seed)
        np.save(os.path.join(POS_DIR3, f"{key}_N{N_NODES}_seed{seed}.npy"), pos)
        row = layout_row(key, seed, pos)
        append_csv(LAYOUT_FILE, row)
        append_csv(RUNS_FILE, {"Method": key, "Seed": seed, "Runtime (s)": rt, "Evaluations/moves": evals,
                               "Budget (s)": budget_s(), **extra})
        eta = (time.time() - t0) / j * (len(jobs) - j)
        print(f"[{j:3d}/{len(jobs)}] {key:11s} seed={seed:2d} fine C={row['Coverage fine (%)']:.2f} "
              f"O={row['Overlap fine (%)']:.2f} EE={row['EE fine (%)']:.3f} out={row['Outside sensing (% of field)']:.2f} "
              f"| opt C={row['Coverage opt-grid (%)']:.2f} O={row['Overlap opt-grid (%)']:.2f} "
              f"{'PASS' if row['Feasible opt-grid'] else 'fail'} ({time.time()-ts:.0f}s, ETA {eta/60:.0f} min)", flush=True)


# ---------------------------------------------------------------- statistics
def a12(x, y, higher=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not higher:
        x, y = -x, -y
    return ((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def a12_label(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def holm(p):
    p = np.asarray(p, float)
    out, run = np.empty_like(p), 0.0
    for i, k in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        out[k] = run
    return out


def paired_tests(L, pairs, metrics, family):
    rows = []
    for col, higher in metrics:
        block = []
        for a, b in pairs:
            A = L[L.Method == a].set_index("Seed")[col]
            B = L[L.Method == b].set_index("Seed")[col]
            seeds = A.dropna().index.intersection(B.dropna().index)
            if len(seeds) < 2:
                continue
            x, y = A.loc[seeds].values.astype(float), B.loc[seeds].values.astype(float)
            diff = x - y
            bt = int(np.sum(diff > 1e-9)) if higher else int(np.sum(diff < -1e-9))
            wr = int(np.sum(diff < -1e-9)) if higher else int(np.sum(diff > 1e-9))
            p = wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(np.abs(diff) > 1e-12) else 1.0
            eff = a12(x, y, higher)
            block.append({"Family": family, "Metric": col, "Comparison": f"{a} vs {b}", "Seeds": len(seeds),
                          "Mean first": x.mean(), "Mean second": y.mean(), "Mean difference": diff.mean(),
                          "First better / tie / worse": f"{bt} / {len(diff)-bt-wr} / {wr}",
                          "p (Wilcoxon)": p, "A12": eff, "Effect": a12_label(eff)})
        if block:
            for r, h in zip(block, holm([r["p (Wilcoxon)"] for r in block])):
                r["p (Holm)"] = h
                r["Significant (0.05)"] = bool(h < 0.05)
            rows += block
    return pd.DataFrame(rows)


def describe_table(L, methods):
    rows = []
    for m in methods:
        d = L[L.Method == m]
        if not len(d):
            continue
        r = {"Method": m, "Label": LABELS[m], "Runs": len(d)}
        for col, higher in METRICS + [("Coverage opt-grid (%)", True), ("Overlap opt-grid (%)", False),
                                      ("Outside sensing (% of sensing area)", False)]:
            v = d[col].dropna().values.astype(float)
            if len(v):
                r[f"{col} mean"] = v.mean()
                r[f"{col} std"] = v.std(ddof=1) if len(v) > 1 else 0.0
                r[f"{col} best"] = v.max() if higher else v.min()
        r["Pass opt-grid (%)"] = 100 * d["Feasible opt-grid"].mean()
        r["Pass fine (%)"] = 100 * d["Feasible fine"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def report_p3():
    if not os.path.exists(LAYOUT_FILE):
        print("No results yet.")
        return
    L = pd.read_csv(LAYOUT_FILE).drop_duplicates(["Method", "Seed"], keep="last")
    R = pd.read_csv(RUNS_FILE).drop_duplicates(["Method", "Seed"], keep="last")
    present = [m for m in CORE_METHODS + WEAK_METHODS if m in set(L.Method)]
    desc = describe_table(L, present)
    prim = paired_tests(L, PRIMARY, METRICS, "primary (DL claim)")
    sec = paired_tests(L, SECONDARY, METRICS, "secondary")
    weak = paired_tests(L, WEAK_PAIRS, METRICS[:5], "weak baselines (6 seeds)")
    tests = pd.concat([prim, sec, weak], ignore_index=True)
    fr = []
    core = [m for m in CORE_METHODS if m in present]
    for col, higher in METRICS[:6]:
        piv = L[L.Method.isin(core)].pivot(index="Seed", columns="Method", values=col).dropna()
        if piv.shape[1] >= 3 and len(piv) >= 3:
            st, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
            rk = (-piv if higher else piv).rank(axis=1).mean()
            fr.append({"Metric": col, "Seeds": len(piv), "Chi2": st, "p": p, **{f"Rank {c}": rk[c] for c in piv.columns}})
    fr = pd.DataFrame(fr)
    ops = R.groupby("Method")[[c for c in R.columns if c.startswith("Share ")]].mean() if any(
        c.startswith("Share ") for c in R.columns) else pd.DataFrame()
    runtime = R.groupby("Method")[["Runtime (s)", "Evaluations/moves"]].mean()
    with pd.ExcelWriter("p3_results.xlsx") as xw:
        desc.to_excel(xw, sheet_name="descriptive", index=False)
        tests.to_excel(xw, sheet_name="wilcoxon_holm", index=False)
        fr.to_excel(xw, sheet_name="friedman_core", index=False)
        if len(ops):
            ops.to_excel(xw, sheet_name="move_type_usage")
        runtime.to_excel(xw, sheet_name="runtime")
        L.to_excel(xw, sheet_name="all_layouts", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    ff = lambda v: f"{v:.4f}"
    print("\n" + "=" * 120 + "\nOPTION 2 RESULTS (optimised on 12,100-point grid with outside penalty; reported on 1 m grid)\n" + "=" * 120)
    show = ["Method", "Runs"] + [f"{c} mean" for c, _ in METRICS] + ["Coverage opt-grid (%) mean", "Pass opt-grid (%)", "Pass fine (%)"]
    print(desc[[c for c in show if c in desc.columns]].to_string(index=False, float_format=ff))
    keep = ["Family", "Metric", "Comparison", "Seeds", "Mean first", "Mean second", "Mean difference",
            "First better / tie / worse", "p (Wilcoxon)", "p (Holm)", "A12", "Effect", "Significant (0.05)"]
    print("\nPaired Wilcoxon (zsplit), Holm within each family and metric")
    print(tests[[c for c in keep if c in tests.columns]].to_string(index=False, float_format=ff))
    print("\nFriedman over the core methods (lower rank = better)")
    print(fr.to_string(index=False, float_format=ff))
    if len(ops):
        print("\nMove-type usage (% of decisions)")
        print(ops.T.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\nMean runtime and evaluations/moves")
    print(runtime.to_string(float_format=lambda v: f"{v:.1f}"))
    pe = prim[prim.Metric == "EE fine (%)"]
    if len(pe) == 2:
        ok = bool(all((pe["Mean difference"] > 0) & (pe["p (Holm)"] < 0.05)))
        print("\nPRE-REGISTERED DECISION: deep-learning claim " + ("SUPPORTED" if ok else "NOT SUPPORTED") +
              " (" + "; ".join(f"{r.Comparison}: diff {r['Mean difference']:+.3f}, p_Holm {r['p (Holm)']:.4f}"
                               for _, r in pe.iterrows()) + ")")


# ---------------------------------------------------------------- figures (600 dpi)
def make_figures_p3():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                         "axes.axisbelow": True, "savefig.dpi": 600, "pdf.fonttype": 42,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    L = pd.read_csv(LAYOUT_FILE).drop_duplicates(["Method", "Seed"], keep="last")
    rng = np.random.default_rng(1)

    def boxes(ax, keys, col):
        data = [L[L.Method == k][col].dropna().values for k in keys]
        bp = ax.boxplot(data, positions=range(len(keys)), widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#0b0b0b", lw=1.0))
        for patch, k in zip(bp["boxes"], keys):
            patch.set_facecolor(COLS3[k]); patch.set_alpha(0.5); patch.set_edgecolor(COLS3[k])
        for i, (k, v) in enumerate(zip(keys, data)):
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=5, color=COLS3[k], edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([LABELS[k] for k in keys], rotation=40, ha="right")
        ax.set_ylabel(col); ax.grid(axis="x", visible=False)

    def save(fig, name):
        fig.savefig(name + ".png", bbox_inches="tight"); fig.savefig(name + ".pdf", bbox_inches="tight"); plt.close(fig)

    abl = [k for k in ["PROPOSED", "DQN_ONLY", "RANDOM_MOVE", "BANDIT_UCB", "ORIGINAL", "SA_PLAIN"] if k in set(L.Method)]
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.8), constrained_layout=True)
    for ax, col, pl in zip(axs, ["EE fine (%)", "Overlap fine (%)", "Coverage fine (%)", "Outside sensing (% of field)"], "abcd"):
        boxes(ax, abl, col); ax.set_title(f"({pl}) {col.split(' (')[0]}")
    save(fig, "p3_fig_ablation")

    allm = [k for k in CORE_METHODS + WEAK_METHODS if k in set(L.Method)]
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    for ax, col, pl in zip(axs, ["EE fine (%)", "Wasted sensing (% of field)", "Coverage fine (%)"], "abc"):
        boxes(ax, allm, col); ax.set_title(f"({pl}) {col.split(' (')[0]}")
    save(fig, "p3_fig_all_methods")

    fig, ax = plt.subplots(figsize=(4.8, 2.8), constrained_layout=True)
    boxes(ax, [k for k in CORE_METHODS if k in set(L.Method)], "Lifetime Rc50 (rounds)")
    ax.set_title("Lifetime at Rc = 2Rs = 50 m (first node dies)")
    save(fig, "p3_fig_lifetime")

    show = [k for k in ["PROPOSED", "RANDOM_MOVE", "ORIGINAL", "CVT"]
            if os.path.exists(os.path.join(POS_DIR3, f"{k}_N{N_NODES}_seed{CORE_SEEDS[0]}.npy"))]
    if show:
        fig, axs = plt.subplots(1, len(show), figsize=(1.95 * len(show), 2.7), constrained_layout=True)
        for ax, k in zip(np.atleast_1d(axs), show):
            pos = np.load(os.path.join(POS_DIR3, f"{k}_N{N_NODES}_seed{CORE_SEEDS[0]}.npy"))
            r = L[(L.Method == k) & (L.Seed == CORE_SEEDS[0])].iloc[0]
            for p in pos:
                ax.add_patch(Circle(p, R_SENSE, color=COLS3[k], alpha=0.18, lw=0))
            ax.scatter(pos[:, 0], pos[:, 1], s=3, color="#0b0b0b", zorder=3)
            ax.add_patch(Rectangle((0, 0), AREA_X, AREA_Y, fill=False, lw=0.8, color="#0b0b0b"))
            ax.set_xlim(-30, AREA_X + 30); ax.set_ylim(-30, AREA_Y + 30); ax.set_aspect("equal"); ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(LABELS[k], fontsize=7)
            ax.set_xlabel(f"C {r['Coverage fine (%)']:.2f}  O {r['Overlap fine (%)']:.2f}\n"
                          f"outside {r['Outside sensing (% of sensing area)']:.1f} % of sensing", fontsize=6)
        save(fig, f"p3_fig_layouts_seed{CORE_SEEDS[0]}")

    P = L[L.Method == "PROPOSED"]
    if len(P):
        b = P.sort_values("EE fine (%)").iloc[-1]
        pos = np.load(os.path.join(POS_DIR3, f"PROPOSED_N{N_NODES}_seed{int(b.Seed)}.npy"))
        fig, ax = plt.subplots(figsize=(3.2, 3.6), constrained_layout=True)
        for p in pos:
            ax.add_patch(Circle(p, R_SENSE, color=COLS3["PROPOSED"], alpha=0.18, lw=0))
        ax.scatter(pos[:, 0], pos[:, 1], s=5, color="#0b0b0b", zorder=3)
        ax.add_patch(Rectangle((0, 0), AREA_X, AREA_Y, fill=False, lw=0.8, color="#0b0b0b"))
        ax.set_xlim(-30, AREA_X + 30); ax.set_ylim(-30, AREA_Y + 30); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"Best proposed layout (seed {int(b.Seed)}, 1 m grid)\nC {b['Coverage fine (%)']:.2f} %, "
                     f"O {b['Overlap fine (%)']:.2f} %, EE {b['EE fine (%)']:.2f} %", fontsize=7)
        save(fig, "p3_fig_best_layout")
    shutil.make_archive("positions_p3", "zip", POS_DIR3)
    print("Saved: p3_results.xlsx, p3_fig_*.png (600 dpi) + .pdf, positions_p3.zip", flush=True)


def run_part_f():
    t_start = time.time()
    setup_output_folder_p3()
    print("=" * 120 + "\nPART F (OPTION 2): optimisation grid "
          f"{GRID_RES} x {GRID_RES} = {NUM_GRID} points, outside-field penalty, reporting on 1 m grid\n" + "=" * 120, flush=True)
    print(f"Core seeds {CORE_SEEDS[0]}-{CORE_SEEDS[-1]} ({', '.join(CORE_METHODS)}); "
          f"weak baselines seeds {WEAK_SEEDS[0]}-{WEAK_SEEDS[-1]} ({', '.join(WEAK_METHODS)}); "
          f"stop starting new runs after {STOP_AFTER_H} h", flush=True)
    run_all(t_start)
    report_p3()
    make_figures_p3()
    print(f"\nTOTAL TIME: {(time.time() - t_start)/3600:.2f} h", flush=True)




# =====================================================================
# PART G - EXTENDED COMPARISON (one Kaggle commit, about 10 h)
# =====================================================================
#
#   Same scoring as Study B: optimisation grid 110 x 110, outside-field
#   (or obstacle) sensing penalised like overlap, results on a 1 m grid.
#   Seeds 20-29 in both fields. Jobs run seed by seed (rectangle, then
#   irregular), so a stop leaves whole seeds missing, never whole methods.
#
#   NEW LEARNING-BASED COMPARISONS (same EnMOGKSO, only the move selector
#   inside the ARLS is changed; same node choice, moves, budget):
#     CNN_DQN  coverage map as a 3-channel image (hole / single / overlap,
#              22 x 22) -> 2-layer CNN -> 16-d embedding -> same DQN head.
#              Answers: why a graph network and not a CNN?
#     QLEARN   tabular Q-learning, no neural network (216 discrete states:
#              progress x acceptance x last gain x last move type).
#              Answers: why deep RL and not classical RL?
#
#   NEW METAHEURISTIC BASELINES (equal time budget)
#     GKSO     original Genghis Khan Shark Optimizer (Hu et al. 2023,
#              Adv. Eng. Inform. 58:102210): hunting, movement, foraging,
#              self-protection, written from the published equations.
#              Constants not given in the equations are marked GKSO_*.
#     WOA_LFGA Xu, Zhang & Zhang 2023, Biomimetics 8:354.
#
#   BLOCK R  RECTANGULAR FIELD 250 x 300 m, 50 nodes
#     PROPOSED, CNN_DQN, QLEARN, GKSO, WOA_LFGA
#   BLOCK I  IRREGULAR FIELD WITH OBSTACLES, 44 nodes
#     Bounding box 250 x 300 m minus a missing corner [175,250]x[210,300]
#     and obstacles [60,110]x[110,150], [150,200]x[40,80]; usable area
#     64,250 m^2; no sensor inside blocked areas; sensing over blocked or
#     outside area penalised like overlap. 44 nodes keep the disk-area to
#     usable-area ratio (1.34) close to the rectangle (1.31).
#     PROPOSED, DQN_ONLY, CNN_DQN, QLEARN, RANDOM_MOVE, BANDIT_UCB,
#     ORIGINAL, SA_PLAIN, CVT, PSO, GA, GKSO, WOA_LFGA
#
#   PRE-SPECIFIED TESTS (fixed before running, EE on the 1 m grid, Holm)
#     T1 (irregular): PROPOSED vs RANDOM_MOVE, PROPOSED vs BANDIT_UCB
#     T2 (each field): PROPOSED vs CNN_DQN, PROPOSED vs QLEARN
#     A claim is SUPPORTED only if PROPOSED is better in both comparisons
#     of the family with p_Holm < 0.05. Everything else is secondary.
#
#   Equal time for time-budgeted methods = mean runtime of the PROPOSED
#   runs already finished in the same field (same machine).
#   SAFETY: no new run starts after STOP_AFTER_H hours (worst case end
#   about 10.9 h, under the 12 h Kaggle limit); each run is saved at once;
#   re-running the cell resumes.
#
#   OUTPUTS: p4_runs.csv, p4_layout_metrics.csv, p4_results.xlsx,
#            positions_p4.zip, p4_fig_*.png (600 dpi)
# =====================================================================

from scipy.special import gamma as _gamma

RUN_BLOCKS = {"R": True, "I": True}
SEEDS4 = list(range(20, 30))
R_SEEDS = SEEDS4
I_SEEDS = SEEDS4
R_METHODS = ["PROPOSED", "CNN_DQN", "QLEARN", "GKSO", "WOA_LFGA"]
I_METHODS = ["PROPOSED", "DQN_ONLY", "CNN_DQN", "QLEARN", "RANDOM_MOVE", "BANDIT_UCB", "ORIGINAL",
             "SA_PLAIN", "CVT", "PSO", "GA", "GKSO", "WOA_LFGA"]
N_RECT, N_IRREG = 50, 44
STOP_AFTER_H = 10.6
RUNS_FILE = "p4_runs.csv"
LAYOUT_FILE = "p4_layout_metrics.csv"
POS_DIR4 = "positions_p4"

LABELS.update({"GKSO": "GKSO (2023)", "WOA_LFGA": "WOA-LFGA (2023)", "CNN_DQN": "CNN-DQN", "QLEARN": "Q-learning"})
COLS3.update({"GKSO": "#7a3e9d", "WOA_LFGA": "#b8860b", "CNN_DQN": "#c0392b", "QLEARN": "#16a085"})
T2 = [("PROPOSED", "CNN_DQN"), ("PROPOSED", "QLEARN")]

BLOCKED = [(175.0, 210.0, 250.0, 300.0), (60.0, 110.0, 110.0, 150.0), (150.0, 40.0, 200.0, 80.0)]
VALID_AREA = AREA_X * AREA_Y - sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in BLOCKED)


def valid_xy(X, Y):
    ok = (X >= 0) & (X <= AREA_X) & (Y >= 0) & (Y <= AREA_Y)
    for x0, y0, x1, y1 in BLOCKED:
        ok &= ~((X > x0) & (X < x1) & (Y > y0) & (Y < y1))
    return ok


# ---------------------------------------------------------------- field switching
_FULL = {"GRID": GRID, "NUM_GRID": NUM_GRID, "FINE": FINE, "OUT_K": OUT_K,
         "disk_indices": disk_indices, "clip_positions": clip_positions,
         "clip_point": clip_point, "out_frac": out_frac}
_VALID = valid_xy(GRID[:, 0], GRID[:, 1])
_MAP = -np.ones(len(GRID), dtype=np.int64)
_MAP[_VALID] = np.arange(int(_VALID.sum()))
_FINE_I = FINE[valid_xy(FINE[:, 0], FINE[:, 1])]


def _project(x, y):
    x = min(max(x, 0.0), AREA_X); y = min(max(y, 0.0), AREA_Y)
    for _ in range(2):
        for x0, y0, x1, y1 in BLOCKED:
            if x0 < x < x1 and y0 < y < y1:
                cand = [(x0, y), (x1, y), (x, y0), (x, y1)]
                cand = [c for c in cand if 0.0 <= c[0] <= AREA_X and 0.0 <= c[1] <= AREA_Y]
                x, y = min(cand, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
    return x, y


def _clip_point_I(p):
    return np.array(_project(float(p[0]), float(p[1])), dtype=np.float64)


def _clip_positions_I(pos):
    pos = np.asarray(pos, dtype=np.float64)
    return np.array([_project(float(a), float(b)) for a, b in pos], dtype=np.float64).reshape(-1, 2)


def _disk_indices_I(p):
    m = _MAP[_FULL["disk_indices"](p)]
    return m[m >= 0]


# outside / blocked share of a sensing disk, raster lookup (2.5 m)
_d = np.arange(-R_SENSE + 0.25, R_SENSE, 0.5)
_DX, _DY = np.meshgrid(_d, _d)
_DPI = np.column_stack([_DX.ravel(), _DY.ravel()])
_DPI = _DPI[(_DPI ** 2).sum(1) <= R_SENSE ** 2]
_RX = np.arange(0.0, AREA_X + 1e-9, 2.5)
_RY = np.arange(0.0, AREA_Y + 1e-9, 2.5)
_OUT_I = np.zeros((len(_RX), len(_RY)))
for _i, _x in enumerate(_RX):
    PX = _x + _DPI[None, :, 0]
    PY = _RY[:, None] + _DPI[None, :, 1]
    _OUT_I[_i] = 1.0 - valid_xy(np.broadcast_to(PX, PY.shape), PY).mean(1)


def _out_frac_I(p):
    i = int(round(min(max(float(p[0]), 0.0), AREA_X) / 2.5))
    j = int(round(min(max(float(p[1]), 0.0), AREA_Y) / 2.5))
    return float(_OUT_I[i, j])


CURRENT_FIELD = "R"


def set_field(name):
    global GRID, NUM_GRID, FINE, OUT_K, disk_indices, clip_positions, clip_point, out_frac, N_NODES, CURRENT_FIELD
    CURRENT_FIELD = name
    if name == "R":
        GRID, NUM_GRID, FINE, OUT_K = _FULL["GRID"], _FULL["NUM_GRID"], _FULL["FINE"], _FULL["OUT_K"]
        disk_indices, clip_positions = _FULL["disk_indices"], _FULL["clip_positions"]
        clip_point, out_frac = _FULL["clip_point"], _FULL["out_frac"]
        N_NODES = N_RECT
    else:
        GRID = _FULL["GRID"][_VALID]; NUM_GRID = len(GRID); FINE = _FINE_I
        OUT_K = np.pi * R_SENSE ** 2 / VALID_AREA
        disk_indices, clip_positions, clip_point, out_frac = _disk_indices_I, _clip_positions_I, _clip_point_I, _out_frac_I
        N_NODES = N_IRREG


# ---------------------------------------------------------------- new baselines
def _levy(rng, d, beta=1.5):
    sig = (_gamma(1 + beta) * np.sin(np.pi * beta / 2) / (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, sig, d) / np.abs(rng.normal(0, 1, d)) ** (1 / beta)


# Constants of the original GKSO that are not fixed by the equations as
# reproduced in the literature. CHECK THESE AGAINST THE ORIGINAL PAPER
# (Hu et al. 2023) BEFORE PUBLISHING; change here if they differ.
GKSO_M, GKSO_I = 1.5, 1.5          # movement step s = m * I^r
GKSO_BETA_MIN, GKSO_BETA_MAX = 0.2, 1.2
GKSO_PROTECT_P = 0.5               # chance that self-protection is applied


def run_gkso(T, rng):
    """Original GKSO (Hu et al. 2023), four behaviours per iteration, each
    followed by greedy selection:
      1 hunting      X = X + (lb + r1 (ub - lb)) / (i t)
      2 movement     Xh = s (Xbest - X), s = m I^r;  X = (Xh + X_{i-1}) / 2
      3 foraging     X = Xbest + r2 (Xbest - X) + lambda p^2 (Xbest - X),
                     p = 2{1 - (t/T)^(1/4) + |w| [(t/T)^(1/4) - (t/T)^3]},
                     |w(t+1)| = 1 - 2 w(t)^4
      4 protection   X = (X or Xbest) + k1 (a1 Xbest - a2 Xk)
                         + k2 rho (a3 (X2 - X1)) + a2/2 (Xu1 - Xu2),
                     Xk = l2 (Xp - Xr) + Xr, rho = alpha (2 rand - 1),
                     alpha = beta sin(3pi/2 + sin(3pi/2 beta)),
                     beta = bmin + (bmax - bmin)(1 - (t/T)^3)^2
    The run is time-budgeted, so t/T is the elapsed fraction of the budget
    and t (iteration counter, from 1) is used only in the hunting step."""
    lo, hi = _bounds(); D = 2 * N_NODES; NP = BASE_POP
    x = rng.uniform(lo, hi, (NP, D)); f = np.array([T.update(xi) for xi in x])
    b = int(np.argmax(f)); xb, fb = x[b].copy(), f[b]
    w = rng.uniform(0.05, 0.95)
    t = 0

    def try_set(i, cand):
        nonlocal xb, fb
        cand = np.clip(cand, lo, hi); fc = T.update(cand)
        if fc > f[i]:
            x[i], f[i] = cand, fc
            if fc > fb:
                xb, fb = cand.copy(), fc
        return T.alive()

    while T.alive():
        t += 1
        tau = min(T.frac(), 1.0)
        w = abs(1.0 - 2.0 * w ** 4)
        p = 2.0 * (1.0 - tau ** 0.25 + w * (tau ** 0.25 - tau ** 3))
        beta = GKSO_BETA_MIN + (GKSO_BETA_MAX - GKSO_BETA_MIN) * (1.0 - tau ** 3) ** 2
        alpha = beta * np.sin(1.5 * np.pi + np.sin(1.5 * np.pi * beta))
        for i in range(NP):                                    # 1 hunting
            if not try_set(i, x[i] + (lo + rng.random(D) * (hi - lo)) / ((i + 1) * t)):
                return
        for i in range(NP):                                    # 2 movement
            s_ = GKSO_M * GKSO_I ** rng.random()
            xh = s_ * (xb - x[i])
            if not try_set(i, (xh + x[i - 1]) / 2.0):
                return
        for i in range(NP):                                    # 3 foraging
            lam = rng.choice([-1.0, 1.0])
            cand = xb + rng.random(D) * (xb - x[i]) + lam * p ** 2 * (xb - x[i])
            if not try_set(i, cand):
                return
        for i in range(NP):                                    # 4 self-protection
            if rng.random() >= GKSO_PROTECT_P:
                continue
            mu = rng.random()
            a1 = 2.0 * rng.random() if mu < 0.5 else 1.0
            a2 = rng.random() if mu < 0.5 else 1.0
            a3 = rng.random() if mu < 0.5 else 1.0
            l2 = 1.0 if mu < 0.5 else 0.0
            k1, k2 = rng.uniform(-1, 1), rng.normal()
            r1, r2, u1, u2, pp = rng.choice(NP, 5, replace=True)
            xr = rng.uniform(lo, hi)
            xk = l2 * (x[pp] - xr) + xr
            rho = alpha * (2.0 * rng.random() - 1.0)
            start = x[i] if rng.random() < 0.5 else xb
            cand = start + k1 * (a1 * xb - a2 * xk) + k2 * rho * (a3 * (x[r2] - x[r1])) + a2 / 2.0 * (x[u1] - x[u2])
            if not try_set(i, cand):
                return


def run_woa_lfga(T, rng):
    """WOA-LFGA after Xu, Zhang & Zhang (2023): tent-map initialisation
    (u = 0.3), WOA encircling and spiral (b = 1), Levy-flight search
    (beta = 1.5, alpha0 = 1.6), and from 20 % of the budget a GA step that
    replaces the worst 20 % by one-point crossover of two elite (top 10 %)
    parents with 0.2 gene mutation; modular boundary handling.
    Reconstruction choices where the paper is not explicit: F ~ U(-2, 2)
    per dimension, uniform re-sampling as mutation."""
    lo, hi = _bounds(); D = 2 * N_NODES; span = hi - lo
    z = np.empty((BASE_POP, D)); z[0] = rng.random(D)
    for k in range(1, BASE_POP):
        prev = z[k - 1]
        z[k] = np.where(prev < 0.3, prev / 0.3, (1 - prev) / 0.7)
        z[k] = np.clip(z[k] + 1e-6 * rng.random(D), 0, 1)
    x = lo + z * span
    f = np.array([T.update(xi) for xi in x])
    wrap = lambda v: (v - lo) % span + lo
    n_el, n_bad = max(2, int(round(0.1 * BASE_POP))), max(1, int(round(0.2 * BASE_POP)))
    while T.alive():
        a = 2.0 * (1.0 - min(T.frac(), 1.0)); best = x[np.argmax(f)].copy()
        for i in range(BASE_POP):
            A = 2 * a * rng.random() - a; C = 2 * rng.random()
            if rng.random() < 0.5:
                if abs(A) < 1:
                    new = best - A * np.abs(C * best - x[i])
                else:
                    alpha = rng.random(D) * 1.6; lv = _levy(rng, D)
                    if rng.random() > 0.95:
                        xr = x[rng.integers(BASE_POP)]
                        new = x[i] + np.sign(rng.random() - 0.5) * alpha * (xr - x[i]) * lv
                    else:
                        F = rng.uniform(-2, 2, D)
                        new = best + F * alpha * (best - x[i]) * lv
            else:
                l = rng.uniform(-1, 1)
                new = np.abs(best - x[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
            x[i] = wrap(new); f[i] = T.update(x[i])
            if not T.alive():
                return
        if T.frac() >= 0.2:
            order = np.argsort(-f); elite = order[:n_el]
            for k in order[-n_bad:]:
                p1, p2 = x[rng.choice(elite, 2, replace=False)]
                cut = int(rng.integers(1, D))
                child = np.concatenate([p1[:cut], p2[cut:]])
                mut = rng.random(D) < 0.2
                child[mut] = rng.uniform(lo[mut], hi[mut])
                x[k] = child; f[k] = T.update(child)
                if not T.alive():
                    return


TIMED.update({"GKSO": run_gkso, "WOA_LFGA": run_woa_lfga})


# ---------------------------------------------------------------- new learning-based move selectors
CNN_POOL = 5                                   # 110 x 110 -> 22 x 22
_NFULL = len(_FULL["GRID"])


def coverage_image(st):
    """3 x 22 x 22 image of the current coverage: hole / covered once / overlap."""
    cnt = np.asarray(st.count)
    img = np.zeros((3, _NFULL), np.float32)
    idx = np.flatnonzero(_VALID) if CURRENT_FIELD == "I" else np.arange(_NFULL)
    img[0, idx] = (cnt == 0); img[1, idx] = (cnt == 1); img[2, idx] = (cnt >= 2)
    g = GRID_RES // CNN_POOL
    img = img.reshape(3, GRID_RES, GRID_RES)[:, :g * CNN_POOL, :g * CNN_POOL]
    return img.reshape(3, g, CNN_POOL, g, CNN_POOL).mean(axis=(2, 4)).astype(np.float32)


class CNNMoveQNet(nn.Module):
    """Same DQN head as the GNN version; the graph encoder is replaced by a CNN
    on the coverage image (16-d embedding, as the GNN)."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.q = nn.Sequential(nn.Linear(8 + N_MOVE_OPS + NGNN_EMB, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, N_MOVE_OPS))

    def forward(self, base, img):
        return self.q(torch.cat([base, self.cnn(img)], dim=1))


class CNNMovePolicy(LearnedSearchPolicy):
    def __init__(self):
        super().__init__("no_gnn")              # node choice: same redundancy prior as PROPOSED
        self.mode = "cnn_dqn"
        self.q, self.q_t = CNNMoveQNet().to(DEVICE), CNNMoveQNet().to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.q_opt = optim.Adam(self.q.parameters(), lr=MDQN_LR)
        self._st = None

    def learn_nodes(self, *args, **kw):         # node choice is not learned (as in PROPOSED)
        return

    def node_probs(self, st):
        self._st = st
        prior = st.redundancy_priority() + 0.05
        p = (1.0 - NODE_UNIFORM_MIX) * prior / prior.sum() + NODE_UNIFORM_MIX / st.n
        return p, coverage_image(st), None, np.zeros(NGNN_EMB, np.float32)

    def choose_op(self, rng, s, progress, graph=None):
        eps = max(MDQN_EPS_END, MDQN_EPS_START * np.exp(-self.decisions / MDQN_EPS_DECAY))
        if rng.random() < eps:
            op = int(rng.integers(N_MOVE_OPS))
        else:
            with torch.no_grad():
                qv = self.q(torch.from_numpy(s[:8 + N_MOVE_OPS]).unsqueeze(0).to(DEVICE),
                            torch.from_numpy(graph[0]).unsqueeze(0).to(DEVICE))
            op = int(torch.argmax(qv, dim=1).item())
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        k = 8 + N_MOVE_OPS
        self.q_buf.append((s[:k], graph[0], op, r, s2[:k], coverage_image(self._st)))
        self.decisions += 1
        if len(self.q_buf) >= MDQN_BATCH and self.decisions % GQ_TRAIN_EVERY == 0:
            b = [self.q_buf[j] for j in rng.choice(len(self.q_buf), GQ_BATCH, replace=False)]
            Tn = lambda arr: torch.from_numpy(np.stack(arr)).to(DEVICE)
            bs, bi = Tn([t[0] for t in b]), Tn([t[1] for t in b])
            ac = torch.tensor([t[2] for t in b], device=DEVICE).unsqueeze(1)
            rw = torch.tensor([t[3] for t in b], dtype=torch.float32, device=DEVICE)
            ns, ni = Tn([t[4] for t in b]), Tn([t[5] for t in b])
            qv = self.q(bs, bi).gather(1, ac).squeeze(1)
            with torch.no_grad():
                tgt = rw + MDQN_GAMMA * self.q_t(ns, ni).max(dim=1).values
            loss = nn.functional.smooth_l1_loss(qv, tgt)
            self.q_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q_opt.step(); self.q_losses.append(float(loss.item()))
        if self.decisions % MDQN_TARGET_EVERY == 0:
            self.q_t.load_state_dict(self.q.state_dict())


QL_ALPHA = 0.1


class QLearnMovePolicy(LearnedSearchPolicy):
    """Tabular Q-learning, no neural network. State = progress (4 bins) x
    acceptance rate (3) x last gain sign (3) x last move type (5 + none)."""

    def __init__(self):
        super().__init__("no_gnn")
        self.mode = "qlearn"
        self.Q = np.zeros((4 * 3 * 3 * (N_MOVE_OPS + 1), N_MOVE_OPS))

    def learn_nodes(self, *args, **kw):
        return

    @staticmethod
    def _sid(s):
        pr = min(int(s[4] * 4), 3)
        ac = 0 if s[6] < 0.1 else (1 if s[6] < 0.4 else 2)
        g = 0 if s[7] < -1e-3 else (1 if s[7] <= 1e-3 else 2)
        oh = s[8:8 + N_MOVE_OPS]
        lo_ = int(np.argmax(oh)) if oh.max() > 0 else N_MOVE_OPS
        return ((pr * 3 + ac) * 3 + g) * (N_MOVE_OPS + 1) + lo_

    def choose_op(self, rng, s, progress, graph=None):
        eps = max(MDQN_EPS_END, MDQN_EPS_START * np.exp(-self.decisions / MDQN_EPS_DECAY))
        q = self.Q[self._sid(s)]
        op = int(rng.integers(N_MOVE_OPS)) if rng.random() < eps else int(rng.choice(np.flatnonzero(q == q.max())))
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        a, b = self._sid(s), self._sid(s2)
        self.Q[a, op] += QL_ALPHA * (r + MDQN_GAMMA * self.Q[b].max() - self.Q[a, op])
        self.decisions += 1


_run_method_p3 = run_method


def run_method(key, seed):
    if key == "CNN_DQN":
        set_seed(seed)
        return run_framework(seed, "no_gnn", CNNMovePolicy())
    if key == "QLEARN":
        set_seed(seed)
        return run_framework(seed, "no_gnn", QLearnMovePolicy())
    return _run_method_p3(key, seed)


# ---------------------------------------------------------------- runs
def budget_field(field):
    if os.path.exists(RUNS_FILE):
        d = pd.read_csv(RUNS_FILE)
        p = d[(d.Field == field) & (d.Method == "PROPOSED")]["Runtime (s)"]
        if len(p):
            return float(p.mean())
    return DEFAULT_BUDGET_S


def budget_s():                       # used by run_method for timed methods
    return budget_field(CURRENT_FIELD)


SHARE_KEYS = list(share_from_counts(np.zeros((2, N_MOVE_OPS))).keys())   # fixed CSV columns


def jobs_todo():
    done = done_keys(RUNS_FILE, ["Field", "Method", "Seed"])
    jobs = []
    for seed in SEEDS4:                       # seed by seed: rectangle, then irregular
        if RUN_BLOCKS["R"] and seed in R_SEEDS:
            jobs += [("R", k, seed) for k in R_METHODS if ("R", k, seed) not in done]
        if RUN_BLOCKS["I"] and seed in I_SEEDS:
            jobs += [("I", k, seed) for k in I_METHODS if ("I", k, seed) not in done]
    return jobs


def run_all_p4(t_start):
    jobs = jobs_todo()
    print(f"\nRUNS: {len(jobs)} to do", flush=True)
    t0 = time.time()
    for j, (field, key, seed) in enumerate(jobs, 1):
        if (time.time() - t_start) / 3600.0 > STOP_AFTER_H:
            print(f"\nSTOP: {STOP_AFTER_H} h reached; {len(jobs) - j + 1} runs left for a later commit.", flush=True)
            break
        set_field(field)
        B = budget_field(field)
        ts = time.time()
        pos, rt, evals, extra = run_method(key, seed)
        pos = clip_positions(np.asarray(pos, float))
        np.save(os.path.join(POS_DIR4, f"{field}_{key}_N{N_NODES}_seed{seed}.npy"), pos)
        row = {"Field": field, **layout_row(key, seed, pos)}
        append_csv(LAYOUT_FILE, row)
        append_csv(RUNS_FILE, {"Field": field, "Method": key, "Seed": seed, "Runtime (s)": rt,
                               "Evaluations/moves": evals, "Budget (s)": B,
                               **{k: np.nan for k in SHARE_KEYS}, **extra})
        eta = (time.time() - t0) / j * (len(jobs) - j)
        print(f"[{j:3d}/{len(jobs)}] {field} {key:11s} seed={seed:2d} fine C={row['Coverage fine (%)']:.2f} "
              f"O={row['Overlap fine (%)']:.2f} EE={row['EE fine (%)']:.3f} out={row['Outside sensing (% of field)']:.2f} "
              f"({time.time()-ts:.0f}s, ETA {eta/60:.0f} min)", flush=True)
    set_field("R")


# ---------------------------------------------------------------- statistics
M5 = METRICS[:5]


def report_p4():
    if not os.path.exists(LAYOUT_FILE):
        print("No results yet."); return
    A = pd.read_csv(LAYOUT_FILE).drop_duplicates(["Field", "Method", "Seed"], keep="last")
    R = pd.read_csv(RUNS_FILE).drop_duplicates(["Field", "Method", "Seed"], keep="last")
    out = {}
    LR, LI = A[A.Field == "R"], A[A.Field == "I"]
    dec = []

    def fam(L, pairs, name):
        pairs = [p for p in pairs if p[0] in set(L.Method) and p[1] in set(L.Method)]
        return paired_tests(L, pairs, M5, name) if pairs else pd.DataFrame()

    def decide(t, title):
        e = t[t.Metric == "EE fine (%)"] if len(t) else t
        if len(e) == 2:
            ok = bool(all((e["Mean difference"] > 0) & (e["p (Holm)"] < 0.05)))
            dec.append(f"{title}: " + ("SUPPORTED" if ok else "NOT SUPPORTED") + " (" + "; ".join(
                f"{r.Comparison}: diff {r['Mean difference']:+.3f}, p_Holm {r['p (Holm)']:.4f}" for _, r in e.iterrows()) + ")")

    if len(LR):
        out["R_descriptive"] = describe_table(LR, [m for m in R_METHODS if m in set(LR.Method)])
        t2 = fam(LR, T2, "rectangle T2: learning design (pre-specified)")
        nb = fam(LR, [("PROPOSED", "GKSO"), ("PROPOSED", "WOA_LFGA")], "rectangle: recent metaheuristics")
        out["R_tests"] = pd.concat([t2, nb], ignore_index=True)
        decide(t2, "Rectangle T2, GNN-DQN better than CNN-DQN and Q-learning")
    if len(LI):
        out["I_descriptive"] = describe_table(LI, [m for m in I_METHODS if m in set(LI.Method)])
        t1 = fam(LI, PRIMARY, "irregular T1: learned vs simple selectors (pre-specified)")
        t2 = fam(LI, T2, "irregular T2: learning design (pre-specified)")
        sec = fam(LI, [("PROPOSED", "DQN_ONLY"), ("PROPOSED", "ORIGINAL"), ("PROPOSED", "SA_PLAIN"),
                       ("BANDIT_UCB", "RANDOM_MOVE"), ("RANDOM_MOVE", "ORIGINAL")], "irregular secondary (components)")
        base = fam(LI, [("PROPOSED", m) for m in ["CVT", "PSO", "GA", "GKSO", "WOA_LFGA"]], "irregular baselines")
        out["I_tests"] = pd.concat([t1, t2, sec, base], ignore_index=True)
        decide(t1, "Irregular T1, GNN-DQN better than random and UCB move selection")
        decide(t2, "Irregular T2, GNN-DQN better than CNN-DQN and Q-learning")
    ops = R.groupby(["Field", "Method"])[[c for c in R.columns if c.startswith("Share ")]].mean() if any(
        c.startswith("Share ") for c in R.columns) else pd.DataFrame()
    with pd.ExcelWriter("p4_results.xlsx") as xw:
        for k, v in out.items():
            v.to_excel(xw, sheet_name=k, index=False)
        if len(ops):
            ops.to_excel(xw, sheet_name="move_type_usage")
        R.groupby(["Field", "Method"])[["Runtime (s)", "Evaluations/moves"]].mean().to_excel(xw, sheet_name="runtime")
        A.to_excel(xw, sheet_name="all_layouts", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    ff = lambda v: f"{v:.4f}"
    show = ["Method", "Runs"] + [f"{c} mean" for c, _ in M5] + ["Pass opt-grid (%)", "Pass fine (%)"]
    keep = ["Family", "Metric", "Comparison", "Seeds", "Mean first", "Mean second", "Mean difference",
            "First better / tie / worse", "p (Wilcoxon)", "p (Holm)", "A12", "Effect", "Significant (0.05)"]
    for blk, title in (("R", "BLOCK R: rectangular field"), ("I", "BLOCK I: irregular field with obstacles")):
        if f"{blk}_descriptive" in out:
            print("\n" + "=" * 120 + f"\n{title}\n" + "=" * 120)
            d = out[f"{blk}_descriptive"]; print(d[[c for c in show if c in d.columns]].to_string(index=False, float_format=ff))
            t = out[f"{blk}_tests"]; print(t[[c for c in keep if c in t.columns]].to_string(index=False, float_format=ff))
    print("\nPRE-SPECIFIED DECISIONS (EE on the 1 m grid, Holm within each family)")
    for d in dec:
        print("  " + d)


# ---------------------------------------------------------------- figures (PNG, 600 dpi, journal style)
def make_figures_p4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8.5, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                         "ytick.right": True, "axes.grid": False, "savefig.dpi": 600})
    A = pd.read_csv(LAYOUT_FILE).drop_duplicates(["Field", "Method", "Seed"], keep="last")

    def boxes(ax, L, keys, col):
        data = [L[L.Method == k][col].dropna().values for k in keys]
        bp = ax.boxplot(data, positions=range(len(keys)), widths=0.55, showfliers=True,
                        flierprops=dict(marker="o", ms=3, mfc="white", mew=0.7))
        for i, k in enumerate(keys):
            c = COLS3[k]
            bp["boxes"][i].set(color=c, lw=1.1); bp["medians"][i].set(color=c, lw=1.3)
            for part in ("whiskers", "caps"):
                bp[part][2 * i].set(color=c); bp[part][2 * i + 1].set(color=c)
            bp["fliers"][i].set(markeredgecolor=c)
            if len(data[i]):
                ax.plot(i, np.mean(data[i]), marker="D", ms=3, color=c, zorder=3)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([LABELS[k] for k in keys], fontsize=6.5, rotation=40, ha="right")
        ax.set_ylabel(col)

    def save(fig, name):
        fig.savefig(name + ".png", dpi=600, facecolor="white", bbox_inches="tight"); plt.close(fig)

    for blk, keys, name in (("R", R_METHODS, "p4_fig_rect_all"), ("I", I_METHODS, "p4_fig_irregular_all")):
        L = A[A.Field == blk]; keys = [k for k in keys if k in set(L.Method)]
        if not keys:
            continue
        fig, axs = plt.subplots(2, 2, figsize=(9.5 if blk == "I" else 6.4, 5.8))
        for ax, col, t in zip(axs.ravel(), ["EE fine (%)", "Overlap fine (%)", "Coverage fine (%)", "Wasted sensing (% of field)"],
                              ["(a) EE", "(b) Overlap", "(c) Coverage", "(d) Wasted sensing"]):
            boxes(ax, L, keys, col); ax.set_title(t, loc="left", fontsize=9)
        fig.tight_layout(h_pad=1.5, w_pad=1.5); save(fig, name)

    L = A[A.Field == "I"]; seed = I_SEEDS[0]
    show = [k for k in ["PROPOSED", "CNN_DQN", "CVT", "GKSO"]
            if os.path.exists(os.path.join(POS_DIR4, f"I_{k}_N{N_IRREG}_seed{seed}.npy"))]
    if show:
        fig, axs = plt.subplots(1, len(show), figsize=(1.9 * len(show), 2.8))
        for ax, k in zip(np.atleast_1d(axs), show):
            pos = np.load(os.path.join(POS_DIR4, f"I_{k}_N{N_IRREG}_seed{seed}.npy"))
            r = L[(L.Method == k) & (L.Seed == seed)].iloc[0]
            ax.add_patch(Rectangle((0, 0), AREA_X, AREA_Y, fill=False, lw=1.0, ec="black"))
            for x0, y0, x1, y1 in BLOCKED:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="0.85", ec="black", lw=0.8, hatch="///"))
            for p in pos:
                ax.add_patch(Circle(p, R_SENSE, fill=False, ec=COLS3[k], lw=0.45))
            ax.scatter(pos[:, 0], pos[:, 1], s=4, color="black", zorder=3)
            ax.set_xlim(-30, AREA_X + 30); ax.set_ylim(-30, AREA_Y + 30); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(LABELS[k], fontsize=8)
            ax.text(0.5, -0.03, f"C {r['Coverage fine (%)']:.2f} %, O {r['Overlap fine (%)']:.2f} %",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        fig.tight_layout(w_pad=0.5); save(fig, f"p4_fig_irregular_layouts_seed{seed}")
    shutil.make_archive("positions_p4", "zip", POS_DIR4)
    print("Saved: p4_results.xlsx, p4_fig_*.png (600 dpi), positions_p4.zip", flush=True)


def run_part_g():
    t_start = time.time()
    if os.path.isdir("/kaggle/working"):
        os.chdir("/kaggle/working")
    os.makedirs(POS_DIR4, exist_ok=True)
    print("=" * 120 + f"\nPART G: CNN-DQN, Q-learning, GKSO, WOA-LFGA; rectangle and irregular field with obstacles\n"
          f"usable area {VALID_AREA:.0f} m^2; optimisation grid {len(_FULL['GRID'])} points "
          f"({int(_VALID.sum())} usable); 1 m grid {len(FINE)} / {len(_FINE_I)} points\n" + "=" * 120, flush=True)
    run_all_p4(t_start)
    report_p4()
    make_figures_p4()
    print(f"\nTOTAL TIME: {(time.time() - t_start)/3600:.2f} h", flush=True)




# ---------------------------------------------------------------- SA (GNN-DQN): one annealing chain with the learned ARLS
LABELS.update({"SA_LEARNED": "SA (GNN-DQN)"})


def run_sa_learned(seed, budget):
    """Same as SA_PLAIN (one annealing chain, equal time), but every ARLS
    block uses the GNN-DQN learned move selector (as in Study A)."""
    global ARLS_MODE, POLICY
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ARLS_MODE, POLICY = "gnn_dqn", LearnedSearchPolicy("gnn_dqn")
    t0 = time.time()
    pos = make_seed(N_NODES, 0.0, 0.0, SEED)
    best, moves, last_dt = None, 0, 0.0
    try:
        while time.time() - t0 < budget:
            el = time.time() - t0
            f0, f1 = el / budget, min(1.0, (el + max(last_dt, 0.05)) / budget)
            Ta = ARLS_T_START * (ARLS_T_END / ARLS_T_START) ** f0
            Tb = ARLS_T_START * (ARLS_T_END / ARLS_T_START) ** f1
            sa = ARLS_STEP_MAX + (ARLS_STEP_MIN - ARLS_STEP_MAX) * f0
            sb = ARLS_STEP_MAX + (ARLS_STEP_MIN - ARLS_STEP_MAX) * f1
            ts = time.time()
            cur, b = ARLS(pos, SA_CHUNK, Ta, Tb, sa, sb, rng)
            last_dt = time.time() - ts
            moves += SA_CHUNK
            pos = cur["pos"]
            if best is None or rank_key(b) > rank_key(best):
                best = b
    finally:
        POLICY, ARLS_MODE = None, "gnn_dqn"
    return best["pos"].copy(), time.time() - t0, moves, {}


_run_method_before_sa = run_method


def run_method(key, seed):
    if key == "SA_LEARNED":
        return run_sa_learned(seed, budget_s())
    return _run_method_before_sa(key, seed)


# =====================================================================
# PART L - STUDIES A, B AND C WITH ONE COMMON SET OF 18 METHODS
#          (single script; set RUN_STUDIES below)
# =====================================================================
#
#   Uses EXACTLY the code of Parts A-G above.  Only the runs are new.
#
#   COMMON METHODS (18) in every study:
#     PROPOSED, ORIGINAL (hand-crafted ARLS), RANDOM_MOVE, BANDIT_UCB,
#     DQN_ONLY, SA_PLAIN, SA_LEARNED, CNN_DQN, QLEARN, CVT (Lloyd), VFA,
#     PSO, GA, DE, GWO, WOA, GKSO, WOA_LFGA
#
#   STUDY A  main setting: 50 nodes, 250 x 300 m, 2,500-point optimisation
#            grid, NO outside penalty, fixed equal time 188 s; scored on the
#            2,500-point grid and the 1 m grid; seeds 20-39.
#     existing: PROPOSED, ORIGINAL, RANDOM_MOVE, BANDIT_UCB, SA_PLAIN,
#               SA_LEARNED, CVT (Study A);  CNN_DQN, QLEARN 20-39 and
#               GKSO, WOA_LFGA 20-29 (main-setting run, p5_runs.csv)
#     run here: DQN_ONLY, PSO, GA, DE, GWO, WOA, VFA (20-39); GKSO, WOA_LFGA (30-39)
#   STUDY B  rectangle, 12,100-point grid, outside penalty, 1 m scoring,
#            equal time = PROPOSED runtime on this machine; seeds 20-34.
#     run here: DE, GWO, WOA, VFA (26-34); CNN_DQN, QLEARN, GKSO, WOA_LFGA
#               (30-34; 20-29 exist from Study C, same field and scoring);
#               SA_LEARNED (20-34); PROPOSED 30, 31 (time calibration)
#   STUDY C  irregular field with obstacles, 44 nodes, same scoring as B;
#            seeds 20-29.
#     run here: SA_PLAIN, CVT, PSO, GA, GKSO, WOA_LFGA (29); DE, GWO, WOA,
#               VFA (20-29); SA_LEARNED (20-29); PROPOSED 28, 29 (calibration)
#
#   TIME: about 7.5 h for B + C and 8.5 h for A (Kaggle limit 12 h), so run
#   this SAME script twice (two notebooks at the same time, or one after
#   the other):   RUN_STUDIES = ["B", "C"]     and     RUN_STUDIES = ["A"]
#   Every run is saved at once; re-running the cell in the same session resumes.
#   OUTPUTS: p9bc_runs.csv, p9bc_layout_metrics.csv, p9bc_summary.csv, positions_p9bc.zip
# =====================================================================
import os, shutil

RUN_STUDIES = ["B", "C"]

RUNS_FILE = "p9bc_runs.csv"
LAYOUT_FILE = "p9bc_layout_metrics.csv"
POS_DIR4 = "positions_p9bc"
STOP_AFTER_H = 10.6
A_BUDGET_S = 188.0
CALIB = {"B": [30, 31], "C": [28, 29]}
PLAN = {
    "A": ([(m, s) for s in range(20, 40) for m in ["DQN_ONLY", "PSO", "GA", "DE", "GWO", "WOA", "VFA"]]
          + [(m, s) for s in range(30, 40) for m in ["GKSO", "WOA_LFGA"]]),
    "B": ([(m, s) for s in range(26, 35) for m in ["DE", "GWO", "WOA", "VFA"]]
          + [(m, s) for s in range(30, 35) for m in ["CNN_DQN", "QLEARN", "GKSO", "WOA_LFGA"]]
          + [("SA_LEARNED", s) for s in range(20, 35)]),
    "C": ([(m, 29) for m in ["SA_PLAIN", "CVT", "PSO", "GA", "GKSO", "WOA_LFGA"]]
          + [(m, s) for s in range(20, 30) for m in ["DE", "GWO", "WOA", "VFA"]]
          + [("SA_LEARNED", s) for s in range(20, 30)]),
}
if os.environ.get("P9_SMOKE"):                     # tiny settings for a quick code test only
    RUN_STUDIES = ["A", "B", "C"]
    A_BUDGET_S, CALIB = 3.0, {"B": [30], "C": [29]}
    PLAN = {"A": [(m, 20) for m in ["DQN_ONLY", "PSO", "VFA", "GKSO", "SA_LEARNED"]],
            "B": [(m, 26) for m in ["DE", "VFA"]] + [(m, 30) for m in ["CNN_DQN", "QLEARN", "WOA_LFGA", "SA_LEARNED"]],
            "C": [(m, 20) for m in ["GWO", "SA_LEARNED"]] + [(m, 29) for m in ["CVT", "GKSO"]]}
    GENERATIONS, ARLS_MOVES_INIT, ARLS_MOVES_POP, ARLS_MOVES_ELITE, ARLS_MOVES_FINAL = 1, 60, 60, 100, 100

# ---------------------------------------------------------------- switching between the study settings
def _grid_set(res):
    gx_ = np.linspace(0.0, AREA_X, res); gy_ = np.linspace(0.0, AREA_Y, res)
    GX_, GY_ = np.meshgrid(gx_, gy_)
    G = np.column_stack((GX_.ravel(), GY_.ravel()))
    return dict(GRID_RES=res, gx=gx_, gy=gy_, GX=GX_, GY=GY_, GRID=G, NUM_GRID=len(G), DX=gx_[1] - gx_[0], DY=gy_[1] - gy_[0])

_GRID_BC = dict(GRID_RES=GRID_RES, gx=gx, gy=gy, GX=GX, GY=GY, DX=DX, DY=DY, MAX_DISK=MAX_DISK)
_GRID_A = _grid_set(50)
_PEN = dict(W_OUT1=W_OUT1, W_OUT2=W_OUT2, separation_violations=separation_violations)
_NODE_VIOL = CoverageState.node_violations
CURRENT_STUDY = "B"


def set_study(st):
    """A: main setting (2,500-point grid, no outside penalty); B: rectangle with
    the Study B scoring; C: irregular field with the Study B scoring."""
    global CURRENT_STUDY, MAX_DISK, W_OUT1, W_OUT2, separation_violations
    CURRENT_STUDY = st
    set_field("I" if st == "C" else "R")
    if st == "A":
        globals().update(_GRID_A)
        MAX_DISK = len(disk_indices(np.array([AREA_X / 2.0, AREA_Y / 2.0])))
        W_OUT1 = W_OUT2 = 0.0
        separation_violations = lambda pos: 0.0
        CoverageState.node_violations = lambda self, i, p: 0.0
    else:
        globals().update(_GRID_BC)
        W_OUT1, W_OUT2 = _PEN["W_OUT1"], _PEN["W_OUT2"]
        separation_violations = _PEN["separation_violations"]
        CoverageState.node_violations = _NODE_VIOL


def budget_study(st):
    if st == "A":
        return A_BUDGET_S
    if os.path.exists(RUNS_FILE):
        d = pd.read_csv(RUNS_FILE)
        p = d[(d.Study == st) & (d.Method == "PROPOSED")]["Runtime (s)"]
        if len(p):
            return float(p.mean())
    return DEFAULT_BUDGET_S


def budget_s():                                     # used by the time-budgeted methods
    return budget_study(CURRENT_STUDY)


def jobs_all():
    done = done_keys(RUNS_FILE, ["Study", "Method", "Seed"])
    jobs = []
    for st in RUN_STUDIES:
        jobs += [(st, "PROPOSED", s) for s in CALIB.get(st, []) if (st, "PROPOSED", s) not in done]
        jobs += [(st, m, s) for m, s in sorted(PLAN[st], key=lambda t: (t[1], t[0])) if (st, m, s) not in done]
    return jobs


def run_part_l():
    t_start = time.time()
    if os.path.isdir("/kaggle/working"):
        os.chdir("/kaggle/working")
    os.makedirs(POS_DIR4, exist_ok=True)
    jobs = jobs_all()
    print("=" * 120 + f"\nPART L: common methods in Studies {', '.join(RUN_STUDIES)}; {len(jobs)} runs to do\n" + "=" * 120,
          flush=True)
    t0 = time.time()
    for j, (st, key, seed) in enumerate(jobs, 1):
        if (time.time() - t_start) / 3600.0 > STOP_AFTER_H:
            print(f"\nSTOP: {STOP_AFTER_H} h reached; {len(jobs) - j + 1} runs left (re-run the cell).", flush=True)
            break
        set_study(st)
        B = budget_study(st)
        ts = time.time()
        pos, rt, evals, extra = run_method(key, seed)
        pos = clip_positions(np.asarray(pos, float))
        np.save(os.path.join(POS_DIR4, f"{st}_{key}_N{N_NODES}_seed{seed}.npy"), pos)
        row = {"Study": st, "Grid points": NUM_GRID, **layout_row(key, seed, pos)}
        append_csv(LAYOUT_FILE, row)
        append_csv(RUNS_FILE, {"Study": st, "Method": key, "Seed": seed, "Runtime (s)": rt,
                               "Evaluations/moves": evals, "Budget (s)": B})
        eta = (time.time() - t0) / j * (len(jobs) - j)
        print(f"[{j:3d}/{len(jobs)}] Study {st} {key:11s} seed={seed:2d} grid C={row['Coverage opt-grid (%)']:.2f} "
              f"O={row['Overlap opt-grid (%)']:.2f} | 1 m C={row['Coverage fine (%)']:.2f} "
              f"O={row['Overlap fine (%)']:.2f} EE={row['EE fine (%)']:.3f} ({time.time()-ts:.0f}s, "
              f"budget {B:.0f}s, ETA {eta/60:.0f} min)", flush=True)
    set_study("B")
    if os.path.exists(LAYOUT_FILE):
        L = pd.read_csv(LAYOUT_FILE).drop_duplicates(["Study", "Method", "Seed"], keep="last")
        cols = ["EE opt-grid (%)", "Coverage fine (%)", "Overlap fine (%)", "EE fine (%)", "Wasted sensing (% of field)"]
        s = L.groupby(["Study", "Method"]).agg(Runs=("Seed", "count"), Seeds=("Seed", lambda v: f"{min(v)}-{max(v)}"),
                                               **{c: (c, "mean") for c in cols})
        s.to_csv("p9bc_summary.csv")
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
        print("\nSUMMARY OF THE NEW RUNS\n" + s.round(2).to_string(), flush=True)
    shutil.make_archive("positions_p9bc", "zip", POS_DIR4)
    print(f"\nSaved: {RUNS_FILE}, {LAYOUT_FILE}, p9bc_summary.csv, positions_p9bc.zip\n"
          f"TOTAL TIME: {(time.time() - t_start)/3600:.2f} h", flush=True)


run_part_l()
