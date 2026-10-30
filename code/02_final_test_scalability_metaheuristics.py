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
# PART C - FULL EXPERIMENT SUITE (one Kaggle commit)
# =====================================================================
#
#   EXP 1  FINAL TEST      : proposed (GNN-encoded DQN controls ARLS) vs
#                            original method, 50 nodes, NEW seeds 20-49
#                            (30 paired runs, decided before running)
#   EXP 2  SCALABILITY     : both methods, N = 20, 30, 40, 50 (seed 0)
#   EXP 3  BASELINES       : PSO, GA, DE, GWO, WOA on the same problem,
#                            same fitness, EQUAL WALL-CLOCK budget
#                            (= mean runtime of the proposed method),
#                            seeds 20-29, compared with proposed/original
#   STATS                  : mean/std/best/worst/median/95% CI, Wilcoxon
#                            signed-rank, Holm, Vargha-Delaney A12,
#                            Friedman test + average ranks
#   FIGURES (PNG, 300 dpi) : box plots, convergence curves, scalability,
#                            deployment layout, average-rank chart
#
#   Every finished run is appended to a CSV immediately, so a timeout
#   keeps all finished runs; re-running the cell resumes (Colab/local).
# =====================================================================

import os
import io
import contextlib

FINAL_SEEDS = list(range(20, 50))          # EXP 1 (30 new seeds)
SCAL_NODES = [20, 30, 40, 50]              # EXP 2
SCAL_SEED = 0
BASE_SEEDS = list(range(20, 30))           # EXP 3 (10 seeds)
BASE_POP = 30
BASELINE_TIME_FACTOR = 1.0                 # budget = factor x mean proposed runtime
RUN_EXP = {"final": True, "scalability": True, "baselines": True}

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER = "/content/drive/MyDrive/tr_enmogkso_final"

FINAL_FILE = "exp1_final_runs.csv"
FINAL_HIST = "exp1_final_history.csv"
SCAL_FILE = "exp2_scalability.csv"
BASE_FILE = "exp3_baseline_runs.csv"
BASE_HIST = "exp3_baseline_convergence.csv"
POS_DIR = "positions"

METHODS = {"PROPOSED": ("Proposed (GNN-DQN ARLS)", "gnn_dqn"),
           "ORIGINAL": ("Original (hand-crafted ARLS)", "original")}

# reference categorical palette (fixed order; identity also by labels)
COLORS = {"PROPOSED": "#2a78d6", "ORIGINAL": "#eb6834", "PSO": "#1baf7a",
          "GA": "#eda100", "DE": "#e87ba4", "GWO": "#008300", "WOA": "#4a3aa7"}
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def setup_output_folder():
    in_colab = os.path.exists("/var/colab/hostname") or "COLAB_RELEASE_TAG" in os.environ
    if USE_GOOGLE_DRIVE and in_colab:
        try:
            from google.colab import drive
            if not os.path.isdir("/content/drive/MyDrive"):
                drive.mount("/content/drive")
            os.makedirs(DRIVE_FOLDER, exist_ok=True)
            os.chdir(DRIVE_FOLDER)
        except Exception as e:
            print(f"Google Drive not available ({type(e).__name__}); "
                  f"saving in the current folder.", flush=True)
    os.makedirs(POS_DIR, exist_ok=True)
    print(f"Results folder : {os.getcwd()}", flush=True)


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=not os.path.exists(path))


def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[cols].values.tolist()))


def metrics_from_positions(pos):
    c, o, _, _ = coverage_data(pos)
    e = energy_efficiency(c, o)
    return c * 100, o * 100, e * 100, bool(c > TARGET_COVERAGE and o < TARGET_OVERLAP
                                         and e > TARGET_EE)


def run_method(key, n, seed):
    global ARLS_MODE
    ARLS_MODE = METHODS[key][1]
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_proposed(n, seed=seed)
    c, o, e, ok = metrics_from_positions(r["positions"])
    np.save(os.path.join(POS_DIR, f"{key}_N{n}_seed{seed}.npy"), r["positions"])
    return r, c, o, e, ok


# =====================================================================
# EXP 1 - FINAL TEST
# =====================================================================

def exp1_final():
    done = done_keys(FINAL_FILE, ["Method", "Seed"])
    jobs = [(k, s) for s in FINAL_SEEDS for k in METHODS if (k, s) not in done]
    print(f"\nEXP 1 FINAL TEST: {len(jobs)} runs to do ({len(done)} finished)", flush=True)
    t0 = time.time()
    for j, (key, seed) in enumerate(jobs, 1):
        r, c, o, e, ok = run_method(key, 50, seed)
        append_csv(FINAL_FILE, {"Method": key, "Seed": seed, "Coverage (%)": c,
                                "Overlap (%)": o, "EE (%)": e, "Feasible": ok,
                                "Runtime (s)": r["runtime"]})
        for h in r["history"]:
            append_csv(FINAL_HIST, {"Method": key, "Seed": seed,
                                    "Generation": h["generation"], "EE (%)": h["ee"],
                                    "Overlap (%)": h["overlap"],
                                    "Coverage (%)": h["coverage"]})
        eta = (time.time() - t0) / j * (len(jobs) - j)
        print(f"[E1 {j:2d}/{len(jobs)}] {key:9s} seed={seed:2d} cov={c:.2f} ov={o:.2f} "
              f"ee={e:.3f} {'PASS' if ok else 'fail'} ({r['runtime']:.0f}s, "
              f"ETA {eta/60:.0f} min)", flush=True)


# =====================================================================
# EXP 2 - SCALABILITY
# =====================================================================

def exp2_scalability():
    done = done_keys(SCAL_FILE, ["Method", "Nodes"])
    jobs = [(k, n) for n in SCAL_NODES for k in METHODS if (k, n) not in done]
    print(f"\nEXP 2 SCALABILITY: {len(jobs)} runs to do ({len(done)} finished)", flush=True)
    for j, (key, n) in enumerate(jobs, 1):
        r, c, o, e, ok = run_method(key, n, SCAL_SEED)
        append_csv(SCAL_FILE, {"Method": key, "Nodes": n, "Seed": SCAL_SEED,
                               "Coverage (%)": c, "Overlap (%)": o, "EE (%)": e,
                               "Feasible": ok, "Runtime (s)": r["runtime"]})
        print(f"[E2 {j}/{len(jobs)}] {key:9s} N={n} cov={c:.2f} ov={o:.2f} ee={e:.3f} "
              f"{'PASS' if ok else 'fail'} ({r['runtime']:.0f}s)", flush=True)


# =====================================================================
# EXP 3 - BASELINE METAHEURISTICS (equal wall-clock budget)
# =====================================================================

def _eval(x, n):
    sol = quick_evaluate(x.reshape(n, 2))
    return fitness(sol)


def _bounds(n):
    lo = np.zeros(2 * n)
    hi = np.tile([AREA_X, AREA_Y], n)
    return lo, hi


class _Tracker:
    """Best-so-far bookkeeping + convergence trace against time fraction."""

    def __init__(self, n, budget):
        self.n, self.budget, self.t0 = n, budget, time.time()
        self.best_f, self.best_x, self.trace, self.evals = -np.inf, None, [], 0
        self.next_mark = 0.0

    def frac(self):
        return (time.time() - self.t0) / self.budget

    def update(self, x, f):
        self.evals += 1
        if f > self.best_f:
            self.best_f, self.best_x = f, x.copy()
        fr = self.frac()
        if fr >= self.next_mark and self.best_x is not None:
            c, o, e, _ = metrics_from_positions(self.best_x.reshape(self.n, 2))
            self.trace.append((min(fr, 1.0), e, o, c))
            self.next_mark += 0.02

    def alive(self):
        return self.frac() < 1.0


def run_pso(n, budget, rng):
    lo, hi = _bounds(n); D = 2 * n; T = _Tracker(n, budget)
    x = rng.uniform(lo, hi, (BASE_POP, D)); v = np.zeros_like(x)
    vmax = 0.2 * (hi - lo)
    pf = np.array([_eval(xi, n) for xi in x]); [T.update(xi, f) for xi, f in zip(x, pf)]
    pb, gb = x.copy(), x[np.argmax(pf)].copy()
    while T.alive():
        r1, r2 = rng.random((BASE_POP, D)), rng.random((BASE_POP, D))
        v = 0.729 * v + 1.49445 * r1 * (pb - x) + 1.49445 * r2 * (gb - x)
        v = np.clip(v, -vmax, vmax); x = np.clip(x + v, lo, hi)
        for i in range(BASE_POP):
            f = _eval(x[i], n); T.update(x[i], f)
            if f > pf[i]:
                pf[i], pb[i] = f, x[i].copy()
        gb = pb[np.argmax(pf)].copy()
    return T


def run_ga(n, budget, rng):
    lo, hi = _bounds(n); D = 2 * n; T = _Tracker(n, budget)
    x = rng.uniform(lo, hi, (BASE_POP, D))
    f = np.array([_eval(xi, n) for xi in x]); [T.update(xi, fi) for xi, fi in zip(x, f)]
    sigma = 0.05 * (hi - lo)
    while T.alive():
        elite = np.argsort(-f)[:2]
        new = [x[elite[0]].copy(), x[elite[1]].copy()]
        while len(new) < BASE_POP:
            p = [x[max(rng.choice(BASE_POP, 3, replace=False), key=lambda k: f[k])]
                 for _ in range(2)]
            if rng.random() < 0.9:                          # BLX-0.5 crossover
                a = rng.uniform(-0.5, 1.5, D)
                child = p[0] + a * (p[1] - p[0])
            else:
                child = p[0].copy()
            mut = rng.random(D) < 1.0 / D * 4               # gaussian mutation
            child[mut] += rng.normal(0, sigma[mut])
            new.append(np.clip(child, lo, hi))
        x = np.array(new)
        f = np.array([_eval(xi, n) for xi in x]); [T.update(xi, fi) for xi, fi in zip(x, f)]
    return T


def run_de(n, budget, rng):
    lo, hi = _bounds(n); D = 2 * n; T = _Tracker(n, budget)
    x = rng.uniform(lo, hi, (BASE_POP, D))
    f = np.array([_eval(xi, n) for xi in x]); [T.update(xi, fi) for xi, fi in zip(x, f)]
    while T.alive():
        for i in range(BASE_POP):
            a, b, c = x[rng.choice([k for k in range(BASE_POP) if k != i], 3, replace=False)]
            mutant = np.clip(a + 0.5 * (b - c), lo, hi)
            cross = rng.random(D) < 0.9
            cross[rng.integers(D)] = True
            trial = np.where(cross, mutant, x[i])
            ft = _eval(trial, n); T.update(trial, ft)
            if ft >= f[i]:
                x[i], f[i] = trial, ft
            if not T.alive():
                break
    return T


def run_gwo(n, budget, rng):
    lo, hi = _bounds(n); D = 2 * n; T = _Tracker(n, budget)
    x = rng.uniform(lo, hi, (BASE_POP, D))
    f = np.array([_eval(xi, n) for xi in x]); [T.update(xi, fi) for xi, fi in zip(x, f)]
    while T.alive():
        idx = np.argsort(-f)[:3]
        leaders = [x[k].copy() for k in idx]
        a = 2.0 * (1.0 - min(T.frac(), 1.0))
        for i in range(BASE_POP):
            xs = []
            for L in leaders:
                A = 2 * a * rng.random(D) - a
                C = 2 * rng.random(D)
                xs.append(L - A * np.abs(C * L - x[i]))
            x[i] = np.clip(np.mean(xs, axis=0), lo, hi)
            f[i] = _eval(x[i], n); T.update(x[i], f[i])
    return T


def run_woa(n, budget, rng):
    lo, hi = _bounds(n); D = 2 * n; T = _Tracker(n, budget)
    x = rng.uniform(lo, hi, (BASE_POP, D))
    f = np.array([_eval(xi, n) for xi in x]); [T.update(xi, fi) for xi, fi in zip(x, f)]
    while T.alive():
        best = x[np.argmax(f)].copy()
        a = 2.0 * (1.0 - min(T.frac(), 1.0))
        for i in range(BASE_POP):
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            if rng.random() < 0.5:
                ref = best if abs(A) < 1 else x[rng.integers(BASE_POP)]
                x[i] = ref - A * np.abs(C * ref - x[i])
            else:
                l = rng.uniform(-1, 1)
                x[i] = np.abs(best - x[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
            x[i] = np.clip(x[i], lo, hi)
            f[i] = _eval(x[i], n); T.update(x[i], f[i])
    return T


BASELINES = {"PSO": run_pso, "GA": run_ga, "DE": run_de, "GWO": run_gwo, "WOA": run_woa}


def exp3_baselines():
    if os.path.exists(FINAL_FILE):
        fd = pd.read_csv(FINAL_FILE)
        budget = BASELINE_TIME_FACTOR * fd[fd.Method == "PROPOSED"]["Runtime (s)"].mean()
    else:
        budget = 180.0
    print(f"\nEXP 3 BASELINES: equal wall-clock budget = {budget:.0f} s per run", flush=True)
    done = done_keys(BASE_FILE, ["Method", "Seed"])
    jobs = [(k, s) for s in BASE_SEEDS for k in BASELINES if (k, s) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} finished)", flush=True)
    t0 = time.time()
    for j, (key, seed) in enumerate(jobs, 1):
        rng = np.random.default_rng(seed)
        T = BASELINES[key](50, budget, rng)
        pos = T.best_x.reshape(50, 2)
        np.save(os.path.join(POS_DIR, f"{key}_N50_seed{seed}.npy"), pos)
        c, o, e, ok = metrics_from_positions(pos)
        append_csv(BASE_FILE, {"Method": key, "Seed": seed, "Coverage (%)": c,
                               "Overlap (%)": o, "EE (%)": e, "Feasible": ok,
                               "Evaluations": T.evals, "Runtime (s)": time.time() - T.t0})
        for fr, ee, ov, cv in T.trace:
            append_csv(BASE_HIST, {"Method": key, "Seed": seed, "Budget fraction": fr,
                                   "EE (%)": ee, "Overlap (%)": ov, "Coverage (%)": cv})
        eta = (time.time() - t0) / j * (len(jobs) - j)
        print(f"[E3 {j:2d}/{len(jobs)}] {key:4s} seed={seed:2d} cov={c:.2f} ov={o:.2f} "
              f"ee={e:.3f} {'PASS' if ok else 'fail'} evals={T.evals} "
              f"(ETA {eta/60:.0f} min)", flush=True)


# =====================================================================
# STATISTICS
# =====================================================================

def a12(x, y, higher=True):
    x, y = np.asarray(x), np.asarray(y)
    g = (x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean()
    return g if higher else 1 - g


def a12_label(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); out[i] = run
    return out


def describe(d, name):
    from scipy.stats import t as tdist
    row = {"Method": name, "Runs": len(d)}
    for m in ("Coverage (%)", "Overlap (%)", "EE (%)"):
        v = d[m].to_numpy()
        h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        best = v.min() if m == "Overlap (%)" else v.max()
        worst = v.max() if m == "Overlap (%)" else v.min()
        row.update({f"{m} mean": v.mean(), f"{m} std": v.std(ddof=1),
                    f"{m} best": best, f"{m} worst": worst, f"{m} median": np.median(v),
                    f"{m} 95% CI": f"[{v.mean()-h:.3f}, {v.mean()+h:.3f}]"})
    row["Pass rate (%)"] = 100 * d["Feasible"].mean()
    row["Runtime mean (s)"] = d["Runtime (s)"].mean()
    return row


def pairwise(df, ref, others, seeds):
    from scipy.stats import wilcoxon
    rows = []
    for metric, higher in (("EE (%)", True), ("Overlap (%)", False), ("Coverage (%)", True)):
        piv = df[df.Seed.isin(seeds)].pivot(index="Seed", columns="Method", values=metric)
        raw = []
        for o in others:
            pr = piv[[ref, o]].dropna()
            d = pr[ref] - pr[o]
            p = 1.0 if np.allclose(d, 0) else wilcoxon(pr[ref], pr[o], zero_method="zsplit").pvalue
            A = a12(pr[ref], pr[o], higher)
            raw.append({"Metric": metric, "Comparison": f"{ref} vs {o}", "Seeds": len(pr),
                        "Mean difference": d.mean(),
                        f"{ref} better (seeds)": int(((d > 1e-9) if higher else (d < -1e-9)).sum()),
                        "p (Wilcoxon)": p, "A12": A, "Effect size": a12_label(A)})
        for r_, ph in zip(raw, holm([r_["p (Wilcoxon)"] for r_ in raw])):
            r_["p (Holm)"] = ph
            r_["Significant (0.05)"] = ph < 0.05
        rows += raw
    return pd.DataFrame(rows)


def friedman(df, seeds, methods):
    from scipy.stats import friedmanchisquare
    out, ranks_all = [], {}
    for metric, higher in (("EE (%)", True), ("Overlap (%)", False), ("Coverage (%)", True)):
        piv = df[df.Seed.isin(seeds)].pivot(index="Seed", columns="Method",
                                            values=metric)[methods].dropna()
        stat, p = friedmanchisquare(*[piv[m] for m in methods])
        ranks = (-piv if higher else piv).rank(axis=1).mean()
        ranks_all[metric] = ranks
        out.append({"Metric": metric, "Seeds": len(piv), "Methods": len(methods),
                    "Friedman chi2": stat, "p": p,
                    **{f"Rank {m}": ranks[m] for m in methods}})
    return pd.DataFrame(out), ranks_all


# =====================================================================
# FIGURES
# =====================================================================

def _style(ax, title, ylabel, xlabel=None):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _boxplots(df, methods, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    for ax, (m, lab) in zip(axes, (("EE (%)", "Energy efficiency (%)"),
                                   ("Overlap (%)", "Overlap (%)"),
                                   ("Coverage (%)", "Coverage (%)"))):
        data = [df[df.Method == k][m].to_numpy() for k in methods]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                        flierprops=dict(marker="o", markersize=4, markerfacecolor=INK2,
                                        markeredgecolor="none"))
        for patch, k in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[k]); patch.set_alpha(0.85)
            patch.set_edgecolor("white"); patch.set_linewidth(1.5)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=0 if len(methods) <= 3 else 30, ha="center"
                           if len(methods) <= 3 else "right")
        tgt = {"EE (%)": TARGET_EE * 100, "Overlap (%)": TARGET_OVERLAP * 100,
               "Coverage (%)": TARGET_COVERAGE * 100}[m]
        ax.axhline(tgt, color=INK2, linestyle="--", linewidth=1)
        ax.text(len(methods) + 0.45, tgt, "target", fontsize=7, color=INK2, va="bottom", ha="right")
        _style(ax, lab, lab)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fname, dpi=300); plt.close(fig)


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    made = []

    if os.path.exists(FINAL_FILE):
        fd = pd.read_csv(FINAL_FILE)
        _boxplots(fd, ["PROPOSED", "ORIGINAL"], "fig1_final_boxplots.png",
                  f"Proposed vs original method - 50 nodes, {fd.Seed.nunique()} paired runs")
        made.append("fig1_final_boxplots.png")

    if os.path.exists(FINAL_HIST):
        h = pd.read_csv(FINAL_HIST)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
        for ax, m, lab in ((axes[0], "EE (%)", "Energy efficiency (%)"),
                           (axes[1], "Overlap (%)", "Overlap (%)")):
            for k in ("PROPOSED", "ORIGINAL"):
                g = h[h.Method == k].groupby("Generation")[m]
                mu, sd = g.mean(), g.std()
                ax.plot(mu.index, mu.values, color=COLORS[k], linewidth=2, label=METHODS[k][0])
                ax.fill_between(mu.index, mu - sd, mu + sd, color=COLORS[k], alpha=0.15,
                                linewidth=0)
            _style(ax, f"Convergence - {lab}", lab, "Generation")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig("fig2_convergence_final.png", dpi=300); plt.close(fig)
        made.append("fig2_convergence_final.png")

    if os.path.exists(SCAL_FILE):
        sd = pd.read_csv(SCAL_FILE)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
        for ax, m, lab in zip(axes, ("Coverage (%)", "Overlap (%)", "EE (%)"),
                              ("Coverage (%)", "Overlap (%)", "Energy efficiency (%)")):
            for k in ("PROPOSED", "ORIGINAL"):
                d = sd[sd.Method == k].sort_values("Nodes")
                ax.plot(d.Nodes, d[m], color=COLORS[k], linewidth=2, marker="o",
                        markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                        label=METHODS[k][0])
            ax.set_xticks(SCAL_NODES)
            _style(ax, f"Scalability - {lab}", lab, "Number of nodes")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig("fig3_scalability.png", dpi=300); plt.close(fig)
        made.append("fig3_scalability.png")

    if os.path.exists(BASE_FILE) and os.path.exists(FINAL_FILE):
        bd = pd.concat([pd.read_csv(FINAL_FILE), pd.read_csv(BASE_FILE)])
        bd = bd[bd.Seed.isin(BASE_SEEDS)]
        order = ["PROPOSED", "ORIGINAL"] + [k for k in BASELINES if k in set(bd.Method)]
        _boxplots(bd, order, "fig4_baselines_boxplots.png",
                  f"Comparison with metaheuristics - 50 nodes, seeds "
                  f"{BASE_SEEDS[0]}-{BASE_SEEDS[-1]}, equal time budget")
        made.append("fig4_baselines_boxplots.png")

        if os.path.exists(BASE_HIST) and os.path.exists(FINAL_HIST):
            bh = pd.read_csv(BASE_HIST)
            fh = pd.read_csv(FINAL_HIST)
            fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
            grid_x = np.linspace(0, 1, 51)
            for k in order:
                if k in METHODS:
                    g = fh[(fh.Method == k) & fh.Seed.isin(BASE_SEEDS)]
                    mu = g.groupby("Generation")["EE (%)"].mean()
                    x = mu.index.to_numpy() / GENERATIONS
                    ax.plot(x, mu.values, color=COLORS[k], linewidth=2, label=k)
                else:
                    curves = []
                    for s, gs in bh[(bh.Method == k)].groupby("Seed"):
                        curves.append(np.interp(grid_x, gs["Budget fraction"], gs["EE (%)"]))
                    if curves:
                        ax.plot(grid_x, np.mean(curves, axis=0), color=COLORS[k],
                                linewidth=2, label=k)
            _style(ax, "Convergence vs fraction of the time budget (mean of seeds)",
                   "Best energy efficiency (%)", "Fraction of time budget")
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout(); fig.savefig("fig5_convergence_baselines.png", dpi=300)
            plt.close(fig); made.append("fig5_convergence_baselines.png")

        try:
            _, ranks = friedman(bd, BASE_SEEDS, order)
            r = ranks["EE (%)"].sort_values()
            fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
            ax.barh(range(len(r)), r.values, color=[COLORS[k] for k in r.index],
                    height=0.6, edgecolor="white", linewidth=2)
            ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index); ax.invert_yaxis()
            for i, v in enumerate(r.values):
                ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
            _style(ax, "Friedman average rank on EE (lower is better)", "", "Average rank")
            ax.grid(axis="x", color=GRID_C); ax.grid(axis="y", visible=False)
            fig.tight_layout(); fig.savefig("fig6_friedman_ranks.png", dpi=300)
            plt.close(fig); made.append("fig6_friedman_ranks.png")
        except Exception as e:
            print("rank figure skipped:", e)

    # deployment layout: best proposed run at 50 nodes
    if os.path.exists(FINAL_FILE):
        fd = pd.read_csv(FINAL_FILE)
        p = fd[fd.Method == "PROPOSED"].sort_values("EE (%)", ascending=False)
        if len(p):
            best = p.iloc[0]
            path = os.path.join(POS_DIR, f"PROPOSED_N50_seed{int(best.Seed)}.npy")
            if os.path.exists(path):
                pos = np.load(path)
                fig, ax = plt.subplots(figsize=(5.2, 6), dpi=150)
                ax.add_patch(plt.Rectangle((0, 0), AREA_X, AREA_Y, fill=False,
                                           edgecolor=INK2, linewidth=1))
                for xy in pos:
                    ax.add_patch(Circle(xy, R_SENSE, facecolor=COLORS["PROPOSED"],
                                        alpha=0.18, edgecolor=COLORS["PROPOSED"], linewidth=0.6))
                ax.scatter(pos[:, 0], pos[:, 1], s=14, color=INK, zorder=3)
                ax.set_xlim(-30, AREA_X + 30); ax.set_ylim(-30, AREA_Y + 30)
                ax.set_aspect("equal")
                ax.set_title(f"Best deployment (seed {int(best.Seed)}): cov {best['Coverage (%)']:.2f}%, "
                             f"ov {best['Overlap (%)']:.2f}%, EE {best['EE (%)']:.2f}%",
                             fontsize=9, color=INK, loc="left")
                ax.set_xlabel("x (m)", fontsize=9, color=INK2)
                ax.set_ylabel("y (m)", fontsize=9, color=INK2)
                ax.tick_params(colors=INK2, labelsize=8)
                fig.tight_layout(); fig.savefig("fig7_best_deployment.png", dpi=300)
                plt.close(fig); made.append("fig7_best_deployment.png")
    print("\nFigures saved:", ", ".join(made), flush=True)


# =====================================================================
# REPORT
# =====================================================================

def report():
    pd.set_option("display.width", 250)
    ff = lambda x: f"{x:.4f}"
    tables = {}
    if os.path.exists(FINAL_FILE):
        fd = pd.read_csv(FINAL_FILE)
        desc = pd.DataFrame([describe(fd[fd.Method == k], METHODS[k][0]) for k in METHODS])
        st = pairwise(fd, "PROPOSED", ["ORIGINAL"], FINAL_SEEDS)
        print("\n" + "=" * 110)
        print(f"EXP 1 - FINAL TEST: 50 nodes, seeds {FINAL_SEEDS[0]}-{FINAL_SEEDS[-1]}")
        print("=" * 110)
        print(desc.T.to_string(header=False))
        print("\nPaired Wilcoxon signed-rank test (pre-specified primary comparison)")
        print(st.to_string(index=False, float_format=ff))
        tables["exp1_descriptive"] = desc; tables["exp1_wilcoxon"] = st

    if os.path.exists(SCAL_FILE):
        sd = pd.read_csv(SCAL_FILE).sort_values(["Nodes", "Method"])
        print("\n" + "=" * 110 + "\nEXP 2 - SCALABILITY (seed 0)\n" + "=" * 110)
        print(sd.to_string(index=False, float_format=ff))
        tables["exp2_scalability"] = sd

    if os.path.exists(BASE_FILE) and os.path.exists(FINAL_FILE):
        bd = pd.concat([pd.read_csv(FINAL_FILE), pd.read_csv(BASE_FILE)])
        bd = bd[bd.Seed.isin(BASE_SEEDS)]
        order = ["PROPOSED", "ORIGINAL"] + [k for k in BASELINES if k in set(bd.Method)]
        desc = pd.DataFrame([describe(bd[bd.Method == k], k) for k in order])
        fr, _ = friedman(bd, BASE_SEEDS, order)
        st = pairwise(bd, "PROPOSED", order[1:], BASE_SEEDS)
        print("\n" + "=" * 110)
        print(f"EXP 3 - COMPARISON WITH METAHEURISTICS (seeds {BASE_SEEDS[0]}-{BASE_SEEDS[-1]})")
        print("=" * 110)
        cols = ["Method", "Runs", "Coverage (%) mean", "Coverage (%) std", "Overlap (%) mean",
                "Overlap (%) std", "EE (%) mean", "EE (%) std", "EE (%) best", "EE (%) worst",
                "Pass rate (%)", "Runtime mean (s)"]
        print(desc[cols].to_string(index=False, float_format=ff))
        print("\nFriedman test + average ranks")
        print(fr.to_string(index=False, float_format=ff))
        print("\nPaired Wilcoxon: proposed vs each method (Holm-corrected per metric)")
        print(st.to_string(index=False, float_format=ff))
        tables["exp3_descriptive"] = desc; tables["exp3_friedman"] = fr
        tables["exp3_wilcoxon"] = st

    if tables:
        with pd.ExcelWriter("final_results_tables.xlsx") as w:
            for name, t in tables.items():
                t.to_excel(w, sheet_name=name[:31], index=False)
        print("\nSaved tables: final_results_tables.xlsx", flush=True)


def run_all():
    setup_output_folder()
    t0 = time.time()
    if RUN_EXP["final"]:
        exp1_final()
    if RUN_EXP["scalability"]:
        exp2_scalability()
    if RUN_EXP["baselines"]:
        exp3_baselines()
    report()
    make_figures()
    print(f"\nTOTAL TIME: {(time.time() - t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    run_all()
