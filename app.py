import random

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# PART 1: FIXED PARAMETERS (same as the tutorial code)
# ---------------------------------------------------------------
GEN = 100   # number of generations
POP = 50    # population size (number of schedules per generation)
EL_S = 2    # elitism: the 2 best schedules always survive

# Default values from the assignment. They stay in the code as constants.
# DEFAULT_MUT_R = 0.2 is outside the slider range (0.01 - 0.05, confirmed by
# the lecturer), so it is not on a slider. It is used only by "Run Default".
DEFAULT_CO_R = 0.8
DEFAULT_MUT_R = 0.2

CO_R_MIN, CO_R_MAX, CO_R_DEFAULT = 0.0, 0.95, 0.8
MUT_R_MIN, MUT_R_MAX, MUT_R_DEFAULT = 0.01, 0.05, 0.05


# ---------------------------------------------------------------
# PART 2: READ THE DATA
# ---------------------------------------------------------------
@st.cache_data
def load_ratings(path="program_ratings.csv"):
    df = pd.read_csv(path)
    programs = df.iloc[:, 0].tolist()                                # program names
    hours = [int(c.replace("Hour ", "")) for c in df.columns[1:]]    # 6 ... 23
    ratings = {row[0]: list(row[1:]) for row in df.itertuples(index=False)}
    return programs, hours, ratings


# ---------------------------------------------------------------
# PART 3: GENETIC ALGORITHM
# ---------------------------------------------------------------
def fitness(schedule, ratings):
    """Total rating: the program at slot i contributes its rating for slot i."""
    return sum(ratings[prog][i] for i, prog in enumerate(schedule))


def crossover(p1, p2):
    """Cut both parents at one point and swap the back parts."""
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def mutate(schedule, programs):
    """Replace ONE random slot with a random program."""
    child = schedule[:]
    child[random.randint(0, len(child) - 1)] = random.choice(programs)
    return child


def run_ga(programs, n_slots, ratings, co_r, mut_r, seed):
    random.seed(seed)

    # 1) initial population: random schedules
    population = [[random.choice(programs) for _ in range(n_slots)]
                  for _ in range(POP)]
    history = []

    for _ in range(GEN):
        # 2) sort by fitness, highest first
        population.sort(key=lambda s: fitness(s, ratings), reverse=True)
        history.append(fitness(population[0], ratings))

        # 3) elitism: keep the EL_S best schedules
        new_pop = population[:EL_S]

        # 4) build children until the population is full
        parents_pool = population[:POP // 2]          # selection: best 50%
        while len(new_pop) < POP:
            p1, p2 = random.sample(parents_pool, 2)
            if random.random() < co_r:                # does crossover happen?
                c1, c2 = crossover(p1, p2)
            else:                                     # no: children copy the parents
                c1, c2 = p1[:], p2[:]
            if random.random() < mut_r:               # does mutation happen?
                c1 = mutate(c1, programs)
            if random.random() < mut_r:
                c2 = mutate(c2, programs)
            new_pop.extend([c1, c2])
        population = new_pop[:POP]

    population.sort(key=lambda s: fitness(s, ratings), reverse=True)
    best = population[0]
    history.append(fitness(best, ratings))
    return best, fitness(best, ratings), history


# ---------------------------------------------------------------
# PART 4: STREAMLIT INTERFACE
# ---------------------------------------------------------------
st.set_page_config(page_title="TV Scheduling - Genetic Algorithm", layout="wide")
st.title("TV Program Scheduling using a Genetic Algorithm")

programs, hours, ratings = load_ratings()

# --- Seed ------------------------------------------------------------
# Same seed = exactly the same result every time. Change it in the sidebar.
seed_input = st.sidebar.number_input(
    "Random seed (same seed = repeatable results)", value=42, step=1
)
st.sidebar.write(f"GEN = {GEN}, POP = {POP}, EL_S = {EL_S}")


# st.session_state keeps results while the page is open (lost on refresh)
if "results" not in st.session_state:
    st.session_state["results"] = {}     # latest result of each trial
if "history" not in st.session_state:
    st.session_state["history"] = []     # every run


def do_run(label, co_r, mut_r, seed):
    """Run the GA once, add it to the history, and return the result."""
    best, score, hist = run_ga(programs, len(hours), ratings, co_r, mut_r, seed)
    st.session_state["history"].append({
        "Run": len(st.session_state["history"]) + 1,
        "Label": label,
        "CO_R": round(co_r, 2),
        "MUT_R": round(mut_r, 2),
        "Seed": seed,
        "Total Ratings": round(score, 2),
    })
    return {"co_r": co_r, "mut_r": mut_r, "best": best,
            "score": score, "history": hist}


def show_result(res):
    st.success(
        f"CO_R = {res['co_r']}, MUT_R = {res['mut_r']}  |  "
        f"Total Ratings = {res['score']:.2f}"
    )
    table = pd.DataFrame({
        "Time Slot": [f"{h:02d}:00" for h in hours],
        "Program": res["best"],
        "Rating": [ratings[p][i] for i, p in enumerate(res["best"])],
    })
    st.table(table)
    st.line_chart(pd.Series(res["history"], name="Best fitness per generation"))


tab_default, tab1, tab2, tab3 = st.tabs(["Default", "Trial 1", "Trial 2", "Trial 3"])
trial_tabs = [tab1, tab2, tab3]
defaults = [(0.8, 0.05), (0.5, 0.03), (0.95, 0.01)]   # starting values for each trial

for n, tab in enumerate(trial_tabs, start=1):
    with tab:
        d_co, d_mut = defaults[n - 1]
        co_r = st.slider(
            "CO_R (Crossover Rate)", CO_R_MIN, CO_R_MAX, d_co, 0.05, key=f"co{n}"
        )
        mut_r = st.slider(
            "MUT_R (Mutation Rate)", MUT_R_MIN, MUT_R_MAX, d_mut, 0.01, key=f"mut{n}"
        )

        if st.button(f"Run Trial {n}", key=f"run{n}"):
            st.session_state["results"][n] = do_run(
                f"Trial {n}", co_r, mut_r, int(seed_input)
            )

        res = st.session_state["results"].get(n)
        if res:
            show_result(res)

# --- Default tab (values fixed in the code, no slider) ---------------
with tab_default:
    st.write(f"CO_R = {DEFAULT_CO_R}, MUT_R = {DEFAULT_MUT_R} "
             "(default values from the assignment, fixed in the code)")
    if st.button("Run Default", key="run_default"):
        st.session_state["results"]["Default"] = do_run(
            "Default", DEFAULT_CO_R, DEFAULT_MUT_R, int(seed_input)
        )
    if "Default" in st.session_state["results"]:
        show_result(st.session_state["results"]["Default"])

# --- Summary table to compare the trials ----------------------------
if st.session_state["results"]:
    st.subheader("Trial Comparison Summary")
    rows = []
    for key in ["Default", 1, 2, 3]:
        r = st.session_state["results"].get(key)
        if r:
            rows.append({
                "Trial": key, "CO_R": r["co_r"], "MUT_R": r["mut_r"],
                "Total Ratings": round(r["score"], 2),
            })
    st.table(pd.DataFrame(rows))

# --- History of every run + averages --------------------------------
st.subheader("History (all runs)")
if st.session_state["history"]:
    hist = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist, hide_index=True, use_container_width=True)

    st.download_button(
        "Download history (CSV)", hist.to_csv(index=False).encode("utf-8"),
        file_name="ga_history.csv", mime="text/csv",
    )
    st.caption("History is lost when the page is refreshed. Download or copy it first.")
else:
    st.info("No runs yet. Press a Run button to start.")
