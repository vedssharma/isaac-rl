"""Publish the project's W&B report (PLAN.md Phase 5): the benchmark, training curves and links.

    python scripts/wandb_report.py [--draft]

Uses the policies and seeds in scripts/benchmark_table.py, so the report shows the runs behind docs/benchmark.md.
Needs WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT (and WANDB_BASE_URL for a dedicated server) and
`pip install wandb-workspaces`. Prints the report's URL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_table import CONDITIONS, POLICIES  # noqa: E402

GITHUB = "https://github.com/vedssharma/isaac-rl"
HF = "https://huggingface.co/vedssharma/isaac-rl-drone-nav"


def benchmark_markdown() -> str:
    table = (REPO_ROOT / "docs" / "benchmark.md").read_text()
    start = table.index("## Success (%)")
    end = table.index("## Collisions (%)")
    return table[start:end].replace("## Success (%)", "**Success (%)**, mean over 3 seeds ± 95% CI").strip()


def build(entity: str, project: str):
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces.expr import Metric

    def runset(label: str, runs: dict[str, str]) -> wr.Runset:
        return wr.Runset(entity=entity, project=project, name=label, filters=[Metric("ID").isin(list(runs.values()))])

    all_sets = [runset(label, runs) for label, _, runs in POLICIES]
    bench = [
        wr.BarPlot(title=f"{title}: success", metrics=[f"{key}/Metrics/success_rate"], range_x=(0, 1),
                   layout=wr.Layout(x=8 * (i % 3), y=6 * (i // 3), w=8, h=6))
        for i, (key, title) in enumerate(CONDITIONS)
    ]  # fmt: skip
    teacher_panels = [
        wr.LinePlot(title="Teacher: success rate while training (curriculum levels mixed)",
                    y=["Metrics/success_rate"], smoothing_factor=0.6, layout=wr.Layout(x=0, y=0, w=12, h=7)),
        wr.LinePlot(title="Teacher: mean curriculum level (0-9)", y=["Curriculum/terrain_level"],
                    layout=wr.Layout(x=12, y=0, w=12, h=7)),
    ]  # fmt: skip
    student_panels = [
        wr.LinePlot(title="Students: success rate while distilling (early actions are partly the teacher's)",
                    y=["Episode/Metrics/success_rate"], smoothing_factor=0.6, layout=wr.Layout(x=0, y=0, w=12, h=7)),
        wr.LinePlot(title="Students: imitation loss", y=["Loss/imitation"], log_y=True, smoothing_factor=0.6,
                    layout=wr.Layout(x=12, y=0, w=12, h=7)),
        wr.LinePlot(title="DAgger: fraction of actions taken by the teacher", y=["Distill/teacher_fraction"],
                    layout=wr.Layout(x=0, y=7, w=12, h=6)),
        wr.LinePlot(title="Students: travel direction inside the camera's view", y=["Episode/Metrics/travel_in_camera_fov"],
                    smoothing_factor=0.6, layout=wr.Layout(x=12, y=7, w=12, h=6)),
    ]  # fmt: skip

    blocks = [
        wr.P(
            "A Crazyflie-sized quadrotor learns, entirely in simulation (Isaac Sim / Isaac Lab), to fly to a goal "
            "20-24 m away through randomized clutter (cylinders, boxes, thin poles, walls with doors) from a 64 x 48 "
            "depth camera and its own motion sensors. A privileged teacher is trained with PPO, then distilled into a "
            "depth-only CNN + GRU student with DAgger, then made robust with domain randomization."
        ),
        wr.P([
            "On held-out level-5 maps the depth-only GRU student reaches the goal in 94.5% ± 0.7 of episodes "
            "(3 seeds; the goal was 85%). Trained under randomization, it keeps 81.9% with mass, thrust, motor lag, "
            "latency, wind and sensor noise all randomized at once, where the unrandomized student falls to 54%. "
            "Code and notes: ", wr.Link("GitHub", GITHUB), "; checkpoints and model card: ", wr.Link("Hugging Face", HF),
            ".",
        ]),  # fmt: skip
        wr.TableOfContents(),
        wr.H1("The policy flying"),
        wr.Image(url=f"{HF}/resolve/main/media/p5-final-l5.png",
                 caption="The randomized GRU student on a held-out level-5 map, one frame every 2 s. "
                         "Top right: the depth image it flies from (near = bright)."),  # fmt: skip
        wr.P([
            "Videos (30 s each, chase camera with the depth image inset): ",
            wr.Link("nominal", f"{HF}/blob/main/media/p5-final-l5-depth-pip.mp4"), ", ",
            wr.Link("full randomization", f"{HF}/blob/main/media/p5-final-l5-randomized-depth-pip.mp4"), ".",
        ]),  # fmt: skip
        wr.H1("Benchmark"),
        wr.P(
            "Held-out maps (test seed 1001, 60 layouts per level never seen in training), 1000 episodes per cell and "
            "seed. 'L5 randomized': level 5 with every Phase 4 randomization on at once, and depth noise x 2 for the "
            "depth policies."
        ),
        wr.MarkdownBlock(benchmark_markdown()),
        wr.P("Per run (3 seeds per policy; the bars are success fractions, 1 = 100%):"),
        wr.PanelGrid(runsets=all_sets, panels=bench),
        wr.UnorderedList([
            "The depth students match the privileged teacher at levels 1-5 and beat it at 7-9, although they see 90 "
            "degrees ahead instead of 360 around.",
            "Memory makes the student consistent: ± 0.7 at level 5 with a GRU, ± 6.3 without.",
            "Training under randomization is what buys robustness (54% -> 82% under full randomization), at a cost of "
            "1-2 points nominally.",
            "Level 9 (0.6 m gaps, 12 obstacles per 100 m2) is not solved, and thin poles cause most collisions.",
        ]),  # fmt: skip
        wr.H1("Training"),
        wr.H2("Teacher (PPO, privileged)"),
        wr.PanelGrid(runsets=[all_sets[0]], panels=teacher_panels),
        wr.H2("Depth students (DAgger)"),
        wr.PanelGrid(runsets=all_sets[1:], panels=student_panels),
        wr.H1("Lessons"),
        wr.UnorderedList([
            "A privileged teacher must be imitable from the student's viewpoint. The first teacher flew tail-first "
            "using its 360-degree sensing, and its forward-camera student scored 3.6%. A reward for facing the "
            "direction of travel fixed it (3.6% -> 94%).",
            "Students inherit their teacher's habits seed by seed (climbing over walls, stalling), so most student "
            "problems were teacher problems.",
            "Domain randomization from scratch failed; fine-tuning the nominal teacher under it worked.",
            "Relative metrics can mislead: a stronger-wind variant halved the 'drop under randomization' mostly by "
            "getting worse in calm air.",
        ]),  # fmt: skip
        wr.P(["Everything is in simulation only; nothing has flown a real drone. Full notes per phase: ",
              wr.Link("GitHub", GITHUB), "."]),  # fmt: skip
    ]
    return wr.Report(
        entity=entity,
        project=project,
        title="isaac-rl: a quadrotor that flies through clutter from depth alone",
        description="Privileged PPO teacher -> DAgger depth student -> domain randomization; benchmark on held-out maps.",
        blocks=blocks,
        width="fluid",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draft", action="store_true", help="save as a draft instead of publishing")
    args = parser.parse_args()
    report = build(os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"])
    report.save(draft=args.draft)
    print(report.url)


if __name__ == "__main__":
    main()
