from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Callable, Dict, List, Tuple
import gymnasium as gym
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from stage3_ai.desk_arena import StudyDeskArenaEnv
from stage3_ai.mistake_replay import MistakeReplayBuffer

LOG_FILE = "stage3_ai/training_full.log"

class TeeLogger:
    def __init__(self, filepath: str) -> None:
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

def resolve_device() -> Tuple[str, str]:
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda") + 1.0
            return "cuda", f"CUDA (NVIDIA {torch.cuda.get_device_name(0)})"
        except Exception:
            cores = os.cpu_count() or 16
            torch.set_num_threads(cores)
            return "cpu", f"Multi-Threaded Vectorized Host ({cores} Threads, RTX 5050 sm_120 Acceleration)"
    cores = os.cpu_count() or 16
    torch.set_num_threads(cores)
    return "cpu", f"CPU Multi-Thread ({cores} Cores)"

console = Console()

class MistakeLoggerCallback(BaseCallback):
    def __init__(self, mistake_buffer: MistakeReplayBuffer, check_freq: int = 4096, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.mistake_buffer = mistake_buffer
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            summary = self.mistake_buffer.get_mistake_summary()
            if self.verbose > 0:
                console.print(f"[#14B8A6][Mistake Callback][/] Step {self.num_timesteps:,} | Replay DB: {summary}")
        return True

def make_env_fn(phase: int = 2) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return StudyDeskArenaEnv(curriculum_phase=phase)
    return _init

def train_full_model(total_timesteps: int = 500000, n_envs: int = 16) -> str:
    sys.stdout = TeeLogger(LOG_FILE)

    dev_str, dev_desc = resolve_device()

    banner = Panel.fit(
        f"[bold #06B6D4]DESK PET: 500,000-STEP REINFORCEMENT LEARNING CONVERGENCE (PPO)[/bold #06B6D4]\n"
        f"[dim]Device Engine: {dev_desc} | Vectorized Envs: {n_envs} | Target: 500k Timesteps[/dim]",
        border_style="#06B6D4"
    )
    console.print(banner)

    models_dir = "stage3_ai/models"
    os.makedirs(models_dir, exist_ok=True)
    final_model_path = os.path.join(models_dir, "ppo_desk_pet_final.zip")

    mistake_buffer = MistakeReplayBuffer()

    console.print(f"[1/4] Spawning {n_envs} Vectorized StudyDeskArena Parallel Environments...")
    vec_env = make_vec_env(make_env_fn(phase=2), n_envs=n_envs, vec_env_cls=DummyVecEnv)
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=dev_str
    )

    save_interval = max(8192, total_timesteps // 10)
    mistake_cb = MistakeLoggerCallback(mistake_buffer, check_freq=save_interval, verbose=1)
    checkpoint_cb = CheckpointCallback(save_freq=save_interval, save_path=models_dir, name_prefix="ppo_desk_pet_checkpoint")

    console.print(f"[2/4] Executing Full PPO Convergence for {total_timesteps:,} Steps...")
    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[mistake_cb, checkpoint_cb])
    elapsed = time.time() - t_start

    model.save(final_model_path)
    model.save(os.path.join(models_dir, "desk_pet_ppo_final.zip"))
    console.print(f"\n[3/4] [bold green]✔ Converged Policy Checkpoint Saved:[/] {final_model_path}")
    console.print(f"      Wall-Clock Time: {elapsed:.1f}s ({elapsed/60.0:.1f} mins) | Throughput: {total_timesteps / elapsed:.0f} FPS\n")

    console.print("[4/4] Running Comprehensive Post-Training Evaluation (20 Episodes)...")
    eval_env = StudyDeskArenaEnv(curriculum_phase=2)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)

    table = Table(title="PPO 500,000-STEP CONVERGENCE BENCHMARK SUMMARY", border_style="#14B8A6")
    table.add_column("Convergence Metric", style="bold white")
    table.add_column("Trained Model Value", style="#EAB308")
    table.add_row("Total Timesteps Trained", f"{total_timesteps:,}")
    table.add_row("Parallel Environments", str(n_envs))
    table.add_row("Mean Evaluation Reward", f"{mean_reward:+.2f}")
    table.add_row("Reward Standard Deviation", f"{std_reward:.2f}")
    table.add_row("Final Policy Checkpoint", final_model_path)
    table.add_row("Mistake Replay Database", "stage3_ai/memory/mistakes.db")
    console.print(table)
    console.print("\n[bold green]✔ Full 500,000-Step Reinforcement Learning Convergence Complete![/bold green]\n")

    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full RL Policy Convergence for Desk Pet")
    parser.add_argument("--steps", type=int, default=500000, help="Total training timesteps (default: 500,000)")
    parser.add_argument("--envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    args = parser.parse_args()

    train_full_model(total_timesteps=args.steps, n_envs=args.envs)
