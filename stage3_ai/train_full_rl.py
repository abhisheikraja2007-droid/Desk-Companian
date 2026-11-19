from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Callable, Dict, List
import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from stage3_ai.desk_arena import StudyDeskArenaEnv
from stage3_ai.mistake_replay import MistakeReplayBuffer

console = Console()

class MistakeLoggerCallback(BaseCallback):
    def __init__(self, mistake_buffer: MistakeReplayBuffer, check_freq: int = 500, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.mistake_buffer = mistake_buffer
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            summary = self.mistake_buffer.get_mistake_summary()
            if self.verbose > 0:
                console.print(f"[#14B8A6][Callback][/] Step {self.num_timesteps:,} | Replay buffer mistakes: {summary}")
        return True

def make_env_fn(phase: int = 1) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return StudyDeskArenaEnv(curriculum_phase=phase)
    return _init

def train_full_model(total_timesteps: int = 50000, n_envs: int = 4) -> str:
    banner = Panel.fit(
        f"[bold #06B6D4]DESK PET: FULL REINFORCEMENT LEARNING TRAINING (PPO)[/bold #06B6D4]\n"
        f"[dim]Total Timesteps: {total_timesteps:,} | Parallel Envs: {n_envs} | Multi-Threaded Engine[/dim]",
        border_style="#06B6D4"
    )
    console.print(banner)

    models_dir = "stage3_ai/models"
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(models_dir, "desk_pet_ppo")

    mistake_buffer = MistakeReplayBuffer()

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
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )

    mistake_cb = MistakeLoggerCallback(mistake_buffer, check_freq=1000, verbose=1)
    checkpoint_cb = CheckpointCallback(save_freq=max(1000, total_timesteps // 5), save_path=models_dir, name_prefix="desk_pet_ppo")

    console.print(f"[bold #EAB308]Starting full PPO training for {total_timesteps:,} timesteps...[/bold #EAB308]")
    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[mistake_cb, checkpoint_cb])
    elapsed = time.time() - t_start

    final_model_path = os.path.join(models_dir, "desk_pet_ppo_final.zip")
    model.save(final_model_path)
    console.print(f"\n[bold green]✔ Full Model Saved Successfully:[/] {final_model_path}")
    console.print(f"[dim]Total training wall-clock time: {elapsed:.1f}s (Average throughput: {total_timesteps / elapsed:.0f} FPS)[/dim]\n")

    console.print("[bold #06B6D4]Evaluating Trained Policy over 10 Episodes...[/bold #06B6D4]")
    eval_env = StudyDeskArenaEnv(curriculum_phase=2)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    table = Table(title="PPO FULL MODEL EVALUATION SUMMARY", border_style="#14B8A6")
    table.add_column("Evaluation Metric", style="bold white")
    table.add_column("Result Value", style="#EAB308")
    table.add_row("Evaluation Episodes", "10")
    table.add_row("Mean Reward", f"{mean_reward:+.2f}")
    table.add_row("Reward Std Dev", f"{std_reward:.2f}")
    table.add_row("Model Checkpoint", final_model_path)
    table.add_row("Replay Database", "stage3_ai/memory/mistakes.db")
    console.print(table)

    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Full RL Policy for Desk Pet")
    parser.add_argument("--steps", type=int, default=50000, help="Total training timesteps (default: 50,000)")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    args = parser.parse_args()

    train_full_model(total_timesteps=args.steps, n_envs=args.envs)
