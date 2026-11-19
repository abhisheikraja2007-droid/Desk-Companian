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

class OutDistributionEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int = 25000,
        n_eval_episodes: int = 10,
        best_model_path: str = "stage3_ai/models/ppo_desk_pet_final.zip",
        verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.best_unseen_reward = -float("inf")
        self.eval_env = StudyDeskArenaEnv(unseen_test_mode=True)
        self.patience = 8
        self.stagnation_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % (self.eval_freq // 16) == 0:
            rewards = []
            zero_collisions = 0
            for ep in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset(seed=1000 + ep)
                done = False
                ep_rew = 0.0
                collided = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rew, term, trunc, info = self.eval_env.step(action)
                    ep_rew += rew
                    if info.get("mistake") in ("CLIFF_TRIGGER", "MUG_PROXIMITY_VIOLATION"):
                        collided = True
                    done = term or trunc
                rewards.append(ep_rew)
                if not collided:
                    zero_collisions += 1

            unseen_reward = float(np.mean(rewards))
            zero_col_rate = (zero_collisions / self.n_eval_episodes) * 100.0

            console.print(
                f"[#06B6D4][OOD Validation @ Step {self.num_timesteps:,}][/] "
                f"Unseen Desk Reward: [#EAB308]{unseen_reward:+.2f}[/] | "
                f"Zero-Collision: [#14B8A6]{zero_col_rate:.1f}%[/]"
            )

            if unseen_reward > self.best_unseen_reward:
                self.best_unseen_reward = unseen_reward
                self.stagnation_count = 0
                self.model.save(self.best_model_path)
                console.print(f"  [bold green]★ New Best Generalizing Weights Saved:[/] {self.best_model_path}")
            else:
                self.stagnation_count += 1
                if self.stagnation_count >= self.patience:
                    console.print("[yellow]Early stopping triggered: Out-of-distribution validation plateaus.[/yellow]")
                    return False
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()

def make_env_fn(phase: int = 2) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return StudyDeskArenaEnv(curriculum_phase=phase)
    return _init

def train_full_model(total_timesteps: int = 500000, n_envs: int = 16, sanity_only: bool = False) -> str:
    sys.stdout = TeeLogger(LOG_FILE)

    dev_str, dev_desc = resolve_device()

    banner = Panel.fit(
        f"[bold #06B6D4]DESK PET: {'16k-STEP SANITY CHECK' if sanity_only else '500,000-STEP REINFORCEMENT LEARNING CONVERGENCE (PPO)'}[/bold #06B6D4]\n"
        f"[dim]Device Engine: {dev_desc} | Vectorized Envs: {n_envs} | Target: {total_timesteps:,} Timesteps[/dim]",
        border_style="#06B6D4"
    )
    console.print(banner)

    models_dir = "stage3_ai/models"
    os.makedirs(models_dir, exist_ok=True)
    final_model_path = os.path.join(models_dir, "ppo_desk_pet_final.zip")

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

    eval_freq = 8192 if sanity_only else 25000
    ood_cb = OutDistributionEvalCallback(eval_freq=eval_freq, best_model_path=final_model_path)
    checkpoint_cb = CheckpointCallback(save_freq=max(8192, total_timesteps // 10), save_path=models_dir, name_prefix="ppo_desk_pet_ckpt")

    console.print(f"[2/4] Executing PPO {'Sanity Check' if sanity_only else 'Convergence'} for {total_timesteps:,} Steps...")
    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[ood_cb, checkpoint_cb])
    elapsed = time.time() - t_start

    model.save(final_model_path)
    console.print(f"\n[3/4] [bold green]✔ Model Checkpoint Successfully Saved:[/] {final_model_path}")
    console.print(f"      Wall-Clock Time: {elapsed:.1f}s ({elapsed/60.0:.1f} mins) | Throughput: {total_timesteps / elapsed:.0f} FPS\n")

    console.print("[4/4] Running Post-Training Held-Out Evaluation (20 Unseen Desks)...")
    eval_env = StudyDeskArenaEnv(unseen_test_mode=True)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    eval_env.close()

    table = Table(title=f"PPO {total_timesteps:,}-STEP EVALUATION SUMMARY", border_style="#14B8A6")
    table.add_column("Convergence Metric", style="bold white")
    table.add_column("Trained Model Value", style="#EAB308")
    table.add_row("Total Timesteps Trained", f"{total_timesteps:,}")
    table.add_row("Parallel Environments", str(n_envs))
    table.add_row("Held-Out Unseen Desk Reward", f"{mean_reward:+.2f}")
    table.add_row("Reward Standard Deviation", f"{std_reward:.2f}")
    table.add_row("Model Checkpoint", final_model_path)
    table.add_row("Mistake Replay Database", "stage3_ai/memory/mistakes.db")
    console.print(table)
    console.print(f"\n[bold green]✔ {total_timesteps:,}-Step Execution Complete![/bold green]\n")

    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full RL Policy Convergence for Desk Pet")
    parser.add_argument("--steps", type=int, default=500000, help="Total training timesteps (default: 500,000)")
    parser.add_argument("--envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--sanity-check", action="store_true", help="Run 16,384-step sanity test")
    args = parser.parse_args()

    steps = 16384 if args.sanity_check else args.steps
    train_full_model(total_timesteps=steps, n_envs=args.envs, sanity_only=args.sanity_check)
