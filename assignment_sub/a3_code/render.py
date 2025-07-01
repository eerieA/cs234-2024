import argparse
import pathlib
from typing import Union

import gym
import numpy as np
import stable_baselines3 as sb3
import torch
from moviepy.editor import ImageSequenceClip


def hopper_state_from_observation(env: gym.Env, obs: np.ndarray):
    qpos = env.unwrapped.data.qpos.flat.copy()
    qpos[1:] = obs[:5]
    qvel = obs[5:]

    return qpos, qvel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="The model checkpoint")
    parser.add_argument("--dataset", default=None, help="The preference dataset")
    parser.add_argument(
        "--idx",
        default=None,
        help="When dataset is provided, the index of the demonstration. If none, pick one at random.",
    )
    parser.add_argument("--dpo", action="store_true")
    args = parser.parse_args()

    env = gym.make("Hopper-v3", terminate_when_unhealthy=False)

    if args.checkpoint:
        if args.dpo:
            from run_dpo import SFT, DPO

            with open(args.checkpoint, "rb") as f:
                agent: Union[DPO, SFT] = torch.load(f)

            def policy(obs):
                return agent.act(obs)
        else:
            agent = sb3.PPO.load(args.checkpoint)

            def policy(obs):
                return agent.predict(obs)[0]

        T = env.spec.max_episode_steps
        obs = env.reset()
        images = []
        images.append(env.render(mode="rgb_array"))
        for _ in range(T):
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            images.append(env.render(mode="rgb_array"))
            if done:
                break
        clip = ImageSequenceClip(images, fps=int(1 / env.dt))
        clip.write_videofile(
            pathlib.Path(args.checkpoint).parent.joinpath("video.mp4").as_posix()
        )
    elif args.dataset:
        video_folder = pathlib.Path(args.dataset).parent.joinpath("videos")
        if not video_folder.exists():
            video_folder.mkdir()

        data = np.load(args.dataset, allow_pickle=True)
        if args.idx is None:
            idx = np.random.randint(data["obs_1"].shape[0])
        else:
            idx = int(args.idx)

        if data["label"][idx].item() == 0.5:
            name1 = f"{idx}_equally_preferred1.mp4"
            name2 = f"{idx}_equally_preferred2.mp4"
        else:
            name1 = [
                f"{idx}_preferred.mp4",
                f"{idx}_not_preferred.mp4",
            ][int(data["label"][idx].item())]
            name2 = [
                f"{idx}_preferred.mp4",
                f"{idx}_not_preferred.mp4",
            ][1 - int(data["label"][idx].item())]

        for name, obs in [[name1, data["obs_1"][idx]], [name2, data["obs_2"][idx]]]:
            images = []
            for o in range(obs.shape[0]):
                env.unwrapped.set_state(*hopper_state_from_observation(env, obs[o]))
                images.append(env.render(mode="rgb_array"))
            clip = ImageSequenceClip(images, fps=int(1 / env.dt))
            clip.write_videofile(video_folder.joinpath(name).as_posix())
