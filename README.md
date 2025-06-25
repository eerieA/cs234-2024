# CS234_Reinforcement_Learning (Spring 2024)

<img alt="Course cover image" src="./RL.png" width="300">

This repo contains Stanford CS234 **2024 spring** assignment's coding problems (*unfilled* templates), and some personal notes after watching the free video lectures on Youtube (also for 2024 spring).

If further interested, [this link](https://web.stanford.edu/class/cs234/modules.html) contains entire public-access course materials for the **latest** offering, for example, winter 2025. Due to being different offerings, please be advised that the assignments in this repo may not match the latest course content exactly.

<!-- TOC -->

- [CS234ReinforcementLearning (Spring 2024)](#cs234reinforcementlearning-spring-2024)
        - [Attribution](#attribution)
        - [Disclaimer](#disclaimer)
    - [Official material links](#official-material-links)
    - [Known issues](#known-issues)
    - [Assignment previews](#assignment-previews)
        - [By prev author](#by-prev-author)
            - [A1P4: RiverSwim MDP](#a1p4-riverswim-mdp)
            - [A2P2: Policy Gradient Methods](#a2p2-policy-gradient-methods)
            - [A3P1-P3: Reward engineering, RLHF, DPO](#a3p1-p3-reward-engineering-rlhf-dpo)
        - [By current author](#by-current-author)
            - [A1: RiverSwim](#a1-riverswim)
            - [A2: REINFORCE and PPO](#a2-reinforce-and-ppo)
            - [A3](#a3)
    - [Personal notes](#personal-notes)
        - [Lecture notes](#lecture-notes)
        - [Assignment Notes](#assignment-notes)

<!-- /TOC -->

### Attribution

Based on https://github.com/Rhyme0730/CS234-Reinforcement-Learning . Commits on and before Feb 4, 2025 were all made by the owner of that repo. Also there are *finished* assignment code in that repo. Please consider forking that repo if you don't need to make your own independent commits.

### Disclaimer

The *assignment_sub* folder in this repo contains personal attempts at solving the assignment problems (written as [Quarto](https://quarto.org/) docs), not guarateed to be correct at all.

## Official material links

Also for spring 2024 offering. These may expire without notice ⚠️.

[Lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX) |
[Lecture materials](https://web.stanford.edu/class/cs234/CS234Spr2024/modules.html) |
[Assignment files](https://web.stanford.edu/class/cs234/CS234Spr2024/assignments).

## Known issues

- For A2, the `gym==0.21` in the original requirements.txt may fail the installing because that version is not compatible with some newer setuptools ([../gym/issues/3176](https://github.com/openai/gym/issues/3176)). And even though there is setuptools==65.5.0 in the txt, pip creates an isolated temporary environment to build gym where the setuptools is of another version.  
For me (Python 3.13, setuptools 75.7.0, doing A2 in May 2025), changing it to gym>=0.21,<0.27 worked, but this will install gym 0.26.2 which requires changing some of the template code. So a safer way is to use older Python like `Python 3.11` and just use gym 0.21.
- In A2, running the `plot.py` for Cheetah environment, if seed is not a comma seperated string,  like this:
    ```
    python code\plot.py --env-name cheetah --seeds 1
    ```
    Then this error might occur:  
    ```
    ...RuntimeWarning: Degrees of freedom <= 0 for slice
        ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,...
    ```
    This might be due to how the plot.py reads data? But despite the error, it seems the plot can still be correctly reflecting the data. So we can likely ignore this.  
    Or, if you have time, run all three methods with Cheetah for another seed number, and run plot.py again with 2 seed numbers, the error will disappear.

## Assignment previews

### By prev author

These are preview of results produced by the original repo owner's work on these assignments. Said owner seems to be a PhD who studied [@gatech](https://github.com/gatech) so these are probably very good references.

#### A1_P4: RiverSwim MDP

<img alt="A1 problem 4 figure" src="./A1_code/RiverSwim_MDP.png" width="500px">

#### A2_P2: Policy Gradient Methods

<img alt="A2 problem 2 figure" src="./A2_code/code/results/results-cartpole.png" width="300">

#### A3_P1-P3: Reward engineering, RLHF, DPO

| PPO (without early termination)  | PPO(with early termination) | RLHF |
| ----------- | ----------- | ----------- |
| <img alt="A3 problem 1 to 3 demo 1" src="./A3_code/results/Hopper-v3-early-termination=False-seed=1/video.gif" width="85%"> | <img alt="A3 problem 1 to 3 demo 2" src="./A3_code/results/Hopper-v3-early-termination=True-seed=1/video.gif" width="85%"> | <img alt="A3 problem 1 to 3 demo 3" src="./A3_code/results_rlhf/Hopper-v3-rlhf-seed=0/video.gif" width="100%"> |

---

### By current author

In case anyone wants to compare answers with more people. Again no guarantee on correctness.

#### A1: RiverSwim

This is a screenshot of output after running the filled-out program.

<img alt="A1 coding task result" src="./assignment_sub/previews/a1.jpg" width="300px">

> Written part is at [a1_text.pdf](./assignment_sub/a1_text.pdf).

#### A2: REINFORCE and PPO

| Cartpole  | Pendulum | Cheetah |
| ----------- | ----------- | ----------- |
| <img alt="A2 coding task plot 1" src="./assignment_sub/a2_code/results/results-cartpole.png" width="100%"> | <img alt="A2 coding task plot 2" src="./assignment_sub/a2_code/results/results-pendulum.png" width="100%"> | <img alt="A2 coding task plot 3" src="./assignment_sub/a2_code/results/results-cheetah.png" width="100%"> |

> Written part is at [a2_text.pdf](./assignment_sub/a2_text.pdf).

#### A3

(TBC)

## Personal notes

Made mostly because I did not take the pre-requisite Machine Learning course at the first time of going through this course.

### Lecture notes

[./notes/NOTES_LECTURES.md](./notes/NOTES_LECTURES.md)

Includes:

- Some hand-written process for worked examples mentioned in lectures.
- Some notes on pre-requisite knowledge points, e.g. maximum likelihood estimation (MLE).

### Assignment Notes

[./notes/NOTES_ASGNMTS.md](./notes/NOTES_ASGNMTS.md)

Includes:
- Notes on some pre-requisite knowledge points, e.g. backpropagation, ReLU, etc.