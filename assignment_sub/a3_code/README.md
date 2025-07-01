# Starter code for HW3

## Installing MuJoCo

For this assignment we will be using the last version of MuJoCo released before
DeepMind acquired and open-sourced it. Follow the instructions in the link below

https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

to download the precompiled binaries for Linux or OSX (Windows is not supported).
After that, make sure to unpack the archive contents into `~/.mujoco/mujoco210`.
Python will look for the MuJoCo binaries in there.

## Environment set up

To set up the environment for this assignment, you will need to create a new
`conda` environment.

    conda create -n cs234_hw3 python=3.8

Once you activate it, run

    pip install -r requirements.txt

## Using `mujoco_py`

Add the following to your `.bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

Launch `python` and try running

```python
import mujoco_py
```

Python interacts with MuJoCo through Cython, and this import should trigger the
compilation process. If you get an error about `c_warning_callback` and
`c_error_callback`, open the `cymj.pyx` file (should be at
`[path to conda folder]/envs/cs234_hw3/lib/python3.8/site-packages/mujoco_py/cymj.pyx`)
and replace the following two lines

```
...
cdef void c_warning_callback(const char *msg) with gil:
...
cdef void c_error_callback(const char *msg) with gil:
...
```

with

```
...
cdef void c_warning_callback(const char *msg) noexcept with gil:
...
cdef void c_error_callback(const char *msg) noexcept with gil:
...
```
On Linux, you may also need to install X11, GLEW, and patchelf.
```
sudo apt install libx11-dev
sudo apt install libglew-dev
pip install patchelf
```

After this, open a new Python interpreter and try importing `mujoco_py` again.
If it works, try running the script below

```
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)

sim.step()

print(sim.data.qpos)
```

If you get the following (or similar) output

```
>>> print(sim.data.qpos)
[0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
>>> sim.step()
>>> print(sim.data.qpos)
[-1.12164337e-05  7.29847036e-22  1.39975300e+00  9.99999999e-01
  1.80085466e-21  4.45933954e-05 -2.70143345e-20  1.30126513e-19
 -4.63561234e-05 -1.88020744e-20 -2.24492958e-06  4.79357124e-05
 -6.38208396e-04 -1.61130312e-03 -1.37554006e-03  5.54173825e-05
 -2.24492958e-06  4.79357124e-05 -6.38208396e-04 -1.61130312e-03
 -1.37554006e-03 -5.54173825e-05 -5.73572648e-05  7.63833991e-05
 -2.12765194e-05  5.73572648e-05 -7.63833991e-05 -2.12765194e-05]
```

you're all set!
