<!-- TOC -->

- [A2](#a2)
    - [Q2](#q2)
        - [What Is a Neural Network](#what-is-a-neural-network)
            - [Example: Predicting Weather](#example-predicting-weather)
        - [What is backpropagation](#what-is-backpropagation)
            - [Example](#example)
            - [Example with Numbers (1 Neuron)](#example-with-numbers-1-neuron)
        - [Activation functions](#activation-functions)
        - [What is ReLu](#what-is-relu)
        - [Categorical policy](#categorical-policy)
            - [Example](#example)
        - [Softmax function](#softmax-function)
        - [Gaussian policy](#gaussian-policy)
            - [Example](#example)

<!-- /TOC -->

# A2

## Q2

### What Is a Neural Network

Neural networks are computing systems with interconnected nodes (inspired by human neurons). Can also be thought of as learning algorithms that model the input-output relationship from data.  

A neural network transforms input data by applying a nonlinear function to a weighted sum of the inputs. The transformation is known as a neural layer and the function is referred to as a neural unit ([NVIDIA Developer](https://developer.nvidia.com/discover/artificial-neural-network)).

It’s made up of layers of units ("neurons"):

    Input Layer  →  Hidden Layers  →  Output Layer

Each neuron:

- Takes a bunch of numbers as input;
- Applies a formula to mix them up (with weights and possibly bias);
- Passes the result through an activation function (like ReLU).

#### Example: Predicting Weather

> Suppose we want to predict tomorrow's temperature based on:
> 
> - Today's temperature
> - Wind speed
> - Cloud cover
> 
> Our neural network might look like this:
> 
>     Inputs:      20°C, 10 km/h, 80%
>     ↓
>     Hidden Layer: processes these numbers with weights and ReLU
>     ↓
>     Output:      22°C (prediction)
> 
> Then the neuron:
> 1. Guess  
> The network starts with random weights and makes a prediction (like “Tomorrow will be 25°C”).  
> 2. Check the error  
> Compare the prediction to the correct answer (actual was 22°C). That difference is the error or loss.  
> 3. Adjust weights  
> Use an algorithm called backpropagation with gradient descent to tweak the weights a little to reduce the error. Repeat this process over many examples.

### What is backpropagation

Backpropagation calculates how changes to any of the weights or biases of a neural network will affect the accuracy of model predictions. I.e. "How are the weights and biases of the laters contribute to the overall error". It facilitates the use of gradient descent algorithms to update network weights ([IBM Think](https://www.ibm.com/think/topics/backpropagation)).

#### Example

> Imagine a simple neural network:
> 
>     Input → [w1, w2] → Output
> 
> Let’s say we want the network to learn:
> Input: [2, 3] → Target output: 1
> 
> Initially, the weights w1 and w2 are random. The network produces an output (say 0.6), > and we see that it's wrong. Backpropagation helps figure out:
> 
> “How much did w1 and w2 contribute to the mistake?”
> 
> “How should we tweak w1 and w2 to make the output closer to 1?”
> 
> 1. Forward Pass  
>     Input goes through the network.  
>     Each neuron computes its output:
> 
>         z = w·x + b
>         a = activation(z)
>         
>     The final output is produced.
> 
> 2. Compute Loss
>     Use a loss function to measure how far off the output is:
>         Loss = (target - prediction)^2
> 
> 3. Backward Pass (Backpropagation)
> 
>     Compute the gradient (rate of change) of the loss with respect to each weight:
> 
>         ∂Loss / ∂w
> 
>     Use the chain rule from calculus to break this into small parts:
> 
>         ∂Loss/∂w = ∂Loss/∂output × ∂output/∂activation × ∂activation/∂z × ∂z/∂w
>     Each of these terms is easy to compute if using known activation functions like ReLU or sigmoid.
> 
> 4. Update Weights
> 
>     Update each weight a little bit in the opposite direction of the gradient:
> 
>         w = w - learning_rate × ∂Loss/∂w
> 
>     Repeat over many inputs.

#### Example with Numbers (1 Neuron)

> Setup:
> 
>     Input x = 2
> 
>     Weight w = 0.5
> 
>     Bias b = 0
> 
>     Activation function = identity (for simplicity)
> 
>     Target output = 4
> 
> Forward Pass:
> 
>     z = w·x + b = 0.5 × 2 = 1
>     Output = 1
>     Loss = (4 - 1)^2 = 9
> 
> Backward Pass:
> 
>     dLoss/dOutput = 2 × (1 - 4) = -6
>     dOutput/dw = x = 2
>     dLoss/dw = -6 × 2 = -12
> 
> Weight Update:
> 
>     w = w - lr × dLoss/dw
> 
> Assume lr = 0.01
> 
>     w = 0.5 - 0.01 × (-12) = 0.5 + 0.12 = 0.62
> 
> So after one learning iteration, the weight is nudged in a direction that would have made the output closer to 4.

Note that `lr` stands for learning rate, one of the important hyperparameters in training neural networks. It is basically analogous to the `step size` in various numerical computing and optimization methods.

### Activation functions

An activation function, or transfer function, applies a transformation on weighted input data (matrix multiplication between input data and weights). The function can be either linear or nonlinear.

Usually there is one applied to the output of each neuron. But if it is linear, no matter how many layers are stacked, we'd only be able to model linear relationships ([NVIDIA Developer](https://developer.nvidia.com/discover/artificial-neural-network)).

They have two main uses ([Kaggle](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning)):

- Help a model account for interaction effects.  
What is an interactive effect? It is when one variable A affects a prediction differently depending on the value of B. For example, if my model wanted to know whether a certain body weight indicated an increased risk of diabetes, it would have to know an individual's height. Some bodyweights indicate elevated risks for short people, while indicating good health for tall people. So, the effect of body weight on diabetes risk depends on height, and we would say that weight and height have an interaction effect.

- Help a model account for non-linear effects.  
This just means that if I graph a variable on the horizontal axis, and my predictions on the vertical axis, it isn't a straight line. Or said another way, the effect of increasing the predictor by one is different at different values of that predictor.

    A not-accurate but more interesting analogy:

    - No or linear activation function → boring straight-line thinking.
    - With activation function → the network can think in curves, twists, and jumps — real-world stuff.

### What is ReLu

ReLU, short for Rectified Linear Unit:

    ReLU(x) = max(0, x).

([Kaggle](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning).)

Why Is It So Popular?
- It's fast to compute: no exponentials like in sigmoid or tanh.
- It's easy to differentiate (needed for backpropagation).
- It avoids the "vanishing gradient" problem that made deep networks hard to train before ReLU became popular.

### Categorical policy

Stochastic policy that defines a probability distribution over a discrete set of actions using a categorical distribution. Well-suited for discrete action spaces.

For example:

$$π(a∣s;θ)=Cat(a;softmax(f_θ(s)))$$

where $f_𝜃(s)$ is a function outputting unnormalized logits (one per action), and softmax converts them into probabilities.

About $f_{\theta}(s)$, it is a parametrized function, which can be a neural network. For example, imagine a robot in a grid world with 3 possible actions at each step: `Move Left`, `Move Right`, `Stay Still`. Let’s say the state is just a number: the robot’s x-coordinate on the grid. Then a $f_{\theta}(s)$ might look like

$$f_{\theta}(s) = W \cdot s + b,$$

where $W \in \mathbb{R}^{3 \times 1}$ is a weight matrix, $b \in \mathbb{R}^{3}$ is a bias vector, and $s \in \mathbb{R}^{1}$ is the state (x coord). Then here the $\theta$ is $\theta = [W, b]$.

#### Example

> Suppose an RL agent is in a state s and has 4 possible actions: A1, A2, A3, and A4. A categorical policy might output:
> 
>     π(a|s) = [0.05, 0.2, 0.6, 0.15]
> 
> Then the agent samples from this distribution to decide which action to take.

### Softmax function

The softmax function, also known as softargmax or normalized exponential function, converts a tuple of K real numbers (`logits`) into a probability distribution of K possible outcomes. It is a generalization of the logistic function to multiple dimensions, and is used in multinomial logistic regression ([Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)).

The standard (unit) softmax function

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}},$$

good for modelling a probability distribution.

It is smooth and differentiable, which is essential for gradient-based optimization (like backpropagation).

One intuition is `bigger values become exponentially more dominant`. To have more control, there is the introduction of a "tempratured" version
$$\text{softmax}_T(x_i) = \frac{e^{{x_i/T}}}{\sum_{j=1}^{n} e^{x_j/T}},$$
where the $T$ has effects:
- High $T$ → softer distribution.
- Low $T$ → sharper distribution.
- $T$ → 0: approaches argmax.

🔥 High Temperature ($T≫1$)
- Makes logits closer together → output probabilities are more uniform.
- Encourages exploration — we're more likely to try lesser-valued actions.

❄️ Low Temperature ($T≪1$)
- Sharpens differences → the highest logit dominates.
- Encourages exploitation — we mostly pick the action with the highest score.

### Gaussian policy

Stochastic policy that defines a normal (Gaussian) distribution over actions:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_{\theta}(s), \sigma_{\theta}^2(s)) = \frac{1}{\sqrt{2\pi\sigma_{\theta}^2(s)}} \exp\left(-\frac{(a - \mu_\theta(s))^2}{2\sigma_{\theta}^2(s)}\right).$$

This allows the policy to pick actions near the mean, but still with some stochastic variation for exploration. Continous actions are like steering angles, speed, forces.

#### Example

Imagine a self-driving car needs to decide on steering angle:

- Input: current state $s$, e.g. position, velocity, etc.
- Network outputs:
    - $\mu = 0.2$ radians (turn slightly right),
    - $\sigma = 0.1$.

Then:

$$a \sim \mathcal{N}(0.2, 0.01).$$

So the car will usually turn slightly right, but with some randomness for exploration.