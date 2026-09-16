# Data Fusion Interview Challenge — extracted text (Roshni Sahoo). Math spacing is as extracted from the PDF; equations are restated cleanly in docs/math_fixed.md.

Data Fusion Interview Challenge
Roshni Sahoo
r.sahoo@columbia.edu
Columbia University
1 Statistical Setup
We consider two populations that both evolve over time t = 1, 2, . . . . The population over
individuals in the general US population at time t is denoted by Pt, and the second is the the
population over individuals in the US who respond to a survey at time t is denoted by St.
Let Pt be a joint distribution over an outcome Y ∈ Y ⊂ R and a binary response indicator
R ∈ {0, 1} that corresponds to whether or not an individual responds to the survey. The
outcome Y is a socioeconomic or health indicator, such as an individual’s vaccination status,
their employment status or their living standards. Let St be a distribution over outcomes
conditional on respond to the survey Y | R = 1. This means that Pt is the joint distribution
over (Y, R), and St is the distribution over Y | R = 1. Note that the people who respond to
the survey may not necessarily be representative of the US population, i.e. St ̸= Pt.
We are interested in estimating the average outcome in the general population of the US at a
current time step t. However, we only have access to timely but potentially non-representative
data on the US population from surveys, i.e. we have access to St for only time steps t =
1, 2, . . . M, and representative but lagged data on the general US population, i.e. we have
access to Pt for only for time steps t = 1, 2, . . . m for m < M.
Our estimand of interest is
µ(t) := EPt[Y ] t = m + 1, . . . M.
Note that µ(t) for t = m + 1, . . . M cannot be directly derived from our data because we do not
have access to Pt when t > m.
Define the relative probability of sample selection between two individuals with outcome y
and y
′ at time-step t as
w(y, y′;t) := PPt
[R = 1 | Y = y]
PPt[R = 1 | Y = y
′
]
. (1)
For example, suppose that Y corresponds to vaccination status, a binary variable. Then, we
can omit the dependence of w on y, y′ and define
w(t) := PPt
[R = 1 | Y = 1]
PPt[R = 1 | Y = 0].
When Y corresponds to vaccination status, the quantity w(t) corresponds to how much more
likely a vaccinated individual is to respond to survey than an unvaccinated individual.
We assume that that this relative probability is stable over time.
Assumption 1 (Stationary Selection). The function w(y, y′;t) is constant in t.
In this note, we study to whether µ(t) can be identified, provided that Assumption 1 holds.
1
2 Algorithm
Let T be a random variable that denotes the time step and let Z be a pseudolabel that captures
whether a sample is drawn from the survey or target population. Let Pt,Y denote the marginal
distribution over outcomes Y in Pt. The distribution F is a distribution over (T, Z, Y ), where
Z ∼ Bernoulli(1/2), T ∼ Uniform([m]), FY |T =t,Z=1 = Pt,Y , FY |T =t,Z=0 = St.
As a result, F can be viewed as a pooled distribution over the paired data from St, Pt for
t = 1, 2, . . . m.
Define a loss function
L(θ, a, z) := (1 − z) · exp(θ + a) − z · (θ + a). (2)
We solve the augmented convex optimization problem
{
˜θ(·), α˜(·)} ∈ arg min
(θ,α)∈Θ×A
{EF [L(θ(Y ), α(T), Z)]}, (3)
where Θ denotes a flexible function class (neural networks) and A ⊂ R
m such that A = {α ∈
R
m | α(m) = 0}, where α(t) denotes the t-th index of vector α.
After solving the optimization problem in (3), we can obtain an estimator
µ˜(t) := ESt
h
exp(˜θ(Y )) · Y
i
/ESt
h
exp(˜θ(Y ))i.
3 Task
For t = 1, . . . , m, you observe independent samples from both the population outcome distribution P
Y
t and the survey distribution St. For t = m + 1, . . . , M, you observe samples only
from St. Your goal is to use the historical paired data t ≤ m to learn a selection correction ˜θ
and use it to estimate ˜µ.
It may be helpful to skim Sahoo et al. [2022] Appendix Section B to see an example of a
simple one-dimensional simulation.
Data-Generating Process
1. Let Y be a continuous-valued outcome. Specify distributions for St, Pt for t = 1, 2, . . . M
that match the setting. Note that St, Pt can be defined arbitrarily but Pt should be a
distribution over (Y, R), St should be the Y | R = 1-marginal of Pt, and Pt and St can
change over time but Assumption 1 must hold. Ideally, PPt[R = 1] also changes over
time. Create figures of St, Pt over time.
2. Write down a mathematical specification of the data generating process. How did you
define Pt? How did you define St? How do you simulate drawing samples from St? As
an example, see Eq. 39 and Eq. 40 from Sahoo et al. [2022] Section B.
3. Draw n i.i.d. samples drawn from St for each t = 1, 2, . . . m and n i.i.d. samples drawn
from Pt for each t = 1, 2, . . . m. Create a dataset of i.i.d. samples from F (note that this
dataset should have 2mn samples).
Optimization
1. Implement a Pytorch pipeline for learning ˜θ, α˜ via (3) using your synthetic dataset from
F. Hint: The set up of (θ, α) is closely related to the RU Regression parametrization
from Sahoo et al. [2022].
2
2. Estimate ˜µ(t) for t = m + 1, . . . M using your learned estimate of ˜θ and the samples from
St for t > m.
Conclusions
1. Compare µ(t) to ˜µ(t) and to ESt[Y ] for t = m + 1 . . . M in a figure.
4 Deliverable
Please prepare concise three slides on your results and be prepared to talk through your implementation.
1. The first slide should capture the data-generating process in mathematical notation and
in a figure.
2. The second slide should capture how you implemented solving the optimization problem.
3. The third slide should capture the comparison of µ(t),ESt[Y ] , µ˜(t) and any conclusions
and learnings.
References
Roshni Sahoo, Lihua Lei, and Stefan Wager. Learning from a biased sample. arXiv preprint
arXiv:2209.01754, 2022.
3