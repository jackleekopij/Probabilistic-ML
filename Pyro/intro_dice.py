from __future__ import print_function
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import matplotlib.pyplot as plt



# clear the param store in case we're in a REPL
pyro.clear_param_store()

# Simulate data for rolling of a die where we roll 60 ones, 6 twos, 5 threes, 7 fours, 0 fives and 0 sixes
data = []
for _ in range(6):
    data.append(torch.tensor([1.0,0.0, 0.0,0.0,0.0,0.0]))
for _ in range(6):
    data.append(torch.tensor([0.0,1.0, 0.0,0.0,0.0,0.0]))
for _ in range(5):
    data.append(torch.tensor([0.0,0.0, 1.0,0.0,0.0,0.0]))
for _ in range(7):
    data.append(torch.tensor([0.0,0.0, 0.0,1.0,0.0,0.0]))
for _ in range(7):
    data.append(torch.tensor([0.0,0.0, 0.0,1.0,1.0,0.0]))
for _ in range(7):
    data.append(torch.tensor([0.0,0.0, 0.0,1.0,0.0,1.0]))

# GLOBAL PARAMETERS
NUMBER_OF_FACES = 6


def model(data):
    # lets define the parameters for the 6 sided die.
    # A Dirichlet prior is a standard non-informative prior
    # for a multinomial distribution. Such a prior is useful
    # when there aren't any current beliefs of the about the
    # distrbution of the latent variables.

   f = pyro.sample("latent_fairness", dist.Dirichlet(torch.ones(6)))

   for i in range(len(data)):
       # observe datapoint i using the multinomial likelihood i.e. a die having 6 faces.
        pyro.sample("obs_{}".format(i), dist.Multinomial(probs=f), obs= data[i])


def guide(data):
    '''
    Constructs param 'latent_fairness' which samples from a Dirichlet distribution.
    '''
    # Set a
    alpha_q1 = pyro.param("alpha_q1", torch.tensor(torch.ones(6)),
                         constraint=constraints.positive)
    pyro.sample("latent_fairness",
                dist.Dirichlet(alpha_q1))


# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient step and save losses
losses = [ ]
for step in range(10000):
    losses.append(svi.step(data))
    if step % 100 == 0:
        print('.', end='')

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()

# grab the learned variational parameters
#alpha_q = pyro.param("alpha_q1")[0].item()
#print("The output parameter values for alpha_q: {0}".format(alpha_q))
#beta_q = pyro.param("beta_q").item()
for counter, param in enumerate(pyro.param("alpha_q1")):
    print("The output parameter values for alpha_q_{0}: {1}".format(counter, param))

probability_roll_one = pyro.param("alpha_q1")[0].item() / sum([x.item() for x in  pyro.param("alpha_q1")])


param_norm_cosntant = sum([x.item() for x in  pyro.param("alpha_q1")])
factor = (param_norm_cosntant - pyro.param("alpha_q1")[0].item())/(param_norm_cosntant * (param_norm_cosntant + 1))


print("From the data and the prior beliefs {0} and variance {1}".format(probability_roll_one, probability_roll_one*factor))


# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
#inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
#factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
#inferred_std = inferred_mean * math.sqrt(factor)

#print("\nbased on the data and our prior belief, the fairness " +
#      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))