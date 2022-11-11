# Project Resilience Data Requirements and Tips

## Format and Features

Data features in each row of the data should include columns that can be cast
as **Context**, **Actions** and **Outcomes** of a decision pertaining to the
unit of decision-making.

For example, if the problem is carbon emissions decisions per power plant:
- the unit of decision making is a power plant, so each row should represent 
a decision made for a power plant
- Context features are features about the plant that can't be changed
(e.g., location, weather, reactor type)
- Actions are policies for the plant that can be changed within reasonable
time so that the effect can be observed and associated to the action
(e.g., generator setup config, carbon capture level, change in generation
hours)
- Outcomes are quantifiable values that can be attributed to a single region
within a reasonable lag (e.g., carbon emissions, cost of actions, energy
- output)

## Predictability

We need some a priori theory of why/how Actions could affect Outcomes,
and why we should expect prediction of Outcomes to be easier from
Context/Actions rather than from a Context alone. A human being should,
just by looking at the context / action data, be able to predict more or less
what the outcome should be. At least be able to reason about it.
Alternatively, a basic predictor model mapping Context/Actions to Outcomes
should be able to show that it uses the Actions to make predictions better
than with Context alone. This simple predictor model does not need to use
the full data or input/output spaces, it just needs to make it clear that
there's something there.

## Rules of Thumb

### Time-series

Either (1) we have an outcome value at each time step, in which case the row
should indicate the time step; or (2) we have an outcome value only at
particular time steps (e.g., if we have daily power plant CO2 output,
but only monthly cost reports). In any case, if there are time steps which
are missing some values (context, action, or outcome), it's ok if they are
NA in the row for that time step: we can still construct time series to train
on from this dataset.

### Missing Data

To give the project the best chance of success, the amount of missing data
should be minimal and/or structured, e.g., we only get cost reports monthly.

### Data Sufficiency

Data rows should cover variations of decisions sufficiently, and so, in the
case of time-series data, we need historical decision instances that include
different actions taken for similar context.

A single row should represent one observation, which includes context, 
actions, outcomes for that observation.

We need enough cases for our predictor to learn something about how Actions
affect Outcomes. If we have thousands of samples to begin with, that certainly
gives us a better shot. A quick-and-dirty check for correlations between
actions and outcomes could be used as a gating function, i.e., the
correlation matrix should not look like noise. If it looks like noise,
the project may be possible but hard.

Data requirement grows exponentially with number of outcome objectives.

### Consistency

Same context and actions should result in similar outcomes. Contradicting
samples should be minimal. In other words, not too many rows with same
Context and Actions resulting in different Outcomes.

It should be possible to observe the outcome of an action in a reasonable
amount of time (e.g., less than 3 months)

### Availability and Updates to the Data

As a rule of thumb, the number of new samples should be at least on the order
of the problem dimension, or (dim(A) + dim(C)) x (dim(O)). More important
than the number of new samples is which data is sampled: one sample in a
previously-unknown region of interest may be more useful than thousands
in a region we already know well or don't care about. So, if we control
which data is sampled, we don't need as much of it.

### Transparency/Accountability

Data should come from reliable, trusted, scientific, ethical sources.
(e.g. not blackboxes or your mom's Facebook surveys).
