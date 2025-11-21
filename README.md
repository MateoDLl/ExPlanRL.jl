# ExPlanRL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MateoDLl.github.io/ExPlanRL.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MateoDLl.github.io/ExPlanRL.jl/dev/)
[![Build Status](https://github.com/MateoDLl/ExPlanRL.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MateoDLl/ExPlanRL.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/MateoDLl/ExPlanRL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MateoDLl/ExPlanRL.jl)

ExPlanRL.jl is a research-oriented Julia package that applies Reinforcement Learning (RL) to the Transmission Network Expansion Planning (TNEP) problem.
The package focuses on training a neural-network-based agent to learn how to make optimal expansion decisions in power systems.

Unlike traditional optimization approaches—such as mathematical programming or metaheuristic algorithms—ExPlanRL.jl explores a novel RL-driven perspective for solving TNEP, where an agent learns decision-making policies directly from simulated interaction with the electrical system.

**Core Idea**

The main goal of the package is to train a policy (a neural network) that learns:

Which lines to build

In what order

Under which conditions

so as to minimize cost, improve reliability, or adapt to system constraints.

The agent is trained on a baseline scenario (simple in composition: only AC lines, no shunt compensation, etc.), and the idea is to later transfer this learned knowledge to more complex scenarios such as:

N−1 contingency settings

Multi-stage expansion planning

Additional operational constraints

This makes ExPlanRL.jl suitable for studying generalization and policy transfer in planning problems.

## Install
```julia
using Pkg
Pkg.add(url="https://github.com/MateoDLl/ExPlanRL.jl")
```