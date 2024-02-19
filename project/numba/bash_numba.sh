#!/bin/bash

for n_particles in {250..2000..250}; do
    python numba_experiments.py $n_particles
done

