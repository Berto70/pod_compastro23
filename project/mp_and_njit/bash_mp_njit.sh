#!/bin/bash
for n_sim in {1..5};do
    for n_particles in {250..2000..250}; do
        python numba_experiments.py $n_particles $n_sim
    done
done

