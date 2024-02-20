#!/bin/bash
for i in {1..100};do
    for n_sim in {1..5};do
        for n_particles in {250..2000..250}; do
            python for_mp_njit.py $n_particles $n_sim
        done
    done
done

