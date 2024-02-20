#!bin/bash

for i in {1..100}; do
    for n_particles in {250..2000..250};do
        python for_loop_parallel.py $n_particles
    done
done