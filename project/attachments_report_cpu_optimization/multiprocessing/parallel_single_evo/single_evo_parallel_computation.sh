#!/bin/bash

for i in {1..100}; do
  for n_particles in {100..5000..250}; do # RICORDATI DI FARE QUELLO PER 100
    python vectorized_parallel_evo.py $n_particles
  done
done