#!/bin/bash

for n_particles in {1250..2000..250}; do # RICORDATI DI FARE QUELLO PER 100
  for n_simulations in {1..5}; do
    python final_multievo_experiments.py $n_particles $n_simulations 
  done
done
