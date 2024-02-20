#!/bin/bash
for i in {1..100}; do
  for n_particles in {250..2000..250}; do 
    for n_simulations in {1..5}; do
      python final_multievo_experiments.py $n_particles $n_simulations 
    done
  done
done
