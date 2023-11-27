# Assignment 2: May the Force...
## Group: I Miloncini

#### Tasks:
- **E)** **Comparison and benchmark**:  
	In `data` directory there's a file, `dt_tot.txt`, containing the parameters and the results of the tests.  
	In the `plots` folder you can find 3 .pdf files:
	-	`plot_dir_model.pdf` contains the benchmark and comparison between the acceleration estimation functions built with the __direct__ method. Data gathered with `pyfalcon` are displayed too. These functions were run for a maximun of 5000 particles, due to the long computational time and expensive computational power. You can see that among the three `acc_dir_*`, `acc_dir_diego` is slightly better than `acc_dir_vepe`. Thus we decided to implement it in the main code as our `acceleration_direct`.
	- `plot_vect_model.pdf` contains the benchmark and comparison between the acceleration estimation functions built in a __vectorized__ fashion. Data gathered with `pyfalcon` are displayed too. This time we run the simulation till a maximum of 50.000 particles, thanks to the computational power of the **Demoblack** server. You can see that among them, the one with the better performance is `acc_onearray_vepe` (in this case slightly better than `acc_vect_diego`). Thus we decided to implement it in the main code as our `acceleration_direct_vectorized`. The function uses the broadcasting operations of `numpy.array`.
	In this function we also implemented the computation of the jerk: the line `jerk_vepe` refers to this case (further considerations below).  
	- `plot_tot.pdf` contains the comparison between all the models we implemented. It's remrkable the increasing in performace (almost 2 orders) relative to the vectorized models. [`pyfalcon`, using C and memory allocation is playing in another league (it's like Pinerolo vs Bayern Monaco)]

