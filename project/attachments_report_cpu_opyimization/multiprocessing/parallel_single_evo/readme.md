# Parallel computations single evolution
Using `multiprocessing`library to parallelize the computation of acceleration within a single simulation.

* `bash_for_loop.sh`: bash script that iterates over `for_loop_parallel.py` to create data
* `for_loop_parallel.py`: uses multiprocessing to compute accelerations using two nested for loops (direct method). Create csv and diagnostic plots.
* `for_loop_single_evo_mpx.csv`: data obtained with `for_loop_parallel.py`
* `for_single_evo_parallel_compute.pdf`: plot compraing performance of `for_loop_parallel.py`
* `parallel_single_evo_plot.ipynb`: notebook used to creates the plots presented in the report
* `POOL_single_evo_parallel_computation.csv`: data obtained with `vectorized_parallel_evo.py`, using the Pool method to crete workers
* `Pool_vs_ThreadPool.pdf`: plot showing the performance of Pool vs ThreadPool to create workers
* `single_evo_parallel_computation.pdf`: plot showing the performance of `vectorized_parallel_evo.py``
* `THREADPOOL_single_evo_parallel_computation.csv`: data using Threadpool instead of Pool
* `vectorized_parallel_evo.py`: script used to parallelize the computation of acceleration using the vectorized function ( uses numpy broadcasting). Plots diagnostics.

