The arguments for different parts of the code

1. train_simple :- The baseline 1-hidden layer neural network, saved as 'h20_best_alphabetagamma.pth'.
                    Currently False as the graphs are not that important
2. gen_alg_input :- Genetic algorithm for input space feature selection, results of some experiments are
                    logged in 'log_input_genetic_alg.txt'. Currently False as takes a long time to run.
3. train_chromosome:- train with features corresponding to a chromosome. Needs a string of chromosome.
                    Saved as 'h20_best_chromosome_alphabetagamma.pth'.
4. prune_distinct:- distinctiveness pruning. Needs a string of chromosome for size of the input layer. Saved as 'best_prune_distinct.pth'
5. prune_gen_alg :- prune through genetic algorithm. Needs a string of chromosome. Saved as 'best_prune_gen_alg.pth'.
6. eval_paths :- to run the saved models through validation and test sets.

The chromosomes required for train_chromosome, prune_distinct, prune_gen_alg are from the line 360 of
'log_input_genetic_alg.txt'. Relevant hyper-parameters of genetic algorithm can be found around there.

To get Figure 3 of the paper, uncomment the last line of utils.py