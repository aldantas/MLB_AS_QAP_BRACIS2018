#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "aco.h"

int main(int argc, char *argv[])
{
	int n;
	long cost;
	double best_time, total_time;
	readParameter(argc, argv);

	FILE *file = fopen(input_file, "r");
	if(file == NULL)
		return -1;
	fscanf(file, "%i\n\n", &n);

	// Convert our command line parameters to the ACO framework parameters
	int aco_argc = 9;
	char *aco_argv[9];

	aco_argv[0] = argv[0];

	aco_argv[1] = "-i";
	aco_argv[2] = input_file;

	aco_argv[3] = "--time";
	aco_argv[4] = timeout;

	aco_argv[5] = "--seed";
	aco_argv[6] = seed;
	srand(strtol(seed, NULL, 10));

	aco_argv[7] = "-l";
	aco_argv[8] = ls_choice;

	run_aco(aco_argc, aco_argv, &cost, &best_time, &total_time);

	write_results(cost, best_time, total_time);
	return 0;
}
