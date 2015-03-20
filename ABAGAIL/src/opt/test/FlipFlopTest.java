package opt.test;

import java.util.Arrays;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
	private static final int numIter = 10;

    /** The n value */
    private static final int N = 80;

    private static final int T = N/10;

    public static void main(String[] args) {
    	double[] rhc_results = new double[numIter];
    	double[] sa_results = new double[numIter];
    	double[] sga_results = new double[numIter];
    	double[] mimic_results = new double[numIter];
        double rhc_avg = 0, sa_avg =0, sga_avg = 0, mimic_avg =0, rhc_stdev = 0,
        		sa_stdev = 0, sga_stdev = 0, mimic_stdev=0;
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        for(int i=0;i<rhc_results.length;++i){

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        rhc_results[i] = ef.value(rhc.getOptimal());


        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        sa_results[i] = ef.value(sa.getOptimal());


        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        sga_results[i] = ef.value(sa.getOptimal());

        MIMIC mimic = new MIMIC(200, 5, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        mimic_results[i] = ef.value(mimic.getOptimal());
    }
    for(int k = 0; k < rhc_results.length;++k){
    	rhc_avg += rhc_results[k];
    	sa_avg += sa_results[k];
    	sga_avg += sga_results[k];
    	mimic_avg += mimic_results[k];
    }
    rhc_avg /= rhc_results.length;
    sa_avg /= sa_results.length;
    sga_avg /= sga_results.length;
    mimic_avg /= mimic_results.length;

    // Calculate Standard Deviations

   for(int i=0;i<rhc_results.length;++i){
	   rhc_stdev += Math.pow(rhc_results[i]-rhc_avg,2);
	   sa_stdev += Math.pow(sa_results[i]-sa_avg,2);
	   sga_stdev += Math.pow(sga_results[i]-sga_avg,2);
	   mimic_stdev += Math.pow(mimic_results[i]-mimic_avg,2);
   }
   rhc_stdev = Math.sqrt(rhc_stdev/rhc_results.length);
   sa_stdev = Math.sqrt(sa_stdev/sa_results.length);
   sga_stdev = Math.sqrt(sga_stdev/sga_results.length);
   mimic_stdev = Math.sqrt(mimic_stdev/mimic_results.length);

   // printing the results
   System.out.print("RHC,");
   for(int i=0;i<rhc_results.length;++i){
	   System.out.print(rhc_results[i]+",");
   }
   System.out.println();
   System.out.print("SA,");
   for(int i=0;i<rhc_results.length;++i){
	   System.out.print(sa_results[i]+",");
   }
   System.out.println();
   System.out.print("SGA,");
   for(int i=0;i<rhc_results.length;++i){
	   System.out.print(sga_results[i]+",");
   }
   System.out.println();
   System.out.print("MIMIC,");
   for(int i=0;i<rhc_results.length;++i){
	   System.out.print(mimic_results[i]+",");
   }
   System.out.println();
   //System.out.println("RHC STDEV: " + rhc_stdev);
   //System.out.println("SA STDEV: " + sa_stdev);
   //System.out.println("SGA STDEV: " + sga_stdev);
   //System.out.println("MIMIC STDEV: " + mimic_stdev);
}
}
