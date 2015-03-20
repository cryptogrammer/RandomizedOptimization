package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

/**
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    private static final int numIter = 10;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	double[] rhc_results = new double[numIter];
    	double[] sa_results = new double[numIter];
    	double[] sga_results = new double[numIter];
    	double[] mimic_results = new double[numIter];
        double rhc_avg = 0, sa_avg =0, sga_avg = 0, mimic_avg =0, rhc_stdev = 0,
        		sa_stdev = 0, sga_stdev = 0, mimic_stdev=0;
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        for(int i=0; i<rhc_results.length;++i){
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
	        fit.train();
	        rhc_results[i] = ef.value(rhc.getOptimal());
	        //System.out.println("Randomized Hill Climbing: " + ef.value(rhc.getOptimal()));

	        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
	        fit = new FixedIterationTrainer(sa, 20000);
	        fit.train();
	        sa_results[i] = ef.value(sa.getOptimal());
	        //System.out.println("Simulated Annealing: " + ef.value(sa.getOptimal()));

	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
	        fit = new FixedIterationTrainer(ga, 1000);
	        fit.train();
	        sga_results[i] = ef.value(ga.getOptimal());
	        //System.out.println("Standard Genetic Algorithm: " + ef.value(ga.getOptimal()));

	        // for mimic we use a sort encoding
	        ef = new TravelingSalesmanSortEvaluationFunction(points);
	        int[] ranges = new int[N];
	        Arrays.fill(ranges, N);
	        odd = new  DiscreteUniformDistribution(ranges);
	        Distribution df = new DiscreteDependencyTree(.1, ranges);
	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

	        MIMIC mimic = new MIMIC(500, 100, pop);
	        fit = new FixedIterationTrainer(mimic, 1000);
	        fit.train();
	        mimic_results[i] = ef.value(mimic.getOptimal());
	        //System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
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
       System.out.println("RHC AVG: " + rhc_avg);
       System.out.println("SA AVG: " + sa_avg);
       System.out.println("SGA AVG: " + sga_avg);
       System.out.println("MIMIC AVG: " + mimic_avg);
    }

}
