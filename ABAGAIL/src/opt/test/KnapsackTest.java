package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.KnapsackEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
	private static final int numIter = 10;
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME =
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
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
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        for(int i=0;i<rhc_results.length;++i){
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        rhc_results[i] = ef.value(rhc.getOptimal());

        //System.out.println("Randomized Hill Climbing, " + ef.value(rhc.getOptimal()));

        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        sa_results[i] = ef.value(sa.getOptimal());

        //System.out.println("Simulated Annealing, " + ef.value(sa.getOptimal()));

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 180, 25, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        sga_results[i] = ef.value(ga.getOptimal());

        //System.out.println("Standard Genetic Algorithm, " + ef.value(ga.getOptimal()));

        MIMIC mimic = new MIMIC(200, 180, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        mimic_results[i] = ef.value(mimic.getOptimal());

        //System.out.println("MIMIC, " + ef.value(mimic.getOptimal()));
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
