package opt.test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Scanner;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;


/**
*	Implementation of backprop, RHC, SA, and GA to find optimal weights to
*	a neural network classifying heart disease. A lot of the code came
*	from AbaloneTest by Hannah Lau and NNClassificationTest by Andrew Guillory
*
*	@author 	Mohamed El Banani
*	@version 	1.0
*/

public class myTest {

	//
	private static final int NUMTRAINING = 10000;
	private static final int NUMTESTING = NUMTRAINING+1;
	private static final int TOTAL = 20000;

	//get data instances
	private static Instance[] allInstances = initializeInstances();

    //Split instances into a test set and a training set
	private static Instance[] instances = Arrays.copyOfRange(allInstances, 0,NUMTRAINING);
	private static Instance[] testInstances = Arrays.copyOfRange(allInstances, NUMTESTING,TOTAL);

	//neural network properties (output layer corresponds to number of classes)
	private static int inputLayer = 16, hiddenLayer = 10 , outputLayer = 26;

	//training properties
	private static int[] trainingItterations = {10, 50, 100, 250, 500, 1000, 2000};
    private static int[] numRepetitions = {10, 10, 10};
    private static int[] maxEpochs = {10, 50, 100, 200, 500, 1000, 2000, 5000};
    private static double learningRate = 0.5;
    private static double momentum = 0.2;
    private static ErrorMeasure measure = new SumOfSquaresError();

	//Results to report (training time and evaluation)
    private static double[][] time = new double[3][];
    private static double[][] eval = new double[3][];


    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static DataSet set = new DataSet(instances);
    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA ", "GA "};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int numRuns = trainingItterations.length;
        for (int z = 0; z < numRuns; z++){
            String header = "";
            results = "";
            header = "-----------------------------------------------------------\n";
            header += "-----------------------------------------------------------\n";
            header += "myTest \n--------- \n\nDataset: Alphabet Identification\n";
            header += "Input Layer: " + inputLayer+ "\nHidden Layer: " + hiddenLayer;
            header += "\nOutput Layer: " + outputLayer+ "\n\n";
            header += "Number of Itterations: (itterationsXrepititions) \n";
            header += "Randomized Hill Climbing: " + trainingItterations[z] +"x"+ numRepetitions[0];
            header += "\n";
            header += "Simulated Annealing:      " + trainingItterations[z] +"x" + numRepetitions[1];
            header += "\n";
            header += "Genetic Algorithm:        " + trainingItterations[z] +"x" + numRepetitions[2];
            header += "\n";
            header += "BackProp-RPROP (max):     " + maxEpochs[z] +"x" + numRepetitions[2];
            header += "\n";
            System.out.println(header);


            int[] numEval = new int[3];
            for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {

            double tCorrect = 0, tIncorrect = 0, correct = 0, incorrect = 0;
            double start =0 , end=0, trainingTime=0, testingTime=0;
            int[][] confusion, confTrain;

            confusion = new int[outputLayer][outputLayer];
            confTrain = new int[outputLayer][outputLayer];
            start = System.nanoTime();
            System.out.print("Running: ");
            System.out.print(oaNames[i]);

            for (int k = 0; k < numRepetitions[i]; k++) {
                    int loadingFactor = numRepetitions[i]/10;
                    if (((k+1)%loadingFactor) == 0) {
                        System.out.print("*");
                    }
                    start = System.nanoTime();
                    train(oa[i], networks[i], oaNames[i], z); //trainer.train();
                    end = System.nanoTime();
                    trainingTime += end - start;

                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());

                    int predicted, actual;
                    start = System.nanoTime();
                    for(int j = 0; j < testInstances.length; j++) {
                        networks[i].setInputValues(testInstances[j].getData());
                        networks[i].run();

                        actual = testInstances[j].getLabel().getData().argMax();
                        predicted = networks[i].getOutputValues().argMax();
                        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                        confusion[actual][predicted]++;
                    }
                    for(int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();
                        actual = instances[j].getLabel().getData().argMax();
                        predicted = networks[i].getOutputValues().argMax();
                        double trash = Math.abs(predicted - actual) < 0.5 ? tCorrect++ : tIncorrect++;
                        confTrain[actual][predicted]++;
                    }
                    end = System.nanoTime();
                    testingTime += end - start;
                }
                trainingTime /=10E6;
                testingTime /= 10E6;
                trainingTime /= numRepetitions[i];
                testingTime /= numRepetitions[i];


                results +=  "\nResults for " + oaNames[i] + ": \n\n"  + "Average Training Time (millisec): " + trainingTime + "\nAverage Testing Time (millisec): " + testingTime+ "\nTesting Accuracy:\nPercent correctly classified: " + df.format(correct/(correct+incorrect)*100);

                results += "\n\nTesting Confusion Matrix:\n";
                results += printConfusion(confusion);

                results += "\n\nTraining Accuracy:\nPercent correctly classified: " + df.format(tCorrect/(tCorrect+tIncorrect)*100);
                results += "\nTraining Confusion Matrix: \n";
                results += printConfusion(confTrain);

                System.out.println();
            }

                     try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("src/opt/test/newResults.txt", true)))) {
                out.println(header);
                out.println(results);
            }catch (IOException e) {
                    //exception handling left as an exercise for the reader
            }
        }
    }



    private static Instance[] initializeInstances() {

//        double[][][] attributes = new double[297][][];
//
//        try {
//            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/letterData.txt")));
//
//            for(int i = 0; i < attributes.length; i++) {
//                Scanner scan = new Scanner(br.readLine());
//                scan.useDelimiter(",");
//
//                attributes[i] = new double[2][];
//                    attributes[i][0] = new double[13]; // 7 attributes
//                    attributes[i][1] = new double[1];
//
//                    for(int j = 0; j < 13; j++)
//                        attributes[i][0][j] = Double.parseDouble(scan.next());
//                    attributes[i][1][0] = Double.parseDouble(scan.next());
//                }
//        }catch(Exception e) {
//                e.printStackTrace();
//        }
        double[][][] attributes = new double[20000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/letterData.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[16]; // 16 attributes
                attributes[i][1] = new double[1];
                String tempLabel = scan.next();
                char tempCharLabel = tempLabel.charAt(0);
                attributes[i][1][0] = Double.parseDouble(""+ ((tempCharLabel)-65));
                for(int j = 0; j < 16; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            double classification = attributes[i][1][0];
            double[] label =  new double[26];
            for(int m=0;m<label.length;++m) label[m] = 0;
            label[(int)classification] = 1;
            instances[i].setLabel(new Instance(label));
        }

        return instances;
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int index) {
        for(int i = 0; i < trainingItterations[index]; i++) {
            oa.train();
            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel();
                Instance example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));
                error += measure.value(output, example);
            }
        }
    }

    private static String printConfusion(int[][] conf) {
        String out = "";
    	for(int i = 0; i < conf.length; i++) {
        	for(int j = 0; j < conf[0].length; j++) {
        		if (i == j) {
    				out += "(";
    			}
        		out  += conf[i][j];
        		if (i == j) {
        			out += ")";
                }
                out  += ",  ";
            }
            out += "\n";
        }
        return out;
    }

}