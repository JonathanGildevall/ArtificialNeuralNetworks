package Homework2;

import java.io.*;
import java.sql.SQLOutput;
import java.util.Arrays;

public class Main {
    static double outputThreshold;
    static int firstLayerSize = 3;
    static int secondLayerSize = 3;

    public static void main(String[] args) {
        outputThreshold = (Math.random() * 2) - 1;
        //Threshold
        double[][] firstLayer = new double[10000][firstLayerSize];
        double[][] secondLayer = new double[10000][secondLayerSize];
        double[] trainingOutput = new double[10000];

        double[][] thresholds = new double[2][3];
        double[][] trainingSet = new double[10000][2];
        double[][] validationSet = new double[5000][2];
        double[] trainingTarget = new double[10000];
        double[] target = new double[5000];
        readSet("training_set.csv", trainingSet, validationSet, trainingTarget, target);
        readSet("validation_set.csv", trainingSet, validationSet, trainingTarget, target);
        double eta = 0.02;

        double[][] firstWeights = generateWeights(firstLayerSize, 2);
        double[][] middleWeights = generateWeights(secondLayerSize, firstLayerSize);
        double[] outputWeight = generateOutputWeights();

        /*
        for (int i = 0; i < trainingSet.length; i++) {
            //Break thresholds from same array!
            updateLayer(thresholds[0],firstWeights,trainingSet[i], firstLayer[i]);
            updateLayer(thresholds[1],middleWeights,trainingSet[i], secondLayer[i]);
        }
        */
        int i = 0;

        while (i % 10000 != 0 || classificationError(generateOutput(outputWeight,validationSet,thresholds,firstWeights,middleWeights), target) >= 0.12) {
            i++;
            int my = (int) (Math.random() * trainingSet.length);
            long start = System.nanoTime();
            propigateForward(outputThreshold,outputWeight,trainingSet,thresholds,firstWeights,middleWeights, my, firstLayer, secondLayer, trainingOutput);
            long end = System.nanoTime();
            System.out.println(end-start);

            start = System.nanoTime();
            updateWeights(trainingTarget, trainingOutput, outputWeight, thresholds, firstLayer, secondLayer, firstWeights, middleWeights, trainingSet, eta, my);
            end = System.nanoTime();
            System.out.println(end-start);

            if (i % 10000 == 0) {
                System.out.println(classificationError(generateOutput(outputWeight,validationSet,thresholds,firstWeights,middleWeights), target));
            }
        }

        printMatrix(firstWeights, "w1.csv");
        printMatrix(middleWeights, "w2.csv");
        printArray(outputWeight, "w3.csv");
        printArray(thresholds[0], "t1.csv");
        printArray(thresholds[1], "t2.csv");
        printElement(outputThreshold, "t3.csv");




    }

    private static void printMatrix(double[][] matrix, String name) {
        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(name));
            StringBuilder sb = new StringBuilder();

            // Append strings from array
            for (double[] row : matrix) {
                sb.append(Arrays.toString(row).replaceAll("\\[","").replaceAll(" ","").replaceAll("\\]",""));
                sb.append(System.getProperty("line.separator"));
            }

            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printArray(double[] array, String name) {
        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(name));
            StringBuilder sb = new StringBuilder();

            // Append strings from array
            for (double element : array) {
                sb.append(element);
                sb.append(System.getProperty("line.separator"));
            }

            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printElement(double element, String name) {
        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(name));
            StringBuilder sb = new StringBuilder();

            // Append strings from array
            sb.append(element);

            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void updateWeights(double[] target, double[] output, double[] outputWeight, double[][] thresholds,
                                      double[][] firstLayer, double[][] secondLayer, double[][] firstWeight,
                                      double[][] secondWeight, double[][] input, double eta, int ny) {
        double delta3 = (1 - Math.pow(Math.tanh(calculateB3(outputWeight, outputThreshold, secondLayer[ny])), 2)) * (target[ny] - output[ny]);
        double[] delta2 = calculateDelta2(delta3, outputWeight, secondWeight, thresholds[1], firstLayer[ny]);
        double[] delta1 = calculateDelta1(delta2, secondWeight, thresholds[0], input[ny], firstWeight); //Bytte till first utan att kolla

        outputThreshold -= eta*delta3;
        for (int i = 0; i < thresholds[1].length; i++) {
            thresholds[1][i] -= eta * delta2[i];
        }

        for (int i = 0; i < thresholds[0].length; i++) {
            thresholds[0][i] -= eta * delta1[i];
        }

        for (int i = 0; i < outputWeight.length; i++) {
            outputWeight[i] += eta*delta3*secondLayer[ny][i];
        }

        for (int i = 0; i < secondWeight.length; i++) {
            for (int j = 0; j < secondWeight[i].length; j++) {
                secondWeight[i][j] += eta*delta2[i]*firstLayer[ny][j];
            }
        }

        for (int i = 0; i < firstWeight.length; i++) {
            for (int j = 0; j < firstWeight[i].length; j++) {
                firstWeight[i][j] += eta*delta1[i]*input[ny][j];
            }
        }

    }

    private static double[] generateOutput(double[] outputWeight, double[][] set, double[][] thresholds, double[][] firstWeights,
                                           double[][] middleWeights) {
        double[] output = new double[set.length];
        double[][] firstLayer = new double[set.length][firstLayerSize];
        double[][] secondLayer = new double[set.length][secondLayerSize];
        for (int i = 0; i < set.length; i++) {
            propigateForward(outputThreshold,outputWeight,set,thresholds,firstWeights,middleWeights, i, firstLayer, secondLayer, output);
        }
        return output;
    }

    private static double[] calculateDelta2(double delta3, double[] outputWeights ,double[][] weights, double[] thresholds, double[] input) {
        double[] delta2 = new double[weights.length];
        double[] bi = calculateB(input, thresholds, weights);
        for (int i = 0; i < delta2.length; i++) {
            delta2[i] = delta3*outputWeights[i]*(1-Math.pow(Math.tanh(bi[i]), 2));
        }
        return delta2;
    }

    private static double[] calculateDelta1(double[] delta2 ,double[][] secondWeights, double[] thresholds, double[] input, double[][] firstWeights) {
        double[] delta1 = new double[firstLayerSize];
        double[] bj = calculateB(input, thresholds, firstWeights);
        for (int j = 0; j < delta1.length; j++) {
            double sum = 0;
            for (int i = 0; i < secondLayerSize; i++) {
                sum += delta2[i]*secondWeights[i][j];
            }
            delta1[j] = sum  * (1 - Math.pow(Math.tanh(bj[j]), 2));
        }
        return delta1;
    }

    private static double[] calculateB(double[] input, double[] thresholds, double[][] weights) {
        double[] bi = new double[weights.length];

        for (int i = 0; i < bi.length; i++) {
            double sum = 0;
            for (int j = 0; j < weights[i].length; j++) {
                sum += weights[i][j] * input[j];
            }
            bi[i] = sum - thresholds[i];
        }
        return bi;
    }

    private static double calculateB3(double[] weights, double threshold, double[] input) {
        double sum = 0;
        for (int j = 0; j < weights.length; j++) {
            sum += weights[j] * input[j];
        }
        return sum - threshold;
    }




    private static void updateLayer(double[] thresholds, double[][] weights, double[] input, double[] neurons) {
        for (int i = 0; i < neurons.length; i++) {
            double sum = 0;
            for (int j = 0; j < input.length; j++) {
                sum += weights[i][j] * input[j];
            }
            neurons[i] = Math.tanh(-thresholds[i] + sum);
        }
    }

    //TODO: fix output
    private static void propigateForward(double threshold, double[] weights, double[][] set, double[][] thresholds,
                                           double[][] firstWeights, double[][] middleWeights, int my,
                                           double[][] firstLayer, double[][] secondLayer, double[] output) {
        long start = System.nanoTime();
        updateLayer(thresholds[0],firstWeights,set[my], firstLayer[my]);
        long end = System.nanoTime();
        System.out.println("Layer 1: " + (end -start));
        start = System.nanoTime();
        updateLayer(thresholds[1], middleWeights,firstLayer[my], secondLayer[my]);
        end = System.nanoTime();
        System.out.println("Layer 2: " + (end - start));

        start = System.nanoTime();
        double sum = 0;
        for (int i = 0; i < secondLayer[my].length; i++) {
            sum += weights[i]*secondLayer[my][i];
        }
        output[my] = Math.tanh(sum - threshold);
        end = System.nanoTime();
        System.out.println("Output: " + (end-start));
    }

    private static double classificationError(double[] output, double[] validationSet) {
        int sum = 0;
        for (int my = 0; my < output.length; my++) {
            sum+= (int) Math.abs(sgn(output[my]) - validationSet[my]);
        }
        return sum / 2.0 / validationSet.length;
    }

    private static double[][] generateWeights(int firstLayerSize, int secondLayerSize) {
        double[][] weights = new double[firstLayerSize][secondLayerSize];

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (Math.random() * 0.4) - 0.2;
            }
        }
        return weights;
    }

    private static double[] generateOutputWeights() {
        double[] outputWeights = new double[secondLayerSize];
        for (int i = 0; i < outputWeights.length; i++) {
            outputWeights[i] = Math.random() * 0.4 - 0.2;
        }
        return outputWeights;
    }

    private static int sgn(double value) {
        return value >= 0 ? 1 : -1;
    }

    public static void readSet (String csvFile, double[][] input, double[][] inputV, double[] t, double[] tV) {
        //String csvFile = "training_set.csv";
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";

        try {
            br = new BufferedReader(new FileReader(csvFile));
            int mu = 0;
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] temp = line.split(cvsSplitBy);
                if (csvFile.contains("training")) {
                    input[mu][0] = Double.parseDouble(temp[0]);
                    input[mu][1] = Double.parseDouble(temp[1]);
                    t[mu++] = Double.parseDouble(temp[2]);
                } else {
                    inputV[mu][0] = Double.parseDouble(temp[0]);
                    inputV[mu][1] = Double.parseDouble(temp[1]);
                    tV[mu++] = Double.parseDouble(temp[2]);
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
