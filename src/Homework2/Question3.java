package Homework2;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class Question3 {

    private static double[][] input;

    private static double[] t;

    private static double[][] inputV;

    private static double[] tV;

    private static double[] v_j;

    private static double[] v_i;

    private static double output;

    private static double[] theta_j;

    private static double[] theta_i;

    private static double theta;

    private static double[][] weights_jk;

    private static double[][] weights_ij;

    private static double[] weights_i;

    private static int m1;

    private static int m2;

    private static double etha;

    private static Random rand = new Random();

    public static void main(String[] args) {
        input = new double[10000][2];
        t = new double[10000];
        inputV = new double[5000][2];
        tV = new double[5000];
        readSet("resources/training_set.csv");
        readSet("resources/validation_set.csv");
        m1 = 4;
        m2 = 2;
        etha = 0.02;
        v_j = new double[m1];
        weights_jk = new double[v_j.length][2];
        v_i = new double[m2];
        weights_ij = new double[v_i.length][v_j.length];
        weights_i = new double[v_i.length];
        output = 0;

        for (int j = 0; j < weights_jk.length; j++) {
            for (int k = 0; k < weights_jk[j].length; k++) {
                weights_jk[j][k] = (rand.nextDouble() % 0.4) - 0.2;
            }
        }

        for (int i = 0; i < weights_ij.length; i++) {
            for (int j = 0; j < weights_ij[i].length; j++) {
                weights_ij[i][j] = (rand.nextDouble() % 0.4) - 0.2;
            }
        }

        for (int i = 0; i < weights_i.length; i++) {
            weights_i[i] = (rand.nextDouble() % 0.4) - 0.2;
        }

        theta_j = new double[v_j.length];
        theta_i = new double[v_i.length];
        /*for (int j = 0; j < theta_j.length; j++) {
            theta_j[j] = rand.nextInt(2) + rand.nextDouble() - 1;
        }
        for (int i = 0; i < theta_i.length; i++) {
            theta_i[i] = rand.nextInt(2) + rand.nextDouble() - 1;
        }*/
        theta = 0;
        //theta = rand.nextInt(2) + rand.nextDouble() - 1;
        int i = 0;
        while (true) {
            int mu = rand.nextInt(t.length);
            //Propagate and update neuron values:
            long start = System.nanoTime();
            computeO(mu);
            long end = System.nanoTime();
            //System.out.println(end-start);
            //Backpropagation

            start = System.nanoTime();
            updateNetwork(mu);
            end = System.nanoTime();
            //System.out.println(end-start);

            /*double h = 0;
            for (int m = 0; m<t.length;m++) {
                computeO(m);
                h += Math.pow((t[m] - output),2);
            }
            h *= 0.5;
            System.out.println(h);
            System.out.println(computeClassification());*/
            start = System.nanoTime();
            if (i%1000==0) {
                double error = computeClassificationError();
                end = System.nanoTime();
                //System.out.println(end - start);
                if (error < 0.12) {
                    System.out.println("Success!");
                    printMatrix(weights_jk, "w1.csv");
                    printMatrix(weights_ij, "w2.csv");
                    printArray(weights_i, "w3.csv");
                    printArray(theta_j, "t1.csv");
                    printArray(theta_i, "t2.csv");
                    printElement(theta, "t3.csv");
                    break;
                } else if (i % 10000 == 0) {
                    System.out.println(i + ": " + error);
                /*double hC = 0;
                for (int m = 0; m<tV.length;m++) {
                    computeOV(m);
                    hC += Math.pow((tV[m] - output),2);
                }
                hC *= 0.5;
                System.out.println(hC);*/
                }
            }
            i++;
        }


    }

    private static void propagateV_j(int mu) {
        for (int j = 0; j < v_j.length; j++) {
            double b = -theta_j[j];
            for (int k = 0; k < input[mu].length; k++) {
                b += weights_jk[j][k] * input[mu][k];
            }
            v_j[j] = Math.tanh(b);
        }
    }

    private static void propagateV_i() {
        for (int i = 0; i < v_i.length; i++) {
            double b = -theta_i[i];
            for (int j = 0; j < v_j.length; j++) {
                b += weights_ij[i][j] * v_j[j];
            }
            v_i[i] = Math.tanh(b);
        }
    }

    private static void propagateO() {
        double b = -theta;
        for (int i = 0; i < v_i.length; i++) {
            b += weights_i[i] * v_i[i];
        }
        output = Math.tanh(b );
    }

    private static void computeO(int mu) {
        long start = System.nanoTime();
        propagateV_j(mu);
        long end = System.nanoTime();
        //System.out.println("V_j: " + (end-start));
        start = System.nanoTime();
        propagateV_i();
        end = System.nanoTime();
        //System.out.println("V_i: " + (end-start));
        start = System.nanoTime();
        propagateO();
        end = System.nanoTime();
        //System.out.println("Output: " + (end-start));
    }

    private static double delta3(int mu) {
        //Compute b
        double b = -theta;
        for (int i = 0; i < v_i.length; i++) {
            b += weights_i[i] * v_i[i];
        }
        //b -= theta;

        return (1 - Math.pow(Math.tanh(b), 2)) * (t[mu] - output);
    }

    private static double[] delta2(double delta3) {
        //Compute b
        double[] delta2 = new double[v_i.length];
        for (int i = 0; i < weights_ij.length; i++) {
            double b = -theta_i[i];
            for (int j = 0; j < weights_ij[i].length; j++) {
                b += weights_ij[i][j] * v_j[j];
            }
            //b -= theta_i[i];
            delta2[i] = delta3 * weights_i[i] * (1 - Math.pow(Math.tanh(b), 2));
        }



        return delta2;
    }

    private static double[] delta1(double delta2[], int mu) {
        //Compute b
        double[] delta1 = new double[v_j.length];
        for (int j = 0; j < weights_jk.length; j++) {
            double b = -theta_j[j];
            for (int k = 0; k < weights_jk[j].length; k++) {
                b += weights_jk[j][k] * input[mu][k];
            }

            // -= theta_j[j];
            for (int i = 0; i < v_i.length; i++) {
                delta1[j] += delta2[i] * weights_ij[i][j] * (1 - Math.pow(Math.tanh(b), 2));
            }
        }


        return delta1;
    }

    private static void updateNetwork(int mu) {
        double delta3 = delta3(mu);
        double[] delta2 = delta2(delta3);
        double[] delta1 = delta1(delta2, mu);
        updateWeights(delta3, delta2, delta1, mu);
        updateThetas(delta3, delta2, delta1);
    }

    private static void updateWeights(double delta3, double[] delta2, double[] delta1, int mu) {
        for (int i = 0; i < weights_i.length; i++) {
            weights_i[i] += etha * delta3 * v_i[i];
        }
        for (int i = 0; i < weights_ij.length; i++) {
            for (int j = 0; j < weights_ij[i].length; j++) {
                weights_ij[i][j] += etha * delta2[i] * v_j[j];
            }
        }
        for (int j = 0; j < weights_jk.length; j++) {
            for (int k = 0; k < weights_jk[j].length; k++) {
                weights_jk[j][k] += etha * delta1[j] * input[mu][k];
            }
        }
    }

    private static void updateThetas(double delta3, double[] delta2, double[] delta1) {
        theta -= etha * delta3;
        for (int i = 0; i < theta_i.length; i++) {
            theta_i[i] -= etha * delta2[i];
        }
        for (int j = 0; j < theta_j.length; j++) {
            theta_j[j] -= etha * delta1[j];
        }
    }

    private static double computeClassificationError() {
        double error = 0;
        for (int mu = 0; mu < tV.length; mu++) {
            computeOV(mu);
            error += Math.abs(sgn(output) - tV[mu]);
        }
        return error / (2 * tV.length);
    }

    private static void propagateV_jV(int mu) {
        for (int j = 0; j < v_j.length; j++) {
            double b = 0;
            for (int k = 0; k < inputV[mu].length; k++) {
                b += weights_jk[j][k] * inputV[mu][k];
            }
            v_j[j] = Math.tanh(-theta_j[j] + b);
        }
    }

    private static void computeOV(int mu) {
        propagateV_jV(mu);
        propagateV_i();
        propagateO();
    }


    private static int sgn(double o) {
        if (o > 0) {
            return 1;
        } else {
            return -1;
        }
    }

    private static void readSet(String csvFile) {
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
}
