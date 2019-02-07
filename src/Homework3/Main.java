package Homework3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class Main {

    public static double[][] xTrain;

    public static double[][] tTrain;

    public static double[][] xValid;

    public static double[][] tValid;

    public static double[][] xTest;

    public static double[][] tTest;

    private static double[] mean;

    public static void main(String[] args) {

        xTrain = new double[50000][784];
        loadFile("input.csv",xTrain);
        tTrain = new double[50000][10];
        loadFile("target.csv",tTrain);
        xValid = new double[10000][784];
        loadFile("inputValidation.csv",xValid);
        tValid = new double[10000][10];
        loadFile("targetValidation.csv",tValid);
        xTest = new double[10000][784];
        loadFile("inputTest.csv",xTest);
        tTest = new double[10000][10];
        loadFile("targetTest.csv",tTest);
        mean = new double[784];
        getMean(xTrain);
        centerData(xTrain);
        centerData(xValid);
        centerData(xTest);


        NeuralNetwork network;
        int i = 0;
        int[] layers;
        /*
        double[][] classificationErrors = new double[31][3];
        network = new NeuralNetwork(0.3,784,10,new int[0],xTrain,tTrain,xValid,tValid,xTest,tTest);

        for (i = 0;i<classificationErrors.length-1;i++) {
            classificationErrors[i] = network.getClassificationErrors();
            network.updateNetwork();
        }
        classificationErrors[i] = network.getClassificationErrors();

        double[][] classificationErrors1Layer30 = new double[31][3];
        layers = new int[] {30};
        network = new NeuralNetwork(0.3,784,10,layers,xTrain,tTrain,xValid,tValid,xTest,tTest);
        for (i = 0;i<classificationErrors1Layer30.length-1;i++) {
            classificationErrors1Layer30[i] = network.getClassificationErrors();
            network.updateNetwork();
        }
        classificationErrors1Layer30[i] = network.getClassificationErrors();

        double[][] classificationErrors1Layer100 = new double[31][3];
        layers = new int[] {100};
        network = new NeuralNetwork(0.3,784,10,layers,xTrain,tTrain,xValid,tValid,xTest,tTest);
        for (i = 0;i<classificationErrors1Layer100.length-1;i++) {
            classificationErrors1Layer100[i] = network.getClassificationErrors();
            network.updateNetwork();
        }
        classificationErrors1Layer100[i] = network.getClassificationErrors();


        double[][] classificationErrors2Layer = new double[31][3];
        layers = new int[] {100,100};
        network = new NeuralNetwork(0.3,784,10,layers,xTrain,tTrain,xValid,tValid,xTest,tTest);
        for (i = 0;i<classificationErrors2Layer.length-1;i++) {
            classificationErrors2Layer[i] = network.getClassificationErrors();
            network.updateNetwork();
        }
        classificationErrors2Layer[i] = network.getClassificationErrors();


        System.out.println("1C");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors[j][0]);
        }
        System.out.println("1V");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors[j][1]);
        }
        System.out.println("1T");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors[j][2]);
        }
        System.out.println("2C");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors1Layer30[j][0]);
        }
        System.out.println("2V");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors1Layer30[j][1]);
        }
        System.out.println("2T");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors1Layer30[j][2]);
        }
        System.out.println("3C");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors1Layer100[j][0]);
        }
        System.out.println("3V");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors1Layer100[j][1]);
        }
        System.out.println("3T");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors1Layer100[j][2]);
        }
        System.out.println("4C");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors2Layer[j][0]);
        }
        System.out.println("4V");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors2Layer[j][1]);
        }
        System.out.println("4T");
        for (int j = 0; j<31;j++) {
            System.out.println(j + "\t" + classificationErrors2Layer[j][2]);
        }
        */
        layers = new int[]{30,30,30,30};
        NeuralNetwork2 network2 = new NeuralNetwork2(3*Math.pow(10,-3),784,10,layers,xTrain,tTrain);
        double[][] results = new double[51][7];
        for (i = 0;i<results.length-1;i++) {
            results[i] = network2.getOutput();
            network2.updateNetwork();
        }
        results[i] = network2.getOutput();
        int k = 1;
        for (i = 0;i<results[0].length;i++) {
            System.out.println(i+1);
            for (int j = 0;j<results.length;j++) {
                System.out.println(j + "\t" + results[j][i]);
            }
        }

    }

    private static void getMean(double[][] matrix) {
        for (int i = 0;i<matrix[0].length;i++) {
            for (int j = 0;j<matrix.length;j++) {
                mean[i] += matrix[j][i];
            }
        }
        for (int i = 0;i<mean.length;i++) {
            mean[i] = mean[i]/matrix.length;
        }
    }

    private static void centerData(double[][] matrix) {
        for (int i = 0;i<matrix.length;i++) {
            for (int j = 0;j<matrix[i].length;j++) {
                matrix[i][j] -= mean[j];
            }
        }
    }

    public static void loadFile(String fileName, double[][] matrix) {
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";

        try {
            br = new BufferedReader(new FileReader(fileName));
            int i = 0;
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] temp = line.split(cvsSplitBy);
                for (int mu = 0; mu < temp.length; mu++) {
                    matrix[mu][i] = Double.parseDouble(temp[mu]);
                }
                i++;
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
