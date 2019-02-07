package Homework2;

import java.util.Random;

public class Question2 {

    private static double beta = 0.5;

    private static double etha = 0.02;

    private static double[] weights;

    private static double[] outputs;

    private static double theta;

    private static int[][] t;

    private static int[][] input;

    private static Random rand = new Random();

    public static void main(String[] args) {
        weights = new double[4];
        input = new int[][]{{-1, -1, -1, -1},
                {1, -1, -1, -1},
                {-1, 1, -1, -1},
                {-1, -1, 1, -1},
                {-1, -1, -1, 1},
                {1, 1, -1, -1},
                {1, -1, 1, -1},
                {1, -1, -1, 1},
                {-1, 1, 1, -1},
                {-1, 1, -1, 1},
                {-1, -1, 1, 1},
                {1, 1, 1, -1},
                {1, 1, -1, 1},
                {1, -1, 1, 1},
                {-1, 1, 1, 1},
                {1, 1, 1, 1}};
        t = new int[][]{
                {1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1},
                {1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1},
                {1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1},
                {1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1},
                {-1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1},
                {1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1}};
        for (int trial = 0; trial < t.length; trial++) {
            String linearlySeparable = "False";
            for (int i = 0; i < weights.length; i++) {
                weights[i] = (rand.nextDouble() % 0.4) - 0.2;
            }
            theta = rand.nextInt(2) + rand.nextDouble() -1;
            outputs = new double[input.length];

            for (int i = 0; i < 100000; i++) {
                int mu = rand.nextInt(input.length);
                double[] updateWeights = computeWeights(mu, trial);
                double updateTheta = computeTheta(mu, trial);
                update(updateWeights, updateTheta);

                outputs = computeOutputs();

                double error = computeClassificationError(trial);
                if (error == 0) {
                    linearlySeparable = "True";
                    break;
                }

            }

            System.out.println(trial+1 + ": " +  linearlySeparable);
        }
    }

    private static double[] computeWeights(int mu, int trial) {
        double[] updateWeights = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            updateWeights[i] = etha * beta * (t[trial][mu] - computeOutput(mu)) * (1 - Math.pow(Math.tanh(beta * computeB(mu)), 2)) * input[mu][i];
        }
        return updateWeights;
    }

    private static double computeTheta(int mu, int trial) {
        return -1 * etha * beta * (t[trial][mu] - computeOutput(mu)) * (1 - Math.pow(Math.tanh(beta * computeB(mu)), 2));

    }

    private static void update(double[] updateWeights, double updateTheta) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += updateWeights[i];
        }
        theta += updateTheta;

    }


    private static double[] computeOutputs() {
        double[] outputs = new double[input.length];
        for (int mu = 0; mu < input.length; mu++) {
            outputs[mu] = Math.tanh(beta * computeB(mu));
        }
        return outputs;
    }

    private static double computeOutput(int mu) {
        return Math.tanh(beta * computeB(mu));
    }

    private static double computeB(int mu) {
        double o = 0;
        for (int i = 0; i < weights.length; i++) {
            o += weights[i] * input[mu][i];
        }
        o -= theta;
        return o;
    }

    private static double computeClassificationError(int trial) {
        double error = 0;
        for (int mu = 0; mu < t[trial].length; mu++) {
            int o = sgn(outputs[mu]);
            int t1 = t[trial][mu];
            error += Math.abs(o - t1);
            //error += Math.abs(sgn(outputs[mu]) - t[trial][mu]);
        }
        error *= 0.5;
        return error;
    }

    private static int sgn(double o) {
        if (o > 0) {
            return 1;
        } else {
            return -1;
        }
    }
}
