package Homework1;


import java.util.Random;

public class Question1 {

    private static int N;

    private static int P;

    private static double[][] weights;

    private static int[][] patterns;

    private static int[] neurons;

    private static Random rand = new Random();

    public static void main(String args[]) {
        N = 100;
        //int[][] error = new int[6][100000];
        double[] answers = new double[6];
        for (int r = 0; r < 6; r++) {
            switch (r) {
                case 0:
                    P = 12;
                    break;
                case 1:
                    P = 20;
                    break;
                default:
                    P += 20;
            }
            double sum = 0;
            for (int n = 0; n < 100000; n++) {
                patterns = new int[P][N];
                weights = new double[N][N];
                neurons = new int[N];
                patternGenerator();
                hebb();
                int nr = rand.nextInt(P);
                feed(nr);
                int neuron = rand.nextInt(100);
                step(neuron);
                if (!(neurons[neuron] == patterns[nr][neuron])) {
                    sum++;
                }
            }
            answers[r] = sum / 100000;
            System.out.println(P);
            System.out.println(answers[r]);
        }
        for (int i = 0; i < 6; i++) {
            System.out.println(answers[i]);
        }
    }

    private static int sgn(double value) {
        if (value >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

    private static void hebb() {
        for (int i = 0; i < patterns[0].length; i++) {
            for (int j = 0; j < patterns[0].length; j++) {
                double sum = 0;
                //if (i != j) {
                    for (int nr = 0; nr < patterns.length; nr++) {
                        sum += patterns[nr][i] * patterns[nr][j];
                    }
                //}
                weights[i][j] = sum / N;
            }
        }
    }

    private static void patternGenerator() {
        for (int nr = 0; nr < patterns.length; nr++) {
            for (int i = 0; i < patterns[nr].length; i++) {
                int random = rand.nextInt(2);
                if (random == 0) {
                    random = -1;
                }
                patterns[nr][i] = random;
            }
        }
    }

    private static void feed(int nr) {
        System.arraycopy(patterns[nr], 0, neurons, 0, 100);
    }

    private static void step(int i) {
        double s = 0;
        for (int j = 0; j < neurons.length; j++) {
            s += weights[i][j] * neurons[j];
        }
        neurons[i] = sgn(s);

    }

}
