package Homework1;


import java.util.Random;

public class Question3 {

    private static int N;

    private static int P;

    private static double[][] weights;

    private static int[][] patterns;

    private static int[] neurons;

    private static Random rand = new Random();

    public static void main(String args[]) {
        N = 200;
        P = 40;
        double sum = 0;
        int t;
        for (int r = 0; r < 100; r++) {
            weights = new double[N][N];
            patterns = new int[P][N];
            neurons = new int[N];
            patternGenerator();
            hebb();
            feed();
            double m_u = 0;
            for (t = 0; t < 100000; t++) {
                int neuron = rand.nextInt(N);
                neurons[neuron] = step(neuron);
                double count = 0;
                for (int i = 0; i < N; i++) {
                    count += neurons[i] * patterns[0][i];
                }
                m_u += count / N;
            }
            sum += m_u / t;
        }
        System.out.println(sum / 100);
    }

    private static int sgn(double value) {
        double probability = 1 / (1 + Math.exp(-2 * 2 * value));
        System.out.printf("Probability: %.5f\n", probability);
        System.out.printf("Value: %.5f\n", value);
        if (getRandomBoolean(probability)) {
            return 1;
        } else {
            return -1;
        }
    }

    private static void hebb() {
        for (int i = 0; i < patterns[0].length; i++) {
            for (int j = 0; j < patterns[0].length; j++) {
                double sum = 0;
                if (i != j) {
                    for (int nr = 0; nr < patterns.length; nr++) {
                        sum += patterns[nr][i] * patterns[nr][j];
                    }
                }
                weights[i][j] = sum / N;
            }
        }
    }


    private static void feed() {
        System.arraycopy(patterns[0],0,neurons,0,N);
    }

    private static int step(int i) {
        double s = 0;
        for (int j = 0; j < neurons.length; j++) {
            s += weights[i][j] * neurons[j];
        }
        return sgn(s);

    }


    private static boolean getRandomBoolean(double p) {
        return rand.nextDouble() < p;
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


}
