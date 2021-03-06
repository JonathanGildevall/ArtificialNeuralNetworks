package Homework1;


import java.util.Random;

public class Question2 {

    private static int N;

    private static double[][] weights;

    private static int[][][] patterns;

    private static int[][] neurons;

    private static Random rand = new Random();

    public static void main(String args[]) {
        N = 160;
        setPatterns();
        weights = new double[N][N];
        hebb();
        feed();
        print();
        int t = 0;
        do {
            int neuron = t%N;//rand.nextInt(t%N);
            int row = neuron / 10;
            int col = neuron % 10;
            neurons[row][col] = step(neuron);
            System.out.println(++t);
        } while (!trySyncStep());
        print();
        printAnswer();
        /*neurons = patterns[0];
        print();
        neurons = patterns[1];
        print();
        neurons = patterns[2];
        print();
        neurons = patterns[3];
        print();
        neurons = patterns[4];
        print();*/

    }

    private static int sgn(double value) {
        if (value >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

    private static void hebb() {

        int patternMatrix[][] = new int[patterns.length][patterns[0].length * patterns[0][0].length];
        for (int p = 0; p < patterns.length; p++) {
            for (int i = 0; i < patterns[0].length; i++) {
                int[] row = patterns[p][i];
                for (int j = 0; j < row.length; j++) {
                    int number = patterns[p][i][j];
                    patternMatrix[p][i * row.length + j] = number;
                }
            }
        }


        for (int i = 0; i < patternMatrix[0].length; i++) {
            for (int j = 0; j < patternMatrix[0].length; j++) {
                double sum = 0;
                if (i != j) {
                    for (int nr = 0; nr < patternMatrix.length; nr++) {
                        sum += patternMatrix[nr][i] * patternMatrix[nr][j];
                    }
                }
                weights[i][j] = sum / N;
            }
        }
    }


    private static void feed() {
        //1
        //neurons = new int[][]{{1, -1, 1, -1, 1, -1, 1, -1, 1, -1}, {1, -1, 1, 1, -1, 1, -1, -1, 1, -1}, {1, -1, -1, 1, -1, 1, -1, 1, 1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, -1, -1, 1, -1, 1, -1, 1, 1, -1}, {1, -1, 1, 1, -1, 1, -1, -1, 1, -1}, {1, -1, 1, -1, 1, -1, 1, -1, 1, -1}};
        //neurons = patterns[0];

        //2
        //neurons = new int[][] {{1, 1, 1, 1, 1, 1, 1, 1, -1, -1}, {1, 1, 1, 1, 1, 1, 1, 1, -1, -1}, {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1}, {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1}, {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1}, {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1}, {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1}, {1, 1, 1, 1, 1, 1, 1, 1, -1, -1}, {1, 1, 1, 1, 1, 1, 1, 1, -1, -1}, {1, 1, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 1, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 1, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 1, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 1, 1, -1, -1, -1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 1, 1, 1, -1, -1}, {-1, -1, -1, -1, -1, -1, -1, -1, 1, 1}};

        //3
        //neurons = new int[][] {{1, 1, 1, -1, -1, -1, -1, 1, 1, 1}, {1, 1, 1, -1, -1, -1, -1, 1, 1, 1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}, {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}};


        neurons = new int[][] {{1, -1, 1, -1, 1, -1, 1, -1, 1, -1}, {1, -1, 1, 1, -1, 1, -1, -1, 1, -1}, {1, -1, -1, 1, -1, 1, -1, 1, 1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, 1, -1, 1, 1, -1, -1, 1, -1, -1}, {1, -1, -1, 1, -1, 1, -1, 1, 1, -1}, {1, -1, 1, 1, -1, 1, -1, -1, 1, -1}, {1, -1, 1, -1, 1, -1, 1, -1, 1, -1}};
    }

    private static int step(int i) {

        int neuronArray[] = new int[neurons.length * neurons[0].length];
        for (int k = 0; k < neurons.length; k++) {
            int[] row = neurons[k];
            for (int j = 0; j < row.length; j++) {
                int number = neurons[k][j];
                neuronArray[k * row.length + j] = number;
            }
        }

        double s = 0;
        for (int j = 0; j < neuronArray.length; j++) {
            s += weights[i][j] * neuronArray[j];
        }
        //int row = i / 10;
        //int col = i % 10;
        //neurons[row][col] = sgn(s);
        return sgn(s);

    }

    private static void print() {
        for (int row = 0; row < neurons.length; row++) {
            for (int col = 0; col < neurons[row].length; col++) {
                if (neurons[row][col] == 1) {
                    System.out.print("□");
                } else {
                    System.out.print("■");
                }
            }
            System.out.print("\n");
        }
    }

    private static void printAnswer() {
        System.out.print("[");
        for (int row = 0; row < neurons.length; row++) {
            if (row != 0) {
                System.out.print(", ");
            }
            System.out.print("[");
            for (int col = 0; col < neurons[row].length; col++) {
                if (col != 0) {
                    System.out.print(", ");
                }
                System.out.print(neurons[row][col]);
            }
            System.out.print("]");
        }
        System.out.print("]");
    }

    private static boolean trySyncStep() {

        /*int neuronArray[] = new int[neurons.length * neurons[0].length];
        for (int k = 0; k < neurons.length; k++) {
            int[] row = neurons[k];
            for (int j = 0; j < row.length; j++) {
                int number = neurons[k][j];
                neuronArray[k * row.length + j] = number;
            }
        }

        for (int i = 0; i < N; i++) {
            double s = 0;
            for (int j = 0; j < neuronArray.length; j++) {
                s += weights[i][j] * neuronArray[j];
            }
            if (sgn(s) != step(i)) {
                System.out.println("ERROR");
            }
            if (neuronArray[i] != sgn(s)) {
                return false;
            }
        }
        */
        for (int i = 0; i < N; i++) {
            int row = i / 10;
            int col = i % 10;
            if(step(i) != neurons[row][col]) {
                return false;
            }
        }
        return true;
    }

    private static void setPatterns() {
        patterns = new int[5][10][16];
        patterns[0] = new int[][]{
                {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, -1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, 1, 1, 1, -1, -1, 1, 1, 1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
        patterns[1] = new int[][]{
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1},
                {-1, -1, -1, 1, 1, 1, 1, -1, -1, -1}
        };
        patterns[2] = new int[][]{
                {1, 1, 1, 1, 1, 1, 1, 1, -1, -1},
                {1, 1, 1, 1, 1, 1, 1, 1, -1, -1},
                {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1},
                {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1},
                {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1},
                {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1},
                {-1, -1, -1, -1, -1, 1, 1, 1, -1, -1},
                {1, 1, 1, 1, 1, 1, 1, 1, -1, -1},
                {1, 1, 1, 1, 1, 1, 1, 1, -1, -1},
                {1, 1, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 1, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 1, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 1, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 1, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 1, 1, 1, 1, 1, 1, 1, -1, -1},
                {1, 1, 1, 1, 1, 1, 1, 1, -1, -1},
        };
        patterns[3] = new int[][]{
                {-1, -1, 1, 1, 1, 1, 1, 1, -1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, -1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, -1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, 1, -1},
                {-1, -1, 1, 1, 1, 1, 1, 1, -1, -1},
        };
        patterns[4] = new int[][]{
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, -1, -1, -1, -1, 1, 1, -1},
                {-1, 1, 1, 1, 1, 1, 1, 1, 1, -1},
                {-1, 1, 1, 1, 1, 1, 1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
                {-1, -1, -1, -1, -1, -1, -1, 1, 1, -1},
        };
    }

}
