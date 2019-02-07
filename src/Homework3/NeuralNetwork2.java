package Homework3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class NeuralNetwork2 {

    private double eta;

    private int hiddenLayers;

    private double[] outputThresholds;

    private double[] layer4Thresholds;

    private double[] layer3Thresholds;

    private double[] layer2Thresholds;

    private double[] layer1Thresholds;

    private double[][] outputWeights;

    private double[][] layer4Weights;

    private double[][] layer3Weights;

    private double[][] layer2Weights;

    private double[][] layer1Weights;

    private double[] layer4Neurons;

    private double[] layer3Neurons;

    private double[] layer2Neurons;

    private double[] layer1Neurons;

    private double[][] outputTrain;

    private double[][] xTrain;

    private double[][] tTrain;

    List<Integer> shuffledPatterns;

    private Random rand = new Random();

    public NeuralNetwork2(double eta, int inputSize, int outputSize, int[] hiddenSize, double[][] xDataT, double[][] tDataT) {

        this.eta = eta;
        hiddenLayers = hiddenSize.length;
        setTrainingData(xDataT, tDataT);
        outputThresholds = new double[outputSize];
        layer4Thresholds = new double[hiddenSize[3]];
        layer4Neurons = new double[hiddenSize[3]];
        outputWeights = new double[outputSize][hiddenSize[3]];
        for (int i = 0; i < outputWeights.length; i++) {
            for (int j = 0; j < outputWeights[i].length; j++) {
                outputWeights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(outputWeights[i].length));
            }
        }
        layer3Thresholds = new double[hiddenSize[2]];
        layer3Neurons = new double[hiddenSize[2]];
        layer4Weights = new double[hiddenSize[3]][hiddenSize[2]];
        for (int i = 0; i < layer4Weights.length; i++) {
            for (int j = 0; j < layer4Weights[i].length; j++) {
                layer4Weights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(layer4Weights[i].length));
            }
        }

        layer2Thresholds = new double[hiddenSize[1]];
        layer2Neurons = new double[hiddenSize[1]];
        layer3Weights = new double[hiddenSize[2]][hiddenSize[1]];
        for (int i = 0; i < layer3Weights.length; i++) {
            for (int j = 0; j < layer3Weights[i].length; j++) {
                layer3Weights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(layer3Weights[i].length));
            }
        }

        layer1Thresholds = new double[hiddenSize[0]];
        layer1Neurons = new double[hiddenSize[0]];
        layer2Weights = new double[hiddenSize[1]][hiddenSize[0]];
        for (int i = 0; i < layer2Weights.length; i++) {
            for (int j = 0; j < layer2Weights[i].length; j++) {
                layer2Weights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(layer2Weights[i].length));
            }
        }

        layer1Weights = new double[hiddenSize[0]][inputSize];
        for (int i = 0; i < layer1Weights.length; i++) {
            for (int j = 0; j < layer1Weights[i].length; j++) {
                layer1Weights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(layer1Weights[i].length));
            }
        }
    }


    private void propagateNetwork(int mu, double[][] input) {

        for (int i = 0; i < layer1Neurons.length; i++) {
            double b = -layer1Thresholds[i];
            for (int j = 0; j < input[mu].length; j++) {
                b += layer1Weights[i][j] * input[mu][j];
            }
            layer1Neurons[i] = sigmoid(b);
        }
        for (int i = 0; i < layer2Neurons.length; i++) {
            double b = -layer2Thresholds[i];
            for (int j = 0; j < layer1Neurons.length; j++) {
                b += layer2Weights[i][j] * layer1Neurons[j];
            }
            layer2Neurons[i] = sigmoid(b);

        }
        for (int i = 0; i < layer3Neurons.length; i++) {
            double b = -layer3Thresholds[i];
            for (int j = 0; j < layer2Neurons.length; j++) {
                b += layer3Weights[i][j] * layer2Neurons[j];
            }
            layer3Neurons[i] = sigmoid(b);

        }
        for (int i = 0; i < layer4Neurons.length; i++) {
            double b = -layer4Thresholds[i];
            for (int j = 0; j < layer3Neurons.length; j++) {
                b += layer4Weights[i][j] * layer3Neurons[j];
            }
            layer4Neurons[i] = sigmoid(b);

        }
        for (int i = 0; i < outputWeights.length; i++) {
            double b = -outputThresholds[i];
            for (int j = 0; j < layer4Neurons.length; j++) {
                b += outputWeights[i][j] * layer4Neurons[j];
            }
            outputTrain[mu][i] = sigmoid(b);
        }
    }

    public void updateNetwork() {
        Collections.shuffle(shuffledPatterns);

        for (int mu = 0; mu < xTrain.length - 10; mu += 10) {
            double outputW[][] = new double[outputTrain[mu].length][layer4Neurons.length];
            double layer4W[][] = new double[layer4Neurons.length][layer3Neurons.length];
            double layer3W[][] = new double[layer3Neurons.length][layer2Neurons.length];
            double layer2W[][] = new double[layer2Neurons.length][layer1Neurons.length];
            double layer1W[][] = new double[layer1Neurons.length][xTrain[mu].length];
            double outputT[] = new double[outputTrain[mu].length];
            double layer4T[] = new double[layer4Neurons.length];
            double layer3T[] = new double[layer3Neurons.length];
            double layer2T[] = new double[layer2Neurons.length];
            double layer1T[] = new double[layer1Neurons.length];
            for (int mb = mu; mb < mu + 10; mb++) {
                int pattern = shuffledPatterns.get(mb);
                propagateNetwork(pattern, xTrain);
                //Compute b
                double[] outputDeltas = new double[outputTrain[pattern].length];
                for (int i = 0; i < outputTrain[pattern].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < layer4Neurons.length; j++) {
                        b += outputWeights[i][j] * layer4Neurons[j];
                    }
                    outputDeltas[i] = sigmoidDelta(b) * (tTrain[pattern][i] - outputTrain[pattern][i]);
                }

                double[] layer4Deltas = new double[layer4Neurons.length];
                for (int i = 0; i < layer4Neurons.length; i++) {
                    double b = -layer4Thresholds[i];
                    for (int j = 0; j < layer3Neurons.length; j++) {
                        b += layer4Weights[i][j] * layer3Neurons[j];
                    }
                    for (int k = 0; k < outputDeltas.length; k++) {
                        layer4Deltas[i] += outputDeltas[k] * outputWeights[k][i] * sigmoidDelta(b);
                    }
                }

                double[] layer3Deltas = new double[layer3Neurons.length];
                for (int i = 0; i < layer3Neurons.length; i++) {
                    double b = -layer3Thresholds[i];
                    for (int j = 0; j < layer2Neurons.length; j++) {
                        b += layer3Weights[i][j] * layer2Neurons[j];
                    }
                    for (int k = 0; k < layer4Deltas.length; k++) {
                        layer3Deltas[i] += layer4Deltas[k] * layer4Weights[k][i] * sigmoidDelta(b);
                    }
                }


                double[] layer2Deltas = new double[layer2Neurons.length];
                for (int i = 0; i < layer2Neurons.length; i++) {
                    double b = -layer2Thresholds[i];
                    for (int j = 0; j < layer1Neurons.length; j++) {
                        b += layer2Weights[i][j] * layer1Neurons[j];
                    }
                    for (int k = 0; k < layer3Deltas.length; k++) {
                        layer2Deltas[i] += layer3Deltas[k] * layer3Weights[k][i] * sigmoidDelta(b);
                    }
                }


                double[] layer1Deltas = new double[layer1Neurons.length];
                for (int i = 0; i < layer1Neurons.length; i++) {
                    double b = -layer1Thresholds[i];
                    for (int j = 0; j < xTrain[pattern].length; j++) {
                        b += layer1Weights[i][j] * xTrain[pattern][j];
                    }
                    for (int k = 0; k < layer2Deltas.length; k++) {
                        layer1Deltas[i] += layer2Deltas[k] * layer2Weights[k][i] * sigmoidDelta(b);
                    }
                }

                for (int i = 0; i < outputWeights.length; i++) {
                    for (int j = 0; j < outputWeights[i].length; j++) {

                        outputW[i][j] += outputDeltas[i] * layer2Neurons[j];
                    }
                }
                for (int i = 0; i < layer4Weights.length; i++) {
                    for (int j = 0; j < layer4Weights[i].length; j++) {
                        layer4W[i][j] += layer4Deltas[i] * layer3Neurons[j];
                    }
                }
                for (int i = 0; i < layer3Weights.length; i++) {
                    for (int j = 0; j < layer3Weights[i].length; j++) {
                        layer3W[i][j] += layer3Deltas[i] * layer2Neurons[j];
                    }
                }
                for (int i = 0; i < layer2Weights.length; i++) {
                    for (int j = 0; j < layer2Weights[i].length; j++) {
                        layer2W[i][j] += layer2Deltas[i] * layer1Neurons[j];
                    }
                }
                for (int j = 0; j < layer1Weights.length; j++) {
                    for (int k = 0; k < layer1Weights[j].length; k++) {
                        layer1W[j][k] += layer1Deltas[j] * xTrain[pattern][k];
                    }
                }

                for (int i = 0; i < outputThresholds.length; i++) {
                    outputT[i] += outputDeltas[i];
                }
                for (int i = 0; i < layer4Thresholds.length; i++) {
                    layer4T[i] += layer4Deltas[i];
                }
                for (int i = 0; i < layer3Thresholds.length; i++) {
                    layer3T[i] += layer3Deltas[i];
                }
                for (int i = 0; i < layer2Thresholds.length; i++) {
                    layer2T[i] += layer2Deltas[i];
                }
                for (int i = 0; i < layer1Thresholds.length; i++) {
                    layer1T[i] += layer1Deltas[i];
                }


            }
            for (int i = 0; i < outputWeights.length; i++) {
                for (int j = 0; j < outputWeights[i].length; j++) {

                    outputWeights[i][j] += eta * outputW[i][j];
                }
            }

            for (int i = 0; i < layer4Weights.length; i++) {
                for (int j = 0; j < layer4Weights[i].length; j++) {
                    layer4Weights[i][j] += eta * layer4W[i][j];
                }
            }


            for (int i = 0; i < layer3Weights.length; i++) {
                for (int j = 0; j < layer3Weights[i].length; j++) {
                    layer3Weights[i][j] += eta * layer3W[i][j];
                }
            }


            for (int i = 0; i < layer2Weights.length; i++) {
                        for (int j = 0; j < layer2Weights[i].length; j++) {
                            layer2Weights[i][j] += eta * layer2W[i][j];
                }
            }
            for (int j = 0; j < layer1Weights.length; j++) {
                for (int k = 0; k < layer1Weights[j].length; k++) {
                    layer1Weights[j][k] += eta * layer1W[j][k];
                }
            }

            for (int i = 0; i < outputThresholds.length; i++) {
                outputThresholds[i] -= eta * outputT[i];
            }
            for (int i = 0; i < layer4Thresholds.length; i++) {
                layer4Thresholds[i] -= eta * layer4T[i];
            }
            for (int i = 0; i < layer3Thresholds.length; i++) {
                layer3Thresholds[i] -= eta * layer3T[i];
            }
            for (int i = 0; i < layer2Thresholds.length; i++) {
                layer2Thresholds[i] -= eta * layer2T[i];
            }
            for (int i = 0; i < layer1Thresholds.length; i++) {
                layer1Thresholds[i] -= eta * layer1T[i];
            }

        }
    }

    public double getClassificationErrors() {
        double error = 0;
        for (int mu = 0; mu < tTrain.length; mu++) {
            propagateNetwork(mu, xTrain);
        }
        for (int mu = 0; mu < outputTrain.length; mu++) {
            for (int i = 0; i < outputTrain[mu].length; i++) {
                error += Math.abs(tTrain[mu][i] - largest(outputTrain[mu], i));
            }
        }

        return error / (2 * xTrain.length);
    }

    private void setTrainingData(double[][] xData, double[][] tData) {
        xTrain = new double[xData.length][xData[0].length];
        outputTrain = new double[tData.length][tData[0].length];
        for (int i = 0; i < xData.length; i++) {
            for (int j = 0; j < xData[i].length; j++) {
                xTrain[i][j] = xData[i][j];
            }
        }

        tTrain = new double[tData.length][tData[0].length];
        for (int i = 0; i < tData.length; i++) {
            for (int j = 0; j < tData[i].length; j++) {
                tTrain[i][j] = tData[i][j];
            }
        }
        shuffledPatterns = new ArrayList<>(xData.length);
        for (int i = 0; i < xData.length; i++) {
            shuffledPatterns.add(i);
        }
    }

    private double sigmoid(double b) {
        return (Math.pow((1 + Math.exp(-b)), -1));
    }

    private double sigmoidDelta(double b) {
        return (sigmoid(b) - Math.pow(sigmoid(b), 2));
    }

    private int largest(double[] o, int i) {
        for (int j = 0; j < o.length; j++) {
            if (i != j) {
                if (o[i] <= o[j])
                    return 0;
            }
        }
        return 1;

    }

    private double getEnergy() {
        double energy = 0;
        for (int mu = 0;mu<tTrain.length;mu++) {
            for (int i = 0;i<tTrain[mu].length;i++) {
                energy += Math.pow((tTrain[mu][i]-outputTrain[mu][i]),2);
            }
        }
        return energy/2;
    }

    public double[] getOutput() {
        double error = getClassificationErrors();
        double energy = getEnergy();
        double outputT[] = new double[outputTrain[0].length];
        double layer4T[] = new double[layer4Neurons.length];
        double layer3T[] = new double[layer3Neurons.length];
        double layer2T[] = new double[layer2Neurons.length];
        double layer1T[] = new double[layer1Neurons.length];

        double[] norms = new double[5];
        for (int pattern = 0; pattern < xTrain.length; pattern++) {
            propagateNetwork(pattern, xTrain);
            double[] outputDeltas = new double[outputTrain[pattern].length];
            double[] layer4Deltas = new double[layer4Neurons.length];
            double[] layer3Deltas = new double[layer3Neurons.length];
            double[] layer2Deltas = new double[layer2Neurons.length];
            double[] layer1Deltas = new double[layer1Neurons.length];
            //Compute b

            for (int i = 0; i < outputTrain[pattern].length; i++) {
                double b = -outputThresholds[i];
                for (int j = 0; j < layer4Neurons.length; j++) {
                    b += outputWeights[i][j] * layer4Neurons[j];
                }
                outputDeltas[i] += sigmoidDelta(b) * (tTrain[pattern][i] - outputTrain[pattern][i]);
            }


            for (int i = 0; i < layer4Neurons.length; i++) {
                double b = -layer4Thresholds[i];
                for (int j = 0; j < layer3Neurons.length; j++) {
                    b += layer4Weights[i][j] * layer3Neurons[j];
                }
                for (int k = 0; k < outputDeltas.length; k++) {
                    layer4Deltas[i] += outputDeltas[k] * outputWeights[k][i] * sigmoidDelta(b);
                }
            }


            for (int i = 0; i < layer3Neurons.length; i++) {
                double b = -layer3Thresholds[i];
                for (int j = 0; j < layer2Neurons.length; j++) {
                    b += layer3Weights[i][j] * layer2Neurons[j];
                }
                for (int k = 0; k < layer4Deltas.length; k++) {
                    layer3Deltas[i] += layer4Deltas[k] * layer4Weights[k][i] * sigmoidDelta(b);
                }
            }



            for (int i = 0; i < layer2Neurons.length; i++) {
                double b = -layer2Thresholds[i];
                for (int j = 0; j < layer1Neurons.length; j++) {
                    b += layer2Weights[i][j] * layer1Neurons[j];
                }
                for (int k = 0; k < layer3Deltas.length; k++) {
                    layer2Deltas[i] += layer3Deltas[k] * layer3Weights[k][i] * sigmoidDelta(b);
                }
            }



            for (int i = 0; i < layer1Neurons.length; i++) {
                double b = -layer1Thresholds[i];
                for (int j = 0; j < xTrain[pattern].length; j++) {
                    b += layer1Weights[i][j] * xTrain[pattern][j];
                }
                for (int k = 0; k < layer2Deltas.length; k++) {
                    layer1Deltas[i] += layer2Deltas[k] * layer2Weights[k][i] * sigmoidDelta(b);
                }
            }
            for (int i = 0;i<outputDeltas.length;i++) {
                outputT[i] += outputDeltas[i];
            }
            for (int i = 0;i<layer4Deltas.length;i++) {
                layer4T[i] += layer4Deltas[i];
                layer3T[i] += layer3Deltas[i];
                layer2T[i] += layer2Deltas[i];
                layer1T[i] += layer1Deltas[i];
            }

            /*double temp = 0;
            for (int i = 0;i<layer1Deltas.length;i++) {
                temp += Math.pow(layer1Deltas[i],2);
            }
            norms[0] += Math.sqrt(temp);

            temp = 0;
            for (int i = 0;i<layer2Deltas.length;i++) {
                temp += Math.pow(layer2Deltas[i],2);
            }
            norms[1] += Math.sqrt(temp);

            temp = 0;
            for (int i = 0;i<layer3Deltas.length;i++) {
                temp += Math.pow(layer3Deltas[i],2);
            }
            norms[2] += Math.sqrt(temp);

            temp = 0;
            for (int i = 0;i<layer4Deltas.length;i++) {
                temp += Math.pow(layer4Deltas[i],2);
            }
            norms[3] += Math.sqrt(temp);

            temp = 0;
            for (int i = 0;i<outputDeltas.length;i++) {
                temp += Math.pow(outputDeltas[i],2);
            }
            norms[4] += Math.sqrt(temp);*/
        }

        for (int i = 0;i<layer1T.length;i++) {
            norms[0] += Math.pow(layer1T[i],2);
        }
        norms[0] = Math.sqrt(norms[0]);

        for (int i = 0;i<layer2T.length;i++) {
            norms[1] += Math.pow(layer2T[i],2);
        }
        norms[1] = Math.sqrt(norms[1]);

        for (int i = 0;i<layer3T.length;i++) {
            norms[2] += Math.pow(layer3T[i],2);
        }
        norms[2] = Math.sqrt(norms[2]);

        for (int i = 0;i<layer4T.length;i++) {
            norms[3] += Math.pow(layer4T[i],2);
        }
        norms[3] = Math.sqrt(norms[3]);

        for (int i = 0;i<outputT.length;i++) {
            norms[4] += Math.pow(outputT[i],2);
        }
        norms[4] = Math.sqrt(norms[4]);

        double[] results = new double[7];
        results[0] = error;
        results[1] = energy;
        for (int i = 0;i<norms.length;i++) {
            results[i+2] = norms[i];
        }
        return results;
    }
}

