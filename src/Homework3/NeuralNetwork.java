package Homework3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;



public class NeuralNetwork {

    private double eta;

    private int hiddenLayers;

    private double[] outputThresholds;

    private double[] layer2Thresholds;

    private double[] layer1Thresholds;

    private double[][] outputWeights;

    private double[][] layer2Weights;

    private double[][] layer1Weights;

    private double[] layer2Neurons;

    private double[] layer1Neurons;

    private double[][] outputTrain;

    private double[][] outputValid;

    private double[][] outputTest;

    private double[][] xTrain;

    private double[][] tTrain;

    List<Integer> shuffledPatterns;

    private double[][] xValid;

    private double[][] tValid;

    private double[][] xTest;

    private double[][] tTest;

    private Random rand = new Random();











    public NeuralNetwork(double eta, int inputSize, int outputSize, int[] hiddenSize, double[][] xDataT, double[][] tDataT, double[][] xDataV, double[][] tDataV, double[][] xDataTest, double[][] tDataTest) {

        this.eta = eta;
        hiddenLayers = hiddenSize.length;
        setTrainingData(xDataT, tDataT);
        setValidationData(xDataV, tDataV);
        setTestData(xDataTest, tDataTest);
        outputThresholds = new double[outputSize];
        if (hiddenLayers > 1) {
            layer2Thresholds = new double[hiddenSize[1]];
            layer2Neurons = new double[hiddenSize[1]];
            outputWeights = new double[outputSize][hiddenSize[1]];
            for (int i = 0; i < outputWeights.length; i++) {
                for (int j = 0; j < outputWeights[i].length; j++) {
                    outputWeights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(outputWeights[i].length));
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
        } else if (hiddenLayers > 0) {
            layer1Thresholds = new double[hiddenSize[0]];
            layer1Neurons = new double[hiddenSize[0]];
            outputWeights = new double[outputSize][hiddenSize[0]];
            for (int i = 0; i < outputWeights.length; i++) {
                for (int j = 0; j < outputWeights[i].length; j++) {
                    outputWeights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(outputWeights[i].length));
                }
            }
            layer1Weights = new double[hiddenSize[0]][inputSize];
            for (int i = 0; i < layer1Weights.length; i++) {
                for (int j = 0; j < layer1Weights[i].length; j++) {
                    layer1Weights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(layer1Weights[i].length));
                }
            }
        } else {
            outputWeights = new double[outputSize][inputSize];
            for (int i = 0; i < outputWeights.length; i++) {
                for (int j = 0; j < outputWeights[i].length; j++) {
                    outputWeights[i][j] = rand.nextGaussian() * (1 / Math.sqrt(outputWeights[i].length));
                }
            }
        }
    }


    private void propagateNetwork (int mu, double[][] input) {
        if (hiddenLayers > 1) {
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
            for (int i = 0; i < outputTrain[mu].length; i++) {
                double b = -outputThresholds[i];
                for (int j = 0; j < layer2Neurons.length; j++) {
                    b += outputWeights[i][j] * layer2Neurons[j];
                }
                outputTrain[mu][i] = sigmoid(b);
            }
        } else if (hiddenLayers > 0) {
            for (int i = 0; i < layer1Neurons.length; i++) {
                double b = -layer1Thresholds[i];
                for (int j = 0; j < input[mu].length; j++) {
                    b += layer1Weights[i][j] * input[mu][j];
                }
                layer1Neurons[i] = sigmoid(b);
            }
            for (int i = 0; i < outputTrain[mu].length; i++) {
                double b = -outputThresholds[i];
                for (int j = 0; j < layer1Neurons.length; j++) {
                    b += outputWeights[i][j] * layer1Neurons[j];
                }
                outputTrain[mu][i] = sigmoid(b);
            }
        } else {
            for (int i = 0; i < outputTrain[mu].length; i++) {
                double b = -outputThresholds[i];
                for (int j = 0; j < input[mu].length; j++) {
                    b += outputWeights[i][j] * input[mu][j];
                }
                outputTrain[mu][i] = sigmoid(b);
            }
        }
    }

    private void propagateNetworkValidation () {
        for (int mu = 0;mu<xValid.length;mu++) {
            if (hiddenLayers > 1) {
                for (int i = 0; i < layer1Neurons.length; i++) {
                    double b = -layer1Thresholds[i];
                    for (int j = 0; j < xValid[mu].length; j++) {
                        b += layer1Weights[i][j] * xValid[mu][j];
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
                for (int i = 0; i < outputTrain[mu].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < layer2Neurons.length; j++) {
                        b += outputWeights[i][j] * layer2Neurons[j];
                    }
                    outputValid[mu][i] = sigmoid(b);
                }
            } else if (hiddenLayers > 0) {
                for (int i = 0; i < layer1Neurons.length; i++) {
                    double b = -layer1Thresholds[i];
                    for (int j = 0; j < xValid[mu].length; j++) {
                        b += layer1Weights[i][j] * xValid[mu][j];
                    }
                    layer1Neurons[i] = sigmoid(b);
                }
                for (int i = 0; i < outputTrain[mu].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < layer1Neurons.length; j++) {
                        b += outputWeights[i][j] * layer1Neurons[j];
                    }
                    outputValid[mu][i] = sigmoid(b);
                }
            } else {
                for (int i = 0; i < outputTrain[mu].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < xValid[mu].length; j++) {
                        b += outputWeights[i][j] * xValid[mu][j];
                    }
                    outputValid[mu][i] = sigmoid(b);
                }
            }
        }
    }

    private void propagateNetworkTest () {
        for (int mu = 0;mu<xTest.length;mu++) {
            if (hiddenLayers > 1) {
                for (int i = 0; i < layer1Neurons.length; i++) {
                    double b = -layer1Thresholds[i];
                    for (int j = 0; j < xTest[mu].length; j++) {
                        b += layer1Weights[i][j] * xTest[mu][j];
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
                for (int i = 0; i < outputTrain[mu].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < layer2Neurons.length; j++) {
                        b += outputWeights[i][j] * layer2Neurons[j];
                    }
                    outputTest[mu][i] = sigmoid(b);
                }
            } else if (hiddenLayers > 0) {
                for (int i = 0; i < layer1Neurons.length; i++) {
                    double b = -layer1Thresholds[i];
                    for (int j = 0; j < xTest[mu].length; j++) {
                        b += layer1Weights[i][j] * xTest[mu][j];
                    }
                    layer1Neurons[i] = sigmoid(b);
                }
                for (int i = 0; i < outputTrain[mu].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < layer1Neurons.length; j++) {
                        b += outputWeights[i][j] * layer1Neurons[j];
                    }
                    outputTest[mu][i] = sigmoid(b);
                }
            } else {
                for (int i = 0; i < outputTrain[mu].length; i++) {
                    double b = -outputThresholds[i];
                    for (int j = 0; j < xTest[mu].length; j++) {
                        b += outputWeights[i][j] * xTest[mu][j];
                    }
                    outputTest[mu][i] = sigmoid(b);
                }
            }
        }
    }

    public void updateNetwork() {
        Collections.shuffle(shuffledPatterns);
        if (hiddenLayers > 1) {

            for (int mu = 0; mu < xTrain.length-10;mu+=10) {
                double outputW[][] = new double[outputTrain[mu].length][layer2Neurons.length];
                double layer2W[][] = new double[layer2Neurons.length][layer1Neurons.length];
                double layer1W[][] = new double[layer1Neurons.length][xTrain[mu].length];
                double outputT[] = new double[outputTrain[mu].length];
                double layer2T[] = new double[layer2Neurons.length];
                double layer1T[] = new double[layer1Neurons.length];
                for (int mb=mu;mb<mu+10 ;mb++) {
                    int pattern = shuffledPatterns.get(mb);
                    propagateNetwork(pattern,xTrain);
                    //Compute b
                    double[] outputDeltas = new double[outputTrain[pattern].length];
                    for (int i = 0; i < outputTrain[pattern].length; i++) {
                        double b = -outputThresholds[i];
                        for (int j = 0; j < layer2Neurons.length; j++) {
                            b += outputWeights[i][j] * layer2Neurons[j];
                        }
                        outputDeltas[i] = sigmoidDelta(b) * (tTrain[pattern][i] - outputTrain[pattern][i]);
                    }
                    double[] layer2Deltas = new double[layer2Neurons.length];
                    for (int i = 0; i < layer2Neurons.length; i++) {
                        double b = -layer2Thresholds[i];
                        for (int j = 0; j < layer1Neurons.length; j++) {
                            b += layer2Weights[i][j] * layer1Neurons[j];
                        }
                        for (int k = 0; k < outputDeltas.length; k++) {
                            layer2Deltas[i] += outputDeltas[k] * outputWeights[k][i] * sigmoidDelta(b);
                        }
                    }
                    double[] layer1Deltas = new double[layer1Neurons.length];
                    for (int i = 0; i < layer1Neurons.length; i++) {
                        double b = -layer1Thresholds[i];
                        for (int j = 0; j < xTrain[pattern].length; j++) {
                            b += layer1Weights[i][j] * xTrain[pattern][j];
                        }
                        for (int k = 0; k < outputDeltas.length; k++) {
                            layer1Deltas[i] += layer2Deltas[k] * layer2Weights[k][i] * sigmoidDelta(b);
                        }
                    }

                    for (int i = 0; i < outputWeights.length; i++) {
                        for (int j = 0; j < outputWeights[i].length;j++) {

                            outputW[i][j] += outputDeltas[i] * layer2Neurons[j];
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
                    for (int i = 0; i < layer2Thresholds.length; i++) {
                        layer2T[i] += layer2Deltas[i];
                    }
                    for (int i = 0; i < layer1Thresholds.length; i++) {
                        layer1T[i] += layer1Deltas[i];
                    }



                }
                for (int i = 0; i < outputWeights.length; i++) {
                    for (int j = 0; j < outputWeights[i].length;j++) {

                        outputWeights[i][j] += eta * outputW[i][j];
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
                for (int i = 0; i < layer2Thresholds.length; i++) {
                    layer2Thresholds[i] -= eta * layer2T[i];
                }
                for (int i = 0; i < layer1Thresholds.length; i++) {
                    layer1Thresholds[i] -= eta * layer1T[i];
                }

            }

        } else if (hiddenLayers > 0) {
            for (int mu = 0; mu < xTrain.length-10;mu+=10) {
                double outputW[][] = new double[outputTrain[mu].length][layer1Neurons.length];
                double layer1W[][] = new double[layer1Neurons.length][xTrain[mu].length];
                double outputT[] = new double[outputTrain[mu].length];
                double layer1T[] = new double[layer1Neurons.length];
                for (int mb = mu; mb < mu+10; mb++) {
                    int pattern = shuffledPatterns.get(mb);
                    propagateNetwork(pattern, xTrain);
                    double[] outputDeltas = new double[outputTrain[pattern].length];
                    for (int i = 0; i < outputTrain[pattern].length; i++) {
                        double b = -outputThresholds[i];
                        for (int j = 0; j < layer1Neurons.length; j++) {
                            b += outputWeights[i][j] * layer1Neurons[j];
                        }
                        double gprime  =sigmoidDelta(b);
                        outputDeltas[i] += sigmoidDelta(b) * (tTrain[pattern][i] - outputTrain[pattern][i]);
                    }
                    double[] layer1Deltas = new double[layer1Neurons.length];
                    for (int i = 0; i < layer1Neurons.length; i++) {
                        double b = -layer1Thresholds[i];
                        for (int j = 0; j < xTrain[pattern].length; j++) {
                            b += layer1Weights[i][j] * xTrain[pattern][j];
                        }

                        for (int k = 0; k < outputDeltas.length; k++) {
                            double gprime  =sigmoidDelta(b);
                            layer1Deltas[i] += outputDeltas[k] * outputWeights[k][i] * sigmoidDelta(b);
                        }
                    }

                    for (int i = 0; i < outputWeights.length; i++) {
                        for (int j = 0; j < outputWeights[i].length; j++) {

                            outputW[i][j] += outputDeltas[i] * layer1Neurons[j];
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

                    for (int i = 0; i < layer1Thresholds.length; i++) {
                        layer1T[i] += layer1Deltas[i];
                    }


                }

                for (int i = 0; i < outputWeights.length; i++) {
                    for (int j = 0; j < outputWeights[i].length; j++) {

                        outputWeights[i][j] += eta * outputW[i][j];
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

                for (int i = 0; i < layer1Thresholds.length; i++) {
                    layer1Thresholds[i] -= eta * layer1T[i];
                }
            }
        } else {
            for (int mu = 0; mu < xTrain.length-10;mu+=10) {
                double outputW[][] = new double[outputTrain[mu].length][xTrain[mu].length];
                double outputT[] = new double[outputTrain[mu].length];
                for (int mb = mu; mb < mu+10; mb++) {
                    int pattern = shuffledPatterns.get(mb);
                    propagateNetwork(pattern, xTrain);
                    double[] outputDeltas = new double[outputTrain[pattern].length];
                    for (int i = 0; i < outputTrain[pattern].length; i++) {
                        double b = -outputThresholds[i];
                        for (int j = 0; j < xTrain[pattern].length; j++) {
                            b += outputWeights[i][j] * xTrain[pattern][j];
                        }
                        for (int k = 0; k < outputDeltas.length; k++) {
                            double gprime  =sigmoidDelta(b);
                            double t = tTrain[pattern][i];
                            double o = outputTrain[pattern][i];
                            outputDeltas[i] += sigmoidDelta(b) * (tTrain[pattern][i] - outputTrain[pattern][i]);
                        }
                    }

                    for (int i = 0; i < outputWeights.length; i++) {
                        for (int j = 0; j < outputWeights[i].length; j++) {

                            outputW[i][j] += outputDeltas[i] * xTrain[pattern][j];
                        }
                    }

                    for (int i = 0; i < outputThresholds.length; i++) {
                        outputT[i] += outputDeltas[i];
                    }

                }
                for (int i = 0; i < outputWeights.length; i++) {
                    for (int j = 0; j < outputWeights[i].length; j++) {

                        outputWeights[i][j] += eta * outputW[i][j];

                    }
                }

                for (int i = 0; i < outputThresholds.length; i++) {
                    outputThresholds[i] -= eta * outputT[i];
                }
            }
        }
    }

    public double[] getClassificationErrors () {
        double[] errors = new double[3  ];
        double error = 0;
        for (int mu = 0; mu < tTrain.length;mu++) {
            propagateNetwork(mu, xTrain);
        }
        for (int mu = 0; mu < outputTrain.length;mu++) {
            for (int i = 0; i < outputTrain[mu].length; i++) {
                error += Math.abs(tTrain[mu][i]-largest(outputTrain[mu],i));
            }
        }

        errors[0] =  error / (2 * xTrain.length);

        error = 0;
        propagateNetworkValidation();
        for (int mu = 0; mu < tValid.length;mu++) {
            for (int i = 0; i < tValid[mu].length; i++) {
                error += Math.abs(tValid[mu][i]-largest(outputValid[mu],i));
            }
        }

        errors[1] =  error / (2 * xValid.length);


        error = 0;
        propagateNetworkTest();
        for (int mu = 0; mu < tTest.length;mu++) {
            for (int i = 0; i < tTest[mu].length; i++) {
                error += Math.abs(tTest[mu][i]-largest(outputTest[mu],i));
            }
        }

        errors[2] =  error / (2 * xTest.length);

        return errors;

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

    private void setValidationData(double[][] xData, double[][] tData) {
        xValid = new double[xData.length][xData[0].length];
        outputValid = new double[tData.length][tData[0].length];
        for (int i = 0; i < xData.length; i++) {
            for (int j = 0; j < xData[i].length; j++) {
                xValid[i][j] = xData[i][j];
            }
        }

        tValid = new double[tData.length][tData[0].length];
        for (int i = 0; i < tData.length; i++) {
            for (int j = 0; j < tData[i].length; j++) {
                tValid[i][j] = tData[i][j];
            }
        }
    }

    private void setTestData(double[][] xData, double[][] tData) {
        xTest = new double[xData.length][xData[0].length];
        outputTest = new double[tData.length][tData[0].length];
        for (int i = 0; i < xData.length; i++) {
            for (int j = 0; j < xData[i].length; j++) {
                xTest[i][j] = xData[i][j];
            }
        }

        tTest = new double[tData.length][tData[0].length];
        for (int i = 0; i < tData.length; i++) {
            for (int j = 0; j < tData[i].length; j++) {
                tTest[i][j] = tData[i][j];
            }
        }
    }

    private double sigmoid(double b) {
        return (Math.pow((1 + Math.exp(-b)),-1));
    }

    private double sigmoidDelta(double b) {
        return (sigmoid(b) - Math.pow(sigmoid(b), 2));
    }

    private int largest(double[] o, int i) {
        for (int j = 0;j<o.length;j++) {
            if (i!=j) {
                if (o[i]<=o[j])
                    return 0;
            }
        }
        return 1;

    }
}
