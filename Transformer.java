import java.util.Arrays;

public class Transformer {
    static double[][] embed = {{1, 2}, {3, 4}};
    static double[][][] queryMatrix = {{{0.5, 1}, {-0.5, 1}}};
    static double[][][] keyMatrix = {{{1, 0.5}, {1, -0.5}}};
    static double[][][] valueMatrix = {{{1, 1}, {-0.5, 0.5}}};
    static double[][][] upMatrix = {{{-0.5, 1}, {0.5, 1}}};
    static double[][] bias = {{0, 0}};
    static double[][][] downMatrix = {{{0.5, -0.5}, {1, 1}}};
    static double[][] unembed = {{4, 3}, {2, 1}};
    static final int VOCAB_COUNT = 9974;
    static final int LAYERS = 2;
    static final int DIM_EMBED = 4;
    static final int DIM_ATTENTION = 3;
    static final int DIM_MLP = 5;
    static final int CONTEXT_LEN = 6;
    static final double LEARN_RATE = 0.00002;
    static class ParamLog {
        double[][][] embedVectors;
        double[][][] queries;
        double[][][] keys;
        double[][][] scaledDots;
        double[][][] values;
        double[][][] preMLP;
        double[][][] activations;
        double[][] finalVectors;
        public ParamLog() {
            embedVectors = new double[LAYERS][CONTEXT_LEN][DIM_EMBED];
            queries = new double[LAYERS][CONTEXT_LEN][DIM_ATTENTION];
            keys = new double[LAYERS][CONTEXT_LEN][DIM_ATTENTION];
            scaledDots = new double[LAYERS][CONTEXT_LEN][CONTEXT_LEN];
            values = new double[LAYERS][CONTEXT_LEN][DIM_EMBED];
            preMLP = new double[LAYERS][CONTEXT_LEN][DIM_EMBED];
            activations = new double[LAYERS][CONTEXT_LEN][DIM_MLP];
            finalVectors = new double[CONTEXT_LEN][DIM_EMBED];
        }
    }
    static ParamLog params;

    public static void init() {
        embed = new double[VOCAB_COUNT][DIM_EMBED]; //embed dimensions
        for (int i = 0; i < VOCAB_COUNT; i++) {
            for (int j = 0; j < DIM_EMBED; j++) {
                embed[i][j] = Math.random() * 0.2 - 0.1;
            }
        }
        queryMatrix = new double[LAYERS][DIM_EMBED][DIM_ATTENTION]; //query matrix dimensionw
        for (int i = 0; i < LAYERS; i++) {
            for (int j = 0; j < DIM_EMBED; j++) {
                for (int k = 0; k < DIM_ATTENTION; k++) {
                    queryMatrix[i][j][k] = Math.random() * 0.2 - 0.1;
                }
            }
        }
        keyMatrix = new double[LAYERS][DIM_EMBED][DIM_ATTENTION]; //key matrix dimensions
        for (int i = 0; i < LAYERS; i++) {
            for (int j = 0; j < DIM_EMBED; j++) {
                for (int k = 0; k < DIM_ATTENTION; k++) {
                    keyMatrix[i][j][k] = Math.random() * 0.2 - 0.1;
                }
            }
        }
        valueMatrix = new double[LAYERS][DIM_EMBED][DIM_EMBED]; //value matrix 1 dimensions
        for (int i = 0; i < LAYERS; i++) {
            for (int j = 0; j < DIM_EMBED; j++) {
                for (int k = 0; k < DIM_EMBED; k++) {
                    valueMatrix[i][j][k] = Math.random() * 0.2 - 0.1;
                }
            }
        }
        upMatrix = new double[LAYERS][DIM_EMBED][DIM_MLP]; //up matrix dimensions
        for (int i = 0; i < LAYERS; i++) {
            for (int j = 0; j < DIM_EMBED; j++) {
                for (int k = 0; k < DIM_MLP; k++) {
                    upMatrix[i][j][k] = Math.random() * 0.2 - 0.1;
                }
            }
        }
        bias = new double[LAYERS][DIM_MLP]; //bias dimensions
        for (int i = 0; i < LAYERS; i++) {
            for (int j = 0; j < DIM_MLP; j++) {
                bias[i][j] = Math.random() * 0.2 - 0.1;
            }
        }
        downMatrix = new double[LAYERS][DIM_MLP][DIM_EMBED]; //down matrix dimensions
        for (int i = 0; i < LAYERS; i++) {
            for (int j = 0; j < DIM_MLP; j++) {
                for (int k = 0; k < DIM_EMBED; k++) {
                    downMatrix[i][j][k] = Math.random() * 0.4 - 0.2;
                }
            }
        }
        unembed = new double[DIM_EMBED][VOCAB_COUNT]; //unembed dimensions
        for (int i = 0; i < DIM_EMBED; i++) {
            for (int j = 0; j < VOCAB_COUNT; j++) {
                unembed[i][j] = Math.random() * 0.4 - 0.2;
            }
        }
    }

    public static double[] predict(int[] context, double temp) {
        params = new ParamLog();
        //Embed
        double[][] embedVectors = new double[CONTEXT_LEN][DIM_EMBED];
        for (int i = 0; i < CONTEXT_LEN; i++) {
            for (int j = 0; j < DIM_EMBED; j++) {
                embedVectors[i][j] = (context[i] < 0 ? 0 : embed[context[i]][j]) + ((j % 2 == 0) ? Math.sin(i / Math.pow(DIM_EMBED,  i / (double) DIM_EMBED)) : Math.cos(i / Math.pow(DIM_EMBED,  i / (double) DIM_EMBED)));
            }
        }
        for (int l = 0; l < LAYERS; l++) {
            //Attention
            arraycopy2D(embedVectors, params.embedVectors[l]);
            double[][] queries = matMul(queryMatrix[l], embedVectors);
            arraycopy2D(queries, params.queries[l]);
            double[][] keys = matMul(keyMatrix[l], embedVectors);
            arraycopy2D(keys, params.keys[l]);
            double[][] dots = matMul(transpose(keys), queries);
            double[][] scaledDots = new double[CONTEXT_LEN][CONTEXT_LEN];
            for (int i = 0; i < CONTEXT_LEN; i++) {
                scaledDots[i] = softmax(dots[i], 1);
                for (int j = 0; j < CONTEXT_LEN; j++) {
                    scaledDots[i][j] /= Math.sqrt(DIM_ATTENTION);
                }
            }
            arraycopy2D(scaledDots, params.scaledDots[l]);
            double[][] values = matMul(valueMatrix[l], embedVectors);
            arraycopy2D(values, params.values[l]);
            double[][] attention = matMul(values, scaledDots);
            for (int i = 0; i < CONTEXT_LEN; i++) {
                for (int j = 0; j < DIM_EMBED; j++) {
                    embedVectors[i][j] += attention[i][j];
                }
            }
            //MLP
            arraycopy2D(embedVectors, params.preMLP[l]);
            for (int i = 0; i < CONTEXT_LEN; i++) {
                double[] preActivations = matMul(upMatrix[l], embedVectors[i]);
                double[] activations = new double[DIM_MLP];
                for (int j = 0; j < DIM_MLP; j++) {
                    activations[j] = leakyReLu(preActivations[j] + bias[l][j]);
                }
                System.arraycopy(activations, 0, params.activations[l][i], 0, DIM_MLP);
                double[] postActivations = matMul(downMatrix[l], activations);
                for (int j = 0; j < DIM_EMBED; j++) {
                    embedVectors[i][j] += postActivations[j];
                }
            }
        }
        //Unembed
        arraycopy2D(embedVectors, params.finalVectors);
        double[] prediction = softmax(matMul(unembed, embedVectors[CONTEXT_LEN - 1]), temp);
        return prediction;
    }

    public static void learn(int[][] testData) {
        for (int t = 0; t < 250000; t++) {
            double[][] _embed = new double[VOCAB_COUNT][DIM_EMBED];
            double[][][] _queryMatrix = new double[LAYERS][DIM_EMBED][DIM_ATTENTION];
            double[][][] _keyMatrix = new double[LAYERS][DIM_EMBED][DIM_ATTENTION];
            double[][][] _valueMatrix = new double[LAYERS][DIM_EMBED][DIM_EMBED];
            double[][][] _upMatrix = new double[LAYERS][DIM_EMBED][DIM_MLP];
            double[][] _bias = new double[LAYERS][DIM_MLP];
            double[][][] _downMatrix = new double[LAYERS][DIM_MLP][DIM_EMBED];
            double[][] _unembed = new double[DIM_EMBED][VOCAB_COUNT];
            double cost = 0;
            for (int[] testCase : testData) {
                int[] context = new int[CONTEXT_LEN];
                Arrays.fill(context, -1);
                for (int p = 1; p < testCase.length; p++) {
                    for (int j = 1; j < CONTEXT_LEN; j++) {
                        context[j - 1] = context[j];
                    }
                    context[CONTEXT_LEN - 1] = testCase[p - 1];
                    double[] probability = predict(context, 1.0);
                    cost -= Math.log(probability[testCase[p]]);
                    double[][] _embedVectors = new double[CONTEXT_LEN][DIM_EMBED];
                    //Unembed gradient
                    double[] _output = new double[VOCAB_COUNT];
                    System.arraycopy(probability, 0, _output, 0, VOCAB_COUNT);
                    _output[testCase[p]] = probability[testCase[p]] - 1;
                    for (int i = 0; i < VOCAB_COUNT; i++) {
                        for (int j = 0; j < DIM_EMBED; j++) {
                            _unembed[j][i] += params.finalVectors[CONTEXT_LEN - 1][j] * _output[i];
                        }
                    }
                    for (int i = 0; i < VOCAB_COUNT; i++) {
                        for (int j = 0; j < DIM_EMBED; j++) {
                            _embedVectors[CONTEXT_LEN - 1][j] += unembed[j][i] * _output[i];
                        }
                    }
                    for (int l = LAYERS - 1; l >= 0; l--) {
                        double[][] _prevVectors = new double[CONTEXT_LEN][DIM_EMBED];
                        //MLP gradient
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_MLP; k++) {
                                    _downMatrix[l][k][i] += params.activations[l][j][k] * _embedVectors[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                _prevVectors[j][i] += _embedVectors[j][i];
                            }
                        }
                        double[][] _activations = new double[CONTEXT_LEN][DIM_MLP];
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_MLP; k++) {
                                    _activations[j][k] += downMatrix[l][k][i] * _embedVectors[j][i];
                                }
                            }
                        }
                        double[][] _preActivations = new double[CONTEXT_LEN][DIM_MLP];
                        for (int i = 0; i < DIM_MLP; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                _preActivations[j][i] = _leakyReLu(params.activations[l][j][i]) * _activations[j][i];
                            }
                        }
                        for (int i = 0; i < DIM_MLP; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                _bias[l][i] += _leakyReLu(params.activations[l][j][i]) * _activations[j][i];
                            }
                        }
                        for (int i = 0; i < DIM_MLP; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _upMatrix[l][k][i] += params.preMLP[l][j][k] * _activations[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_MLP; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _prevVectors[j][k] += upMatrix[l][k][i] * _preActivations[j][i];
                                }
                            }
                        }
                        _embedVectors = _prevVectors;
                        //Attention gradient
                        _prevVectors = new double[CONTEXT_LEN][DIM_EMBED];
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                _prevVectors[j][i] = _embedVectors[j][i];
                            }
                        }
                        double[][] _values = new double[CONTEXT_LEN][DIM_EMBED];
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < CONTEXT_LEN; k++) {
                                    _values[k][i] += params.scaledDots[l][j][k] * _embedVectors[j][i];
                                }
                            }
                        }
                        double[][] _scaledDots = new double[CONTEXT_LEN][CONTEXT_LEN];
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < CONTEXT_LEN; k++) {
                                    _scaledDots[j][k] += params.values[l][k][i] * _embedVectors[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _valueMatrix[l][k][i] += params.embedVectors[l][j][k] * _values[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_EMBED; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _prevVectors[j][k] += valueMatrix[l][k][i] * _values[j][i];
                                }
                            }
                        }
                        double[][] _dots = new double[CONTEXT_LEN][CONTEXT_LEN];
                        for (int i = 0; i < CONTEXT_LEN; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < CONTEXT_LEN; k++) {
                                    if (i == k) {
                                        _dots[j][k] += params.scaledDots[l][j][i] * (1 - params.scaledDots[l][j][i]) * _scaledDots[j][i] / Math.sqrt(DIM_ATTENTION);
                                    }
                                    else {
                                        _dots[j][k] -= params.scaledDots[l][j][i] * params.scaledDots[l][j][k] * _scaledDots[j][i] / Math.sqrt(DIM_ATTENTION);
                                    }
                                }
                            }
                        }
                        double[][] _queries = new double[CONTEXT_LEN][DIM_ATTENTION];
                        for (int i = 0; i < CONTEXT_LEN; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_ATTENTION; k++) {
                                    _queries[j][k] += params.keys[l][i][k] * _dots[j][i];
                                }
                            }
                        }
                        double[][] _keys = new double[CONTEXT_LEN][DIM_ATTENTION];
                        for (int i = 0; i < CONTEXT_LEN; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_ATTENTION; k++) {
                                    _keys[i][k] += params.queries[l][i][k] * _dots[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_ATTENTION; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _queryMatrix[l][k][i] += params.embedVectors[l][j][k] * _queries[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_ATTENTION; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _prevVectors[j][k] += queryMatrix[l][k][i] * _queries[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_ATTENTION; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _keyMatrix[l][k][i] += params.embedVectors[l][j][k] * _keys[j][i];
                                }
                            }
                        }
                        for (int i = 0; i < DIM_ATTENTION; i++) {
                            for (int j = 0; j < CONTEXT_LEN; j++) {
                                for (int k = 0; k < DIM_EMBED; k++) {
                                    _prevVectors[j][k] += keyMatrix[l][k][i] * _queries[j][i];
                                }
                            }
                        }
                        _embedVectors = _prevVectors;
                    }
                    //Embed gradient
                    for (int i = 0; i < DIM_EMBED; i++) {
                        for (int j = 0; j < CONTEXT_LEN; j++) {
                            if (context[j] >= 0) {
                                _embed[context[j]][i] += _embedVectors[j][i];
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < VOCAB_COUNT; i++) {
                for (int j = 0; j < DIM_EMBED; j++) {
                    embed[i][j] -= _embed[i][j] * LEARN_RATE;
                }
            }
            for (int i = 0; i < LAYERS; i++) {
                for (int j = 0; j < DIM_EMBED; j++) {
                    for (int k = 0; k < DIM_ATTENTION; k++) {
                        queryMatrix[i][j][k] -= _queryMatrix[i][j][k] * LEARN_RATE;
                    }
                }
            }
            for (int i = 0; i < LAYERS; i++) {
                for (int j = 0; j < DIM_EMBED; j++) {
                    for (int k = 0; k < DIM_ATTENTION; k++) {
                        keyMatrix[i][j][k] -= _keyMatrix[i][j][k] * LEARN_RATE;
                    }
                }
            }
            for (int i = 0; i < LAYERS; i++) {
                for (int j = 0; j < DIM_EMBED; j++) {
                    for (int k = 0; k < DIM_EMBED; k++) {
                        valueMatrix[i][j][k] -= _valueMatrix[i][j][k] * LEARN_RATE;
                    }
                }
            }
            for (int i = 0; i < LAYERS; i++) {
                for (int j = 0; j < DIM_EMBED; j++) {
                    for (int k = 0; k < DIM_MLP; k++) {
                        upMatrix[i][j][k] -= _upMatrix[i][j][k] * LEARN_RATE;
                    }
                }
            }
            for (int i = 0; i < LAYERS; i++) {
                for (int j = 0; j < DIM_MLP; j++) {
                    bias[i][j] -= _bias[i][j] * LEARN_RATE;
                }
            }
            for (int i = 0; i < LAYERS; i++) {
                for (int j = 0; j < DIM_MLP; j++) {
                    for (int k = 0; k < DIM_EMBED; k++) {
                        downMatrix[i][j][k] -= _downMatrix[i][j][k] * LEARN_RATE;
                    }
                }
            }
            for (int i = 0; i < DIM_EMBED; i++) {
                for (int j = 0; j < VOCAB_COUNT; j++) {
                    unembed[i][j] -= _unembed[i][j] * LEARN_RATE;
                }
            }
            System.out.println(t + ": " + cost);
            if (Double.isNaN(cost)) {
                System.out.println(t);
                System.exit(0);
            }
        }
    }

    public static double[][] matMul(double[][] a, double[][] b) {
        double[][] product = new double[b.length][a[0].length];
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a.length; k++) {
                    product[i][j] += a[k][j] * b[i][k];
                }
            }
        }
        return product;
    }

    public static double[] matMul(double[][] a, double[] b) {
        double[] product = new double[a[0].length];
        for (int i = 0; i < a[0].length; i++) {
            for (int j = 0; j < a.length; j++) {
                product[i] += a[j][i] * b[j];
            }
        }
        return product;
    }

    public static double[] softmax(double[] input, double temp) {
        double sum = 0;
        for (double value : input) {
            sum += Math.pow(Math.E / temp, value);
        }
        double[] softmaxed = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            softmaxed[i] = Math.pow(Math.E, input[i] / temp) / sum;
        }
        return softmaxed;
    }

    public static double[][] transpose(double[][] a) {
        double[][] transposed = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                transposed[j][i] = a[i][j];
            }
        }
        return transposed;
    }

    public static double leakyReLu(double input) {
        return input > 0 ? input : input * 0.01;
    }

    public static double _leakyReLu(double input) {
        return input > 0 ? 1 : 0.01;
    }

    public static void arraycopy2D(double[][] src, double[][] dest) {
        for (int i = 0; i < src.length; i++) {
            System.arraycopy(src[i], 0, dest[i], 0, src[i].length);
        }
    }
}
