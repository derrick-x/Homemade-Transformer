import java.io.*;
import java.util.*;

public class Tester {
    public static void main(String[] args) throws IOException {
        BufferedReader in = new BufferedReader(new FileReader("tokens.txt"));
        String[] vocab = new String[9974];
        for (int i = 0; i < 9974; i++) {
            vocab[i] = in.readLine();
        }
        Transformer.init();
        Tokenizer.Trie tokenTrie = Tokenizer.getTrie(vocab);
        List<Integer> tokens = Tokenizer.tokenize("Sherlock Holmes is a brilliant detective, and he can solve any case.", tokenTrie); //Replace with real-world content for full functionality
        int[][] training = {new int[tokens.size()]};
        for (int i = 0; i < tokens.size(); i++) {
            training[0][i] = tokens.get(i);
        }
        //int[][] temp = {{0, 1, 2, 3, 4, 5, 6}};
        Transformer.learn(training);
        FileWriter out = new FileWriter(new File("parameters.txt"));
        for (int i = 0; i < Transformer.VOCAB_COUNT; i++) {
            for (int j = 0; j < Transformer.DIM_EMBED; j++) {
                out.write(Transformer.embed[i][j] + " ");
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.LAYERS; i++) {
            for (int j = 0; j < Transformer.DIM_EMBED; j++) {
                for (int k = 0; k < Transformer.DIM_ATTENTION; k++) {
                    out.write(Transformer.queryMatrix[i][j][k] + " ");
                }
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.LAYERS; i++) {
            for (int j = 0; j < Transformer.DIM_EMBED; j++) {
                for (int k = 0; k < Transformer.DIM_ATTENTION; k++) {
                    out.write(Transformer.keyMatrix[i][j][k] + " ");
                }
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.LAYERS; i++) {
            for (int j = 0; j < Transformer.DIM_EMBED; j++) {
                for (int k = 0; k < Transformer.DIM_EMBED; k++) {
                    out.write(Transformer.valueMatrix[i][j][k] + " ");
                }
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.LAYERS; i++) {
            for (int j = 0; j < Transformer.DIM_EMBED; j++) {
                for (int k = 0; k < Transformer.DIM_MLP; k++) {
                    out.write(Transformer.upMatrix[i][j][k] + " ");
                }
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.LAYERS; i++) {
            for (int j = 0; j < Transformer.DIM_MLP; j++) {
                out.write(Transformer.bias[i][j] + " ");
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.LAYERS; i++) {
            for (int j = 0; j < Transformer.DIM_MLP; j++) {
                for (int k = 0; k < Transformer.DIM_EMBED; k++) {
                    out.write(Transformer.downMatrix[i][j][k] + " ");
                }
            }
        }
        out.write("\n");
        for (int i = 0; i < Transformer.DIM_EMBED; i++) {
            for (int j = 0; j < Transformer.VOCAB_COUNT; j++) {
                out.write(Transformer.unembed[i][j] + " ");
            }
        }
        out.write("\n");
        int[] context = new int[Transformer.CONTEXT_LEN];
        Arrays.fill(context, -1);
        context[context.length - 1] = 1541;
        Scanner scan = new Scanner(System.in);
        while (true) { 
            scan.nextLine();
            double[] probability = Transformer.predict(context, 1);
            int token = 0;
            for (int i = 1; i < probability.length; i++) {
                if (probability[i] > probability[token]) {
                    token = i;
                }
            }
            for (int i = 1; i < context.length; i++) {
                context[i - 1] = context[i];
            }
            context[context.length - 1] = token;
            for (int i = 0; i < context.length; i++) {
                System.out.print(context[i] < 0 ? "" : vocab[context[i]]);
            }
            System.out.println();
        }
    }
}
