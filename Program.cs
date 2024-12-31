using System.Text;

public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            List<float> input = new List<float>();
            for (int i = 1; i < parts.Length; i++)
            {
                input.Add(float.Parse(parts[i]));
            }
            samples.Add(new Sample(input, labelInt));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static void Main()
    {
        Random random = new Random();
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: -1);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        int totalClassCount = 10;
        int startingComponentCount = mnistTrain[0].input.Count;

        // find homogenous
        List<int> homogenousComponents = new List<int>();
        for (int i = 0; i < startingComponentCount; i++)
        {
            float zeroValue = mnistTrain[0].input[i];
            bool homogeneous = true;
            for (int j = 1; j < mnistTrain.Count; j++)
            {
                if (mnistTrain[j].input[i] != zeroValue)
                {
                    homogeneous = false;
                    break;
                }
            }
            if (homogeneous)
            {
                homogenousComponents.Add(i);
            }
        }

        // remove homogenous from train and test
        foreach(Sample sample in mnistTrain)
        {
            for (int i = homogenousComponents.Count - 1; i >= 0; i--)
            {
                sample.input.RemoveAt(homogenousComponents[i]);
            }
        }
        foreach (Sample sample in mnistTest)
        {
            for (int i = homogenousComponents.Count - 1; i >= 0; i--)
            {
                sample.input.RemoveAt(homogenousComponents[i]);
            }
        }

        PerfectFeatureDetector pfd = new PerfectFeatureDetector(totalClassCount, conditionCount: 1, minimumEvidence: 1, mnistTrain);

        int correct = mnistTest.Count(sample => pfd.Predict(sample.input) == sample.output);
        float testFitness = (float)correct / (float)mnistTest.Count;

        Console.WriteLine("Test Fitness: " + testFitness);
        Console.ReadLine();
    }
}

