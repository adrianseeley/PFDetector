public class KPools
{
    public Random random;
    public int totalDimensionCount;
    public int dimensionsPerPool;
    public int classSamplesPerPool;
    public Dictionary<int, List<Sample>> sampleHistogram;
    public int totalClassCount;
    public int lowestClassCount;
    public List<KPool> kPools;

    
    public KPools(int dimensionPerPool, int classSamplesPerPool, int targetPoolCount, int growToPoolCount, List<Sample> trainSamples, List<Sample> validationSamples)
    {
        this.random = new Random();
        this.totalDimensionCount = trainSamples[0].input.Count;
        this.dimensionsPerPool = dimensionPerPool;
        this.classSamplesPerPool = classSamplesPerPool;
        this.sampleHistogram = CalculateHistogram(trainSamples);
        this.totalClassCount = sampleHistogram.Count;
        this.lowestClassCount = CalculateLowestClassCount(sampleHistogram);
        this.kPools = new List<KPool>();
        int[] allDimensions = Enumerable.Range(0, totalDimensionCount).ToArray();

        // safety check
        if (classSamplesPerPool > lowestClassCount)
        {
            throw new ArgumentException("classSamplesPerPool is greater than the minimum samples per class");
        }

        // grow and shrink until we hit the target size after shrinking
        while (kPools.Count < targetPoolCount)
        {
            // grow pool
            while (kPools.Count < growToPoolCount)
            {
                Console.Write($"\rGrowing: {kPools.Count}/{growToPoolCount}");

                // randomly select dimensions (without replacement)
                int[] dimensions = allDimensions.OrderBy(x => random.Next()).Take(totalDimensionCount).ToArray();

                // randomly select balanced samples (without replacement)
                List<Sample> poolSamples = new List<Sample>(classSamplesPerPool * totalClassCount);
                foreach (KeyValuePair<int, List<Sample>> sampleHistogramKVP in sampleHistogram)
                {
                    poolSamples.AddRange(sampleHistogramKVP.Value.OrderBy(x => random.Next()).Take(classSamplesPerPool));
                }

                // create a kpool
                kPools.Add(new KPool(dimensions, poolSamples));
            }
            Console.WriteLine($"\rGrowing: {kPools.Count}/{growToPoolCount}");

            // measure starting fitness
            float fitness = Fitness(validationSamples);
            Console.WriteLine($"Preshrink Fitness: {fitness}, Count: {kPools.Count}");

            // shrink pool
            bool shrunk = true;
            while (shrunk)
            {
                shrunk = false;
                for (int p = 0; p < kPools.Count; p++)
                {
                    Console.Write($"\rShrinking: {p}/{kPools.Count}, F: {fitness}");

                    // remove a pool
                    KPool removed = kPools[p];
                    kPools.RemoveAt(p);

                    // measure fitness
                    float shrunkFitness = Fitness(validationSamples);

                    // if fitness is better or same, keep shrunk state
                    if (shrunkFitness >= fitness)
                    {
                        fitness = shrunkFitness;
                        shrunk = true;
                        p--;
                    }
                    // otherwise, put it back
                    else
                    {
                        kPools.Insert(p, removed);
                    }
                }
            }
            Console.WriteLine($"Postshrink Fitness: {fitness}, Count: {kPools.Count}");
        }
        Console.WriteLine("Done");
    }

    private static Dictionary<int, List<Sample>> CalculateHistogram(List<Sample> samples)
    {
        Dictionary<int, List<Sample>> histogram = new Dictionary<int, List<Sample>>();
        foreach (Sample sample in samples)
        {
            if (!histogram.ContainsKey(sample.output))
            {
                histogram[sample.output] = new List<Sample>();
            }
            histogram[sample.output].Add(sample);
        }
        return histogram;
    }

    public static int CalculateLowestClassCount(Dictionary<int, List<Sample>> histogram)
    {
        int lowestClassCount = int.MaxValue;
        foreach (int key in histogram.Keys)
        {
            lowestClassCount = Math.Min(lowestClassCount, histogram[key].Count);
        }
        return lowestClassCount;
    }

    public float Fitness(List<Sample> testSamples)
    {
        int correct = 0;
        Parallel.ForEach(testSamples, testSample =>
        {
            int prediction = Predict(testSample.input);
            if (prediction == testSample.output)
            {
                Interlocked.Increment(ref correct);
            }
        });
        return (float)correct / (float)testSamples.Count;
    }

    public int Predict(List<float> input)
    {
        // gather votes
        int[] votes = new int[totalClassCount];
        for (int p = 0; p < kPools.Count; p++)
        {
            int vote = kPools[p].Predict(input);
            votes[vote]++;
        }

        // find the majority vote
        int prediction = -1;
        int maxVotes = -1;
        for (int i = 0; i < votes.Length; i++)
        {
            if (votes[i] > maxVotes)
            {
                prediction = i;
                maxVotes = votes[i];
            }
        }

        // return majority vote
        return prediction;
    }
}