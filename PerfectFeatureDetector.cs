using System.Text;

public class PerfectFeatureDetector
{
    public int totalClassCount;
    public List<ConditionalPrediction> perfectFeatures;
    public int[] distribution;

    public PerfectFeatureDetector(int totalClassCount, int conditionCount, int minimumEvidence, List<Sample> samples)
    {
        this.totalClassCount = totalClassCount;
        this.perfectFeatures = new List<ConditionalPrediction>();
        this.distribution = new int[totalClassCount];

        // get all component values
        List<List<float>> allComponentValues = GetAllComponentValues(samples);
        
        // create starting iterator
        List<Condition> conditions = new List<Condition>();
        for (int i = 0; i < conditionCount; i++)
        {
            conditions.Add(new Condition(0, 0, allComponentValues[0][0], false));
        }
        ConditionalPrediction iterator = new ConditionalPrediction(conditions);

        // create vote holder
        int[] votes = new int[totalClassCount];

        // loop
        int log = 0;
        for (; ;)
        {
            log++;
            if (log % 10 == 0)
            {
                Console.Write($"\r{iterator.GetProgressString(allComponentValues)} -> [{string.Join(',', distribution)}]");
            }

            // check if iterator is perfect
            int result = CheckPerfect(samples, minimumEvidence, votes, iterator);
            
            // if it was perfect
            if (result != -1)
            {
                // copy it out, mark prediction, and add to perfect features
                ConditionalPrediction perfectFeature = iterator.Copy();
                perfectFeature.prediction = result;
                perfectFeatures.Add(perfectFeature);
                distribution[result]++;
            }

            // iterate
            bool done = iterator.Next(allComponentValues);

            // if we are done
            if (done)
            {
                // break
                break;
            }
        }
        Console.WriteLine();

        if (perfectFeatures.Count == 0)
        {
            throw new Exception("No perfect features found");
        }
    }

    public List<List<float>> GetAllComponentValues(List<Sample> samples)
    {
        List<List<float>> allComponentValues = new List<List<float>>();
        int componentCount = samples[0].input.Count;
        for (int i = 0; i < componentCount; i++)
        {
            HashSet<float> componentValues = new HashSet<float>();
            foreach (Sample sample in samples)
            {
                componentValues.Add(sample.input[i]);
            }
            allComponentValues.Add(componentValues.ToList());
        }
        return allComponentValues;
    }

    public int CheckPerfect(List<Sample> samples, int minimalEvidence, int[] votes, ConditionalPrediction conditionalPrediction)
    {
        Array.Clear(votes);
        int lastVote = -1;
        foreach(Sample sample in samples)
        {
            if (conditionalPrediction.Satisfies(sample.input))
            {
                if (lastVote != -1 && lastVote != sample.output)
                {
                    return -1;
                }
                lastVote = sample.output;
                votes[sample.output]++;
            }
        }
        int classesWithVotes = 0;
        for (int i = 0; i < votes.Length; i++)
        {
            if (votes[i] > 0)
            {
                classesWithVotes++;
            }
        }
        if (classesWithVotes != 1)
        {
            return -1;
        }
        int maxClass = -1;
        int maxVotes = -1;
        for (int i = 0; i < votes.Length; i++)
        {
            if (votes[i] > maxVotes)
            {
                maxVotes = votes[i];
                maxClass = i;
            }
        }
        if (maxVotes >= minimalEvidence)
        {
            return maxClass;
        }
        return -1;
    }

    public int Predict(List<float> input)
    {
        int[] votes = new int[totalClassCount];
        foreach (ConditionalPrediction perfectFeature in perfectFeatures)
        {
            if (perfectFeature.Satisfies(input))
            {
                votes[perfectFeature.prediction]++;
            }
        }
        int maxVotes = 0;
        int maxVotesIndex = -1;
        for (int i = 0; i < votes.Length; i++)
        {
            if (votes[i] > maxVotes)
            {
                maxVotes = votes[i];
                maxVotesIndex = i;
            }
        }
        return maxVotesIndex;
    }
}

public class ConditionalPrediction
{
    public List<Condition> conditions;
    public int prediction;

    public ConditionalPrediction(List<Condition> conditions)
    {
        this.conditions = conditions;
        this.prediction = -1;
    }

    public bool Satisfies(List<float> input)
    {
        foreach (Condition condition in conditions)
        {
            if (!condition.Satisfies(input))
            {
                return false;
            }
        }
        return true;
    }

    public ConditionalPrediction Copy()
    {
        List<Condition> conditionsCopy = new List<Condition>();
        foreach (Condition condition in conditions)
        {
            conditionsCopy.Add(condition.Copy());
        }
        return new ConditionalPrediction(conditionsCopy);
    }

    public bool Next(List<List<float>> allComponentValues)
    {
        for (int i = 0; i < conditions.Count; i++)
        {
            if (!conditions[i].Next(allComponentValues))
            {
                return false;
            }
        }
        return true;
    }

    public string GetProgressString(List<List<float>> allComponentValues)
    {
        StringBuilder sb = new StringBuilder();
        foreach(Condition condition in conditions)
        {
            sb.Append(Math.Floor(condition.GetProgress(allComponentValues) * 100) + "%");
            sb.Append(" ");
        }
        return sb.ToString();
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        foreach (Condition condition in conditions)
        {
            sb.Append("(" + condition.ToString() + ")");
        }
        return sb.ToString();
    }
}

public class Condition
{
    public int componentIndex;
    public int componentValueIndex;
    public float componentValue;
    public bool isLTE;

    public Condition(int componentIndex, int componentValueIndex, float componentValue, bool isLTE)
    {
        this.componentIndex = componentIndex;
        this.componentValueIndex = componentValueIndex;
        this.componentValue = componentValue;
        this.isLTE = isLTE;
    }

    public bool Satisfies(List<float> input)
    {
        if (isLTE)
        {
            return input[componentIndex] <= componentValue;
        }
        else
        {
            return input[componentIndex] > componentValue;
        }
    }

    public bool Next(List<List<float>> allComponentValues)
    {
        // if we can move to the next is lte flag
        if (!isLTE)
        {
            // flip the flag, done
            isLTE = true;
            return false;
        }
        // otherwise we are at the last lte flag
        else
        {
            // reset lte flag
            isLTE = false;

            // move to the next component value index
            componentValueIndex++;

            // if we are still within component value index range
            if (componentValueIndex < allComponentValues[componentIndex].Count)
            {
                // set the component value, done
                componentValue = allComponentValues[componentIndex][componentValueIndex];
                return false;
            }
            // otherwise we passed the last component value index
            else
            {
                // reset component value index
                componentValueIndex = 0;

                // move to the next component index
                componentIndex++;

                // if we are still within component index range
                if (componentIndex < allComponentValues.Count)
                {
                    // set the component value, done
                    componentValue = allComponentValues[componentIndex][componentValueIndex];
                    return false;
                }
                // otherwise we passed the last component index
                else
                {
                    // reset index and value (flag is already reset above), and iteration wrapped
                    componentIndex = 0;
                    componentValueIndex = 0;
                    componentValue = allComponentValues[componentIndex][componentValueIndex];
                    return true;
                }
            }
        }
    }

    public Condition Copy()
    {
        return new Condition(componentIndex, componentValueIndex, componentValue, isLTE);
    }

    public float GetProgress(List<List<float>> allComponentValues)
    {
        return (float)componentIndex / (float)allComponentValues.Count;
    }

    public override string ToString()
    {
        return $"[{componentIndex}] {(isLTE ? "<=" : ">")} {componentValue}";
    }
}
