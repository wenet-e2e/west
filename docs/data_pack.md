## Data Pack

Training large language models (LLMs) is a computationally demanding task.
It requires vast amounts of data, powerful hardware, and clever optimization techniques.
Naive batching can lead to significant padding tokens, especially when dealing with variable-length sequences, which is common in speech data.

Previously, data was typically sorted by length and then assigned to different batches, which can reduce the amount of padding.
Alternatively, dynamic batch sizes were used, adjusting the batch size based on the length of the longest sequence in each batch.
However, dynamic batch sizes can lead to inconsistent computation and memory usage across batches, affecting training stability and efficiency.

Sequence packing is a more efficient way to handle variable-length sequence data and is widely used in the training of text LLMs.
Packed sequences offer an elegant solution.
Instead of padding, we concatenate multiple shorter sequences into a single, longer sequence.
This minimizes wasted compute (through padding tokens).
It also allows us to process more tokens per batch thus reducing training time.

As long as we ensure the model doesn’t attend across sequence boundaries, we can safely pack sequences.
Fortunately, the self-attention mechanism of Transformers naturally supports this.
Moreover, Flash attention already has a very efficient implementation for this.

In WEST, we support batch and sequence packing for training for both pre-training and fine-tuning stages.

We conduct a simple experiment to illustrate the advantages of sequence pack.
We train an ASR model on an RTX 3090 using WEST, with a 40M Conformer as the Speech Encoder,
QWen-1.5B as the LLM, and AIshell-1 as the training data.
We compare three training methods: static batch, dynamic batch, and sequence pack.
In each method, we maximize GPU memory usage without causing OOM,
and compare the time taken to process 10,000 utterances.

| methods       | Best Configuration       | GPU SM utils(%) | time on training 10000 utts |
|---------------|--------------------------|-----------------|-----------------------------|
| static batch  | batch size 32            | 63.05           | 9 min 33s                   |
| dynamic batch | max token in batch 4096  | 71.49           | 6 min 28s                   |
| sequence pack | Pack token 8192          | 73.87           | 3 min 58s                   |

As shown in table above, sequence pack achieves higher GPU utilization and faster training time without causing OOM.
In the experimental task, compared to static batch,
sequence pack improves training speed by 2.4 times.
