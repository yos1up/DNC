# Implementation of Differentiable Neural Computers (DNC) in Chainer

Differentiable Neural Computers (DNC) is a neural network architecture proposed by DeepMind in [their third paper on Nature](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz).
I have implemented DNC in [Chainer](http://chainer.org/), a flexible framework of neural networks developped by [Preferred Networks](https://www.preferred-networks.jp/en/).

# What is DNC ?
DNC is a newly proposed neural network. In their paper, DNC learns well in several complex tasks, including finding shortest path in a graph and solving a block puzzle game. It is expected to have the capacity to solve complex, structured tasks that are inaccessible to previous neural networks.

DNC consists of a RNN (recurrent neural network) and a "memory matrix", with some heads for reading and writing to it. The RNN can control the heads at will; it can manipulate the heads in a predetermined fashion to read out the content of the memory and write some data to the memory.

In each timestep, a vanilla RNN receives some external input and yields some output (and refreshes its internal state). In contrast, a RNN in a DNC recieves "data read by the read head at the previous timestep" together with external input, and yields "memory manipulation command" in addition to output data. In accordance with this command, the heads are moved, the memory content at the write head is edited, and the memory contents at the read heads are fetched. Fetched data is input to RNN at the next timestep (together with external input data).

A RNN in DNC learns so that it achieves appropriate input-output relationship in the situation that "the memory" --- a convenient tool to compute --- is given to use freely. How the RNN utilizes the tool depends on its learning.

Although "read-write memory" seems to be very special, it can be regarded as a form of internal state of a RNN(*); in other words, DNC is an RNN that has non-trivial internal-state dynamics like LSTM, but the dynamics are very complicated. This memory enables the RNN to perform complicated information processings. Moreover, equipped with "read-write memory", which is fairly convenient for every kind of information processing, I expect that DNC has high versatility --- to perform various types of tasks reasonably well.

Note that their [NTM (Neural Turing Machine)](https://arxiv.org/pdf/1410.5401v2.pdf) proposed in 2014 has similar structure to DNC. The difference between DNC and NTM is that DNC has more reasonable memory heads' movement. (For datails see the Methods in the DNC paper.)

(*): They call the memory as "external". They say that is because "The behaviour of the network is independent of the memory size as long as the memory is not filled to capacity".


# About my code

In my code, a very small-scale DNC learns a very easy "repeat after me" task. It seems to learn correctly without errors, but it does not necessarily mean that this program correctly performs DNC. If you have any comments about my code, please feel free to contact @yos1up (twitter).

The Supplementary Material of their paper is very useful to implement DNC. It contains ALL variables used in the model and ALL equations to construct the computational graph of the model in two pages. Most of the names of the variables shown in my code coincide with that in their paper.
