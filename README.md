# Deep Mutual Learning Framework

This is a [Hackathon 2021 project](https://garagehackbox.azurewebsites.net/hackathons/2356/projects/107707)

We are building a generic high-performance consensus server and a Python client library for it to
train and prune clones of a neural network in parallel.

We use this paper as an inspiration: [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf)

*If you already have a model with **softmax** output and you've tried to apply **structured pruning** to it - come work with us!*
With a few extra lines of code you may be able to train a much better and compact version of your
neural net.

Our project has three parts:

1. **Consensus Server:** a remote server that receives output vectors from each training host and
   sends back the values of a loss function. We'll implement the server in C++ or Rust, but maybe
   we'll have a small prototype in Python first. 
2. **Client Library:** a small library in Python that will connect to the consensus server, send NN
   output to it and receive back the value of a loss function. It should look like a simple loss
   function that we add to the existing NN training code.
3. **Neural Network:** *bring your own Neural Net!* Ideally, we need an existing Python project
   that trains and tests a neural net. Even better if it has some code for iterative structured
   pruning. And don't forget to have your dataset with you. :)
